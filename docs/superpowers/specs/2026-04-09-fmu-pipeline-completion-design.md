# FMU Pipeline Completion Design

**Date:** 2026-04-09
**Builds on:** 2026-04-08-fmu-system-type-design.md

## Goal

Complete the FMU pipeline with four features: composed MTK+FMU equations, CoSimulation support, FMU event handling, and AD-safe initialization.

## Context

The FMUSystem type and ME pipeline are working end-to-end (Dahlquist, VanDerPol, multiple FMUs, repeated solve). Four gaps remain before the FMU integration is production-ready.

---

## Feature 1: Composed MTK+FMU Equations

### Problem

After `extract_fmu_subsystems` removes FMU subsystems from the system tree, references to FMU variables in MTK equations (e.g., `D(y) ~ -y + fmu.x`) become dangling. The symbolic pipeline can't resolve `fmu.x`.

### Design

Inject FMU states as **temporary parameters** before compilation, then swap them to unknowns during merge.

**Pipeline change in `__mtkcompile` (systems.jl):**

1. `extract_fmu_subsystems` removes FMU subsystems (unchanged)
2. `collect_fmu_variables` collects namespaced FMU data (unchanged)
3. **New step:** Add `fmu_data.unknowns` to parent system's parameter list via `@set sys.ps = vcat(get_ps(sys), fmu_data.unknowns)`
4. `mtkcompile!` compiles — `fmu.x` is a parameter, equations are well-formed
5. `merge_fmu_data` removes FMU states from parameters, adds them as unknowns, adds derivative equations

**Change in `merge_fmu_data` (fmu_compilation.jl):**

Before building `new_ps`, filter out temporarily-injected FMU states:

```julia
fmu_state_set = Set(fmu_data.unknowns)
new_ps = filter(p -> p ∉ fmu_state_set, get_ps(compiled_sys))
new_ps = vcat(new_ps, fmu_data.parameters)
```

### Why this works

Code generation is lazy — it happens at `ODEProblem` time, not during `mtkcompile`. By the time the ODE function is generated, `fmu.x` is correctly classified as an unknown (looked up from `u`, not `p`). The symbolic pipeline only needs to see it as a known quantity during tearing/simplification, which parameters satisfy.

### Test

```julia
@variables y(t)
fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :fmu)
eqs = [D(y) ~ -y + fmu_sys.x]
parent = System(eqs, t; systems = [fmu_sys], name = :sys)
compiled = mtkcompile(parent)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(compiled, [...], (0.0, 1.0); build_initializeprob = false)
sol = solve(prob, Tsit5())
```

---

## Feature 2: CoSimulation FMU Support

### Problem

CS FMUs have their own internal solver. The host (MTK) communicates via periodic `do_step()` calls. The extension has `FMI2CSFunctor`/`FMI3CSFunctor` with `fmiCSInitialize!`/`fmiCSStep!` but they're not wired into the FMUSystem pipeline.

### Design

Use `FMUStepCallback` (already defined in types.jl) for periodic stepping, plus a `SymbolicDiscreteCallback` + `ImperativeAffect` for lifecycle management (same pattern as ME).

**In `FMIComponent` CS path (MTKFMIExt.jl):**

1. Create wrapper object (same as ME)
2. Create `FMUStepCallback(wrapper, communication_step_size)` for periodic stepping
3. Create lifecycle `SymbolicDiscreteCallback` with finalize affect (same as ME)
4. Pass both as `discrete_events` to `FMUSystem{CoSimulation}`

**Callback lowering:**

During problem construction, `FMUStepCallback` is lowered to `PeriodicCallback` via the existing `lower_fmu_step_callback` function in `fmu_codegen.jl`. The lowering step needs to be hooked into the problem construction path — scan events for FMU callback types and lower them before passing to SciML.

**Compilation path (already implemented):**

`merge_fmu_data` for CS adds states as discrete parameters (no derivative equations). FMU outputs become observed variables.

**Stub implementations needed in MTKFMIExt:**

- `fmu_do_step!(wrapper, t, dt)` — calls `FMI.fmi2DoStep`/`FMI.fmi3DoStep!`
- `fmu_read_outputs!(wrapper, integrator)` — reads output values into integrator state

### Test

```julia
fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "Dahlquist.fmu"); type = :CS)
fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :CS, communication_step_size = 0.01, name = :cs)
parent = System(Equation[], t; systems = [fmu_sys], name = :sys)
compiled = mtkcompile(parent)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(compiled, [...], (0.0, 1.0); build_initializeprob = false)
sol = solve(prob, Tsit5())
# Verify exponential decay (less precise than ME due to discrete stepping)
```

---

## Feature 3: FMU Event Handling (BouncingBall)

### Problem

FMU event indicators are opaque zero-crossing functions evaluated internally by the FMU. They can't be expressed as symbolic continuous events. The BouncingBall FMU has 1 event indicator (`z[0] = h`, height) that triggers velocity reversal on ground collision.

### Design

Use `FMUContinuousCallback` (already defined in types.jl) with lowering at problem construction — same infrastructure as CS stepping.

**In `FMIComponent` ME path (MTKFMIExt.jl):**

When `n_event_indicators > 0`:
1. Create `FMUContinuousCallback(wrapper, n_event_indicators)`
2. Store in `continuous_events` on the FMUSystem

**Callback lowering:**

`lower_fmu_continuous_callback` (already in `fmu_codegen.jl`) produces a `VectorContinuousCallback`:
- **Condition:** `fmu_get_event_indicators!(wrapper, out, u, t)` — evaluates FMU zero-crossings
- **Affect:** enter event mode → iterate `fmu_update_discrete_states!` → update continuous states if changed → re-enter continuous time mode

**FMI operation implementations needed in MTKFMIExt:**

- `fmu_get_event_indicators!(wrapper, out, u, t)` → `FMI.fmi3GetEventIndicators!`
- `fmu_enter_event_mode!(wrapper)` → `FMI.fmi3EnterEventMode!`
- `fmu_update_discrete_states!(wrapper)` → `FMI.fmi3UpdateDiscreteStates!`
- `fmu_get_continuous_states!(wrapper, u)` → `FMI.fmi3GetContinuousStates!`
- `fmu_enter_continuous_time_mode!(wrapper)` → `FMI.fmi3EnterContinuousTimeMode!`

**Lowering hook:**

Same mechanism as Feature 2. During problem construction, scan events for `FMUContinuousCallback` and `FMUStepCallback`, lower them to SciML callback types. This is a single lowering pass that handles both.

**BouncingBall event flow:**

1. Solver integrates `D(h) = v`, `D(v) = g`
2. `VectorContinuousCallback` evaluates `z[0] = h` at each step
3. When `h` crosses zero → affect fires
4. Affect: enter event mode → `eventUpdate()` reverses velocity → `valuesOfContinuousStatesChanged = true` → get new states → `u_modified!(integrator, true)` → re-enter continuous time mode
5. Solver continues with reversed velocity

### Test

```julia
fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "BouncingBall.fmu"); type = :ME)
fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :bb)
parent = System(Equation[], t; systems = [fmu_sys], name = :sys)
compiled = mtkcompile(parent)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(compiled, [...], (0.0, 5.0); build_initializeprob = false)
sol = solve(prob, Tsit5())
# Height should never go significantly below zero
# Velocity should reverse at each bounce
# Ball should eventually stop (v_min threshold)
```

---

## Feature 4: AD-Safe Initialization

### Problem

The initialization solver uses AD which fails on FMU wrapper calls. Current workaround: `build_initializeprob=false`.

### Design

Mark FMU states as fully determined during initialization by adding pinning equations.

**In `merge_fmu_data` (fmu_compilation.jl):**

For each FMU state with a known default value, add an initialization equation:

```julia
for fmu in fmu_subsystems
    for state in get_unknowns(fmu)
        ns_state = renamespace(fmu, state)
        if haskey(fmu_data.defaults, ns_state)
            push!(new_init_eqs, ns_state ~ fmu_data.defaults[ns_state])
        end
    end
end
```

These initialization equations tell the init solver that FMU states are pinned to their defaults. The solver only needs to find MTK unknowns — no AD through the FMU wrapper.

**Result:**
- Pure FMU systems: initialization is trivial (all states pinned)
- Mixed MTK+FMU: init solver handles MTK unknowns only, FMU states are fixed
- `build_initializeprob=true` works without AD errors

**Deferred:** Steady-state initialization of coupled MTK+FMU systems (would need derivative-free init solvers).

### Test

```julia
# Same as existing tests but WITHOUT build_initializeprob=false
prob = ODEProblem(compiled, [...], (0.0, 1.0))  # default initialization
sol = solve(prob, Tsit5())
```

---

## Shared Infrastructure: FMU Callback Lowering Pass

Features 2 and 3 both need FMU callback types lowered to SciML callbacks at problem construction time. This is a single mechanism.

**Location:** New function in `fmu_compilation.jl` or `fmu_codegen.jl`.

**Function:** `lower_fmu_callbacks(sys)` — scans the compiled system's events for FMU callback types, calls the appropriate lowering function, and returns standard SciML callbacks.

**Hook point:** During `ODEProblem` or `DEProblem` construction, after the system is compiled but before callbacks are passed to the solver. The FMU subsystem references stored in metadata (`FMUSubsystemsKey`) provide access to wrapper objects needed for lowering.

**Lowering dispatch:**
- `FMUContinuousCallback` → `lower_fmu_continuous_callback` → `VectorContinuousCallback`
- `FMUStepCallback` → `lower_fmu_step_callback` → `PeriodicCallback`
- `SymbolicDiscreteCallback` → passed through unchanged (standard pipeline handles it)

---

## Implementation Order

1. **Feature 1: Composed equations** — pipeline change, enables mixed MTK+FMU models
2. **Feature 4: AD-safe initialization** — small addition to merge_fmu_data, removes workaround
3. **Feature 2: CS support** — extension wiring + callback lowering infrastructure
4. **Feature 3: Event handling** — extension FMI stubs + BouncingBall test (uses same lowering infra from Feature 2)

Features 2 and 3 share the callback lowering infrastructure, so they're ordered to build it incrementally.
