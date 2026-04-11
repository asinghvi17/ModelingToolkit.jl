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

---

## Post-Implementation Notes

All four features shipped on the `as/fmicomponent` branch. The implementation follows this spec very closely; the deviations below are all minor and captured for future readers.

### Feature 1 — Composed MTK+FMU equations (commit `86d853cd`)

Implemented exactly as designed. `__mtkcompile` in `src/systems/systems.jl` injects `fmu_data.unknowns` into `get_ps(sys)` immediately after `collect_fmu_variables` and before `extract_top_level_statemachines`. `merge_fmu_data` in `src/systems/fmu_compilation.jl` filters them back out before appending real FMU parameters.

### Feature 2 — CoSimulation support (commit `328893cd`)

Implemented as designed. The CS branch of `FMIComponent` in `ext/MTKFMIExt.jl` (around line 275) constructs an `ImperativeAffect`-based initialize callback using `fmiCSInitialize!`, a lifecycle `SymbolicDiscreteCallback`, and an `FMUStepCallback(wrapper_obj, communication_step_size)`. Both callbacks are passed via `discrete_events = Any[lifecycle_cb, step_cb]` to the `FMUSystem{CoSimulation}` constructor.

`fmu_do_step!` and `fmu_read_outputs!` are implemented at the bottom of `ext/MTKFMIExt.jl` (around lines 1058–1095) for both `FMI2InstanceWrapper` and `FMI3InstanceWrapper`.

### Feature 3 — Event handling / BouncingBall (commit `b868bed0`)

Implemented as designed. In the ME branch of `FMIComponent` (around line 251), when `caps.n_event_indicators > 0` an `FMUContinuousCallback(wrapper_obj, caps.n_event_indicators)` is pushed into `cont_events = Any[]` and passed as `continuous_events = cont_events` to the `FMUSystem` constructor.

All six FMI event operations (`fmu_get_event_indicators!`, `fmu_enter_event_mode!`, `fmu_update_discrete_states!`, `fmu_get_continuous_states!`, `fmu_enter_continuous_time_mode!`, plus the FMI 2 / FMI 3 pair for each) are at the bottom of `ext/MTKFMIExt.jl` (around lines 1097–1168).

One tweak not explicitly called out in the spec: `partiallyCompleteIntegratorStep` (around line 826) had its `@assert enterEventMode[] == FMI.fmi3False` removed because, for FMUs with event indicators, the FMI library can legitimately request an event-mode transition from inside the integrator-step completion — this is now handled by the `VectorContinuousCallback` instead of being an assertion failure. Only the `terminateSimulation[]` assertion remains.

### Feature 4 — AD-safe initialization (commit `e151b40f`)

Implemented as designed in `merge_fmu_data`. For each `FMUSystem{ModelExchange}` in the subsystem list, any state that appears in `get_default_values(fmu)` gets a pinning equation `renamespace(fmu, state) ~ fmu_defaults[state]` appended to `initialization_eqs`. The CS path intentionally does not emit these (CS states are parameters, not unknowns, so they don't enter the init solver).

### Shared infrastructure — Callback lowering pass (commit `0453cd00`)

The spec left the exact mechanism open ("new function in `fmu_compilation.jl` or `fmu_codegen.jl` ... hook into `ODEProblem` construction"). The implementation chose a metadata-injection approach:

1. **`RawCallbacksKey` lives in MTKBase**, not MTK. It is defined in `lib/ModelingToolkitBase/src/systems/callbacks.jl` directly above `process_events` (around line 1525). This is the opposite of an earlier working hypothesis; defining it in MTKBase keeps the consumer self-contained and avoids the "type only needs to exist at runtime" hack.
2. **`merge_fmu_data` partitions FMU events** into opaque (`FMUStepCallback`, `FMUContinuousCallback`) vs symbolic (`SymbolicDiscreteCallback`, etc.). Opaque ones are lowered immediately via `lower_fmu_step_callback` / `lower_fmu_continuous_callback` and the resulting SciML callbacks are bundled into a `CallbackSet` (or left as a single callback if there's only one) and stored on the compiled system's metadata under `MTKBase.RawCallbacksKey`. Symbolic ones are merged into `compiled_sys.discrete_events` as normal.
3. **`process_events` picks them up** during problem construction: after generating the symbolic callback set, it checks `SU.hasmetadata(sys, RawCallbacksKey)` and, if present, merges the stored `CallbackSet` into `cb` before assembling the final return value. It also returns `CallbackSet(cb, discrete_cbs...)` instead of the pre-existing `CallbackSet(contin_cbs, discrete_cbs...)` — the latter was a pre-existing bug that silently dropped any callbacks merged into `cb` via `merge_cb`, and the fix was necessary for the `RawCallbacksKey` merge (and user-supplied `callback=`) to take effect at all.
4. **`namespace_callback` no-ops** for `FMUContinuousCallback`, `FMUTimeCallback`, `FMUStepCallback`, and `FMUStepEventCallback` live in `lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl` (around line 276), so that the generic system-tree namespacing in `collect_fmu_variables` passes them through unchanged.

The result is that there is no separate `lower_fmu_callbacks(sys)` function as the spec suggested — the lowering happens eagerly inside `merge_fmu_data` and the already-lowered SciML callbacks are smuggled through to `process_events` via metadata. Functionally this matches the spec ("scan events for FMU callback types, call the appropriate lowering function, return standard SciML callbacks") but the dispatch point moved from problem construction to compilation.

### Not in the original spec — Reference FMU hosting for CI (commit `66ddd02d`)

The spec did not address how CI would obtain Reference FMUs. In practice:

- `test/fmi/fmu_events.jl` now requires `ENV["REFERENCE_FMUS_DIR"]` to be set and logs a skip message otherwise. There is no local-path fallback.
- `.github/workflows/Tests.yml` gained a "Download Reference FMUs" step (gated on `contains(matrix.pkggroup, 'FMI')`) that `curl`s `https://github.com/modelica/Reference-FMUs/releases/download/v0.0.39/Reference-FMUs-0.0.39.zip`, unpacks the `3.0/` subtree, and passes `REFERENCE_FMUS_DIR=${{ github.workspace }}/Reference-FMUs/3.0` into the FMI test job.
