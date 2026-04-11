# FMU Pipeline Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the FMU pipeline with composed MTK+FMU equations, CoSimulation support, FMU event handling (BouncingBall), and AD-safe initialization.

**Architecture:** Four features built incrementally on the existing FMUSystem/mtkcompile pipeline. Composed equations inject FMU states as temporary parameters during compilation. CS support wires existing functors into SymbolicDiscreteCallback. Event handling uses FMUContinuousCallback lowered to VectorContinuousCallback via a metadata-based callback injection in `process_events`. AD-safe init adds pinning equations for FMU states.

**Tech Stack:** Julia, ModelingToolkit.jl, ModelingToolkitBase, FMI.jl/FMIImport.jl, SciMLBase, OrdinaryDiffEq, Reference FMUs (FMI 3.0)

**Spec:** `docs/superpowers/specs/2026-04-09-fmu-pipeline-completion-design.md`

---

## Implementation Status

All six tasks in this plan are complete. Implementation landed on the `as/fmicomponent` branch across the following commits:

| Commit | Task | Summary |
|--------|------|---------|
| `86d853cd` | Task 1 | Composed MTK+FMU equations via temporary parameter injection |
| `e151b40f` | Task 2 | AD-safe FMU initialization via state pinning equations |
| `0453cd00` | Task 3 | Callback lowering infrastructure (`RawCallbacksKey`, `process_events`, `namespace_callback` no-ops) |
| `328893cd` | Task 4 | CoSimulation FMU support with `ImperativeAffect` init + `FMUStepCallback` |
| `b868bed0` | Task 5 | FMU event handling + BouncingBall (`FMUContinuousCallback` in ME path, FMI event op impls) |
| `d9fd63fe` | Task 6 | Cleanup: removed the deferred composed-MTK+FMU TODO |
| `66ddd02d` | CI     | Download Reference FMUs from the modelica/Reference-FMUs v0.0.39 release in CI |

See the "Post-Implementation Notes" section at the end of this file for deviations from the original plan.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/systems/systems.jl` | Modify | Inject FMU states as temp parameters before compilation |
| `src/systems/fmu_compilation.jl` | Modify | Filter temp params in merge, lower FMU callbacks, store in metadata, add init eqs |
| `lib/ModelingToolkitBase/src/systems/callbacks.jl` | Modify | Add `RawCallbacksKey` metadata check in `process_events` |
| `lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl` | Modify | Add `namespace_callback` no-op for FMU callback types |
| `ext/MTKFMIExt.jl` | Modify | Wire CS callbacks, create FMUContinuousCallback for events, implement FMI operation stubs |
| `test/fmi/fmu_events.jl` | Modify | Add composed, CS, BouncingBall, and init tests |

---

### Task 1: Composed MTK+FMU Equations

Inject FMU states as temporary parameters before `mtkcompile!` so MTK equations can reference FMU variables. Remove them during merge when they become proper unknowns.

**Files:**
- Modify: `src/systems/systems.jl:31-33`
- Modify: `src/systems/fmu_compilation.jl:82-131`
- Modify: `test/fmi/fmu_events.jl`

- [x] **Step 1: Write the failing test** _(commit 86d853cd)_

Add this test to `test/fmi/fmu_events.jl`, inside the `"FMU Pipeline - Reference FMUs"` testset (after the "Repeated solve" testset, before the closing `end`):

```julia
@testset "Composed MTK+FMU equations" begin
    fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "Dahlquist.fmu"); type = :ME)
    fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :fmu)

    @variables y(t)
    eqs = [D(y) ~ -y + fmu_sys.x]
    parent = System(eqs, t; systems = [fmu_sys], name = :sys)
    compiled = mtkcompile(parent)

    @test length(unknowns(compiled)) == 2  # y and fmu.x

    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        compiled,
        [compiled.y => 0.0, compiled.fmu.x => 1.0],
        (0.0, 1.0);
        build_initializeprob = false
    )
    sol = solve(prob, Tsit5(); reltol = 1e-8, abstol = 1e-8)
    @test SciMLBase.successful_retcode(sol)

    # fmu.x decays as exp(-t), y is driven by it
    # fmu.x(1) ≈ exp(-1)
    @test sol[compiled.fmu.x][end] ≈ exp(-1.0) atol = 1e-6
end
```

- [x] **Step 2: Inject FMU states as temporary parameters** _(commit 86d853cd)_

In `src/systems/systems.jl`, after line 33 (`fmu_data = ...`), add parameter injection:

```julia
    # Inject FMU states as temporary parameters so MTK equations can reference them
    if fmu_data !== nothing
        existing_ps = get_ps(sys)
        sys = Setfield.@set sys.ps = vcat(existing_ps, fmu_data.unknowns)
    end
```

This goes right before line 35 (`sys, statemachines = extract_top_level_statemachines(sys)`).

- [x] **Step 3: Filter temporary parameters during merge** _(commit 86d853cd)_

In `src/systems/fmu_compilation.jl`, replace the `new_ps` line at the top of `merge_fmu_data` (line 83):

Old:
```julia
    new_ps = vcat(get_ps(compiled_sys), fmu_data.parameters)
```

New:
```julia
    # Filter out FMU states that were temporarily injected as parameters
    fmu_state_set = Set(fmu_data.unknowns)
    compiled_ps = filter(p -> p ∉ fmu_state_set, get_ps(compiled_sys))
    new_ps = vcat(compiled_ps, fmu_data.parameters)
```

- [x] **Step 4: Run test to verify it passes** _(commit 86d853cd)_

Run: `cd /Users/anshul/temp/ModelingToolkit.jl && julia --project -e 'using Pkg; Pkg.activate("test/fmi"); include("test/fmi/fmu_events.jl")'`

The "Composed MTK+FMU equations" test should pass. The `fmu.x` variable should be resolvable during symbolic compilation and produce correct results.

- [x] **Step 5: Commit** _(commit 86d853cd)_

```bash
git add src/systems/systems.jl src/systems/fmu_compilation.jl test/fmi/fmu_events.jl
git commit -m "feat: support composed MTK+FMU equations via temporary parameter injection"
```

---

### Task 2: AD-Safe Initialization

Add initialization equations pinning FMU states to their defaults, so the initialization solver doesn't need to differentiate through FMU wrappers.

**Files:**
- Modify: `src/systems/fmu_compilation.jl:82-131`
- Modify: `test/fmi/fmu_events.jl`

- [x] **Step 1: Write the failing test** _(commit e151b40f)_

Add this test to `test/fmi/fmu_events.jl`, inside the `"FMU Pipeline - Reference FMUs"` testset:

```julia
@testset "AD-safe initialization (no build_initializeprob=false)" begin
    fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "Dahlquist.fmu"); type = :ME)
    fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :dahlquist)

    parent = System(Equation[], t; systems = [fmu_sys], name = :sys)
    compiled = mtkcompile(parent)

    # Should work WITHOUT build_initializeprob=false
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        compiled, [compiled.dahlquist.x => 1.0], (0.0, 1.0)
    )
    sol = solve(prob, Tsit5(); reltol = 1e-8, abstol = 1e-8)
    @test SciMLBase.successful_retcode(sol)
    @test sol[end, end] ≈ exp(-1.0) atol = 1e-6
end
```

- [x] **Step 2: Add initialization equations in merge_fmu_data** _(commit e151b40f)_

In `src/systems/fmu_compilation.jl`, in the `merge_fmu_data` function, after the existing initial_conditions merge block (after line 128), add:

```julia
    # Add initialization equations pinning FMU states to defaults (AD-safe init)
    new_init_eqs = copy(get_initialization_eqs(compiled_sys))
    for fmu in fmu_subsystems
        if fmu isa FMUSystem{ModelExchange}
            fmu_defaults = get_default_values(fmu)
            for state in get_unknowns(fmu)
                ns_state = renamespace(fmu, state)
                if haskey(fmu_defaults, state)
                    push!(new_init_eqs, ns_state ~ fmu_defaults[state])
                end
            end
        end
    end
    if length(new_init_eqs) != length(get_initialization_eqs(compiled_sys))
        @set! compiled_sys.initialization_eqs = new_init_eqs
    end
```

- [x] **Step 3: Run test to verify it passes** _(commit e151b40f)_

Run: `cd /Users/anshul/temp/ModelingToolkit.jl && julia --project -e 'using Pkg; Pkg.activate("test/fmi"); include("test/fmi/fmu_events.jl")'`

The "AD-safe initialization" test should pass — `ODEProblem` construction succeeds without `build_initializeprob=false`.

- [x] **Step 4: Commit** _(commit e151b40f)_

```bash
git add src/systems/fmu_compilation.jl test/fmi/fmu_events.jl
git commit -m "feat: AD-safe FMU initialization via state pinning equations"
```

---

### Task 3: Callback Lowering Infrastructure

Add the `RawCallbacksKey` metadata mechanism so pre-lowered callbacks (from FMU subsystems) are merged into the SciML callback set during problem construction. Also add `namespace_callback` no-ops for FMU callback types.

**Files:**
- Modify: `lib/ModelingToolkitBase/src/systems/callbacks.jl:1524-1529`
- Modify: `src/systems/fmu_compilation.jl`
- Modify: `lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl`

- [x] **Step 1: Define RawCallbacksKey in fmu_compilation.jl** _(commit 0453cd00 — see Post-Implementation Notes: actually defined in MTKBase)_

In `src/systems/fmu_compilation.jl`, after the existing `FMUSubsystemsKey` struct (line 134), add:

```julia

"""Metadata key for storing pre-lowered SciML callbacks on the compiled system."""
struct RawCallbacksKey end
```

- [x] **Step 2: Add namespace_callback no-ops for FMU callback types** _(commit 0453cd00)_

In `lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl`, at the end of the file (after line 295), add:

```julia

# FMU callback types don't have symbolic variables to namespace — pass through as-is
namespace_callback(cb::FMUContinuousCallback, s) = cb
namespace_callback(cb::FMUTimeCallback, s) = cb
namespace_callback(cb::FMUStepCallback, s) = cb
namespace_callback(cb::FMUStepEventCallback, s) = cb
```

- [x] **Step 3: Modify process_events to check for RawCallbacksKey** _(commit 0453cd00)_

In `lib/ModelingToolkitBase/src/systems/callbacks.jl`, replace the `process_events` function (lines 1524-1529):

Old:
```julia
function process_events(sys; callback = nothing, tspan = nothing, kwargs...)
    contin_cbs = generate_continuous_callbacks(sys; kwargs...)
    discrete_cbs = generate_discrete_callbacks(sys; tspan, kwargs...)
    cb = merge_cb(contin_cbs, callback)
    return (discrete_cbs === nothing) ? cb : CallbackSet(contin_cbs, discrete_cbs...)
end
```

New:
```julia
function process_events(sys; callback = nothing, tspan = nothing, kwargs...)
    contin_cbs = generate_continuous_callbacks(sys; kwargs...)
    discrete_cbs = generate_discrete_callbacks(sys; tspan, kwargs...)
    cb = merge_cb(contin_cbs, callback)
    # Merge pre-lowered raw callbacks (e.g., from FMU subsystems)
    if hasmetadata(sys, RawCallbacksKey)
        raw_cbs = getmetadata(sys, RawCallbacksKey, nothing)
        cb = merge_cb(cb, raw_cbs)
    end
    return (discrete_cbs === nothing) ? cb : CallbackSet(cb, discrete_cbs...)
end
```

Note: `RawCallbacksKey` is defined in MTK (not MTKBase), but `getmetadata` uses `DataType` keys from `get_metadata()` which is an `ImmutableDict{DataType, Any}`. The type just needs to exist at runtime. Since MTK loads before problem construction, this works.

- [x] **Step 4: Add FMU callback lowering in merge_fmu_data** _(commit 0453cd00)_

In `src/systems/fmu_compilation.jl`, in `merge_fmu_data`, replace the event merging section (lines 112-119):

Old:
```julia
    # Merge FMU events into compiled system
    if !isempty(fmu_data.continuous_events)
        new_cont = vcat(get_continuous_events(compiled_sys), fmu_data.continuous_events)
        @set! compiled_sys.continuous_events = new_cont
    end
    if !isempty(fmu_data.discrete_events)
        new_disc = vcat(get_discrete_events(compiled_sys), fmu_data.discrete_events)
        @set! compiled_sys.discrete_events = new_disc
    end
```

New:
```julia
    # Separate FMU opaque callbacks from symbolic callbacks, then lower opaque ones
    lowered_cbs = Any[]
    symbolic_disc = Any[]
    for cb in fmu_data.discrete_events
        if cb isa FMUStepCallback
            push!(lowered_cbs, lower_fmu_step_callback(cb))
        else
            push!(symbolic_disc, cb)
        end
    end
    for cb in fmu_data.continuous_events
        if cb isa FMUContinuousCallback
            push!(lowered_cbs, lower_fmu_continuous_callback(cb))
        else
            push!(symbolic_disc, cb)  # SymbolicContinuousCallback passes through
        end
    end

    # Merge symbolic discrete events (e.g., lifecycle callbacks) into the system
    if !isempty(symbolic_disc)
        new_disc = vcat(get_discrete_events(compiled_sys), symbolic_disc)
        @set! compiled_sys.discrete_events = new_disc
    end
```

- [x] **Step 5: Store lowered callbacks in metadata** _(commit 0453cd00)_

Still in `merge_fmu_data`, right after the metadata line (`fmu_metadata = Base.ImmutableDict(...)`), add the lowered callbacks:

Replace the existing metadata line:
```julia
    fmu_metadata = Base.ImmutableDict(get_metadata(compiled_sys),
        FMUSubsystemsKey => fmu_subsystems)
```

With:
```julia
    fmu_metadata = Base.ImmutableDict(get_metadata(compiled_sys),
        FMUSubsystemsKey => fmu_subsystems)
    if !isempty(lowered_cbs)
        raw_cb_set = length(lowered_cbs) == 1 ? lowered_cbs[1] :
                     SciMLBase.CallbackSet(lowered_cbs...)
        fmu_metadata = Base.ImmutableDict(fmu_metadata, RawCallbacksKey => raw_cb_set)
    end
```

- [x] **Step 6: Commit** _(commit 0453cd00)_

```bash
git add src/systems/fmu_compilation.jl lib/ModelingToolkitBase/src/systems/callbacks.jl lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl
git commit -m "feat: add callback lowering infrastructure for FMU opaque callbacks"
```

---

### Task 4: CoSimulation FMU Support

Wire the existing CS functors (`fmiCSInitialize!`, `fmiCSStep!`) into `SymbolicDiscreteCallback` + `ImperativeAffect` in the `FMIComponent` CS path. Add `FMUStepCallback` for periodic stepping.

**Files:**
- Modify: `ext/MTKFMIExt.jl:269-290`
- Modify: `test/fmi/fmu_events.jl`

- [x] **Step 1: Write the failing test** _(commit 328893cd)_

Add this test to `test/fmi/fmu_events.jl`, inside the `"FMU Pipeline - Reference FMUs"` testset:

```julia
@testset "CoSimulation FMU (Dahlquist)" begin
    fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "Dahlquist.fmu"); type = :CS)
    fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :CS,
        communication_step_size = 0.01, name = :cs)

    @test fmu_sys isa MTKBase.FMUSystem{MTKBase.CoSimulation}
    @test MTKBase.get_communication_step_size(fmu_sys) == 0.01

    parent = System(Equation[], t; systems = [fmu_sys], name = :sys)
    compiled = mtkcompile(parent)

    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        compiled, [], (0.0, 1.0);
        build_initializeprob = false
    )
    sol = solve(prob, Tsit5())
    @test SciMLBase.successful_retcode(sol)
end
```

- [x] **Step 2: Wire CS callbacks in FMIComponent** _(commit 328893cd)_

In `ext/MTKFMIExt.jl`, replace the CS branch of `FMIComponent` (lines 269-290):

```julia
    elseif type == :CS
        # CS FMU: periodic stepping via FMUStepCallback + lifecycle finalize
        cs_functor = Ver == 2 ? FMI2CSFunctor(state_value_references, output_value_references) :
                                FMI3CSFunctor(state_value_references, output_value_references)

        # Initialize callback: fires once at t₀ to instantiate the FMU
        init_affect = MTK.ImperativeAffect(
            fmiCSInitialize!;
            observed = (; wrapper = wrapper_param, inputs = __mtk_internal_x,
                         params = __mtk_internal_p, t = t),
            modified = (; states = __mtk_internal_u, outputs = outputs),
            ctx = cs_functor
        )

        # Step callback: periodic communication with the FMU
        step_affect = MTK.ImperativeAffect(
            fmiCSStep!;
            observed = (; wrapper = wrapper_param, inputs = __mtk_internal_x,
                         params = __mtk_internal_p, t = t, dt = communication_step_size),
            modified = (; states = __mtk_internal_u, outputs = outputs),
            ctx = cs_functor
        )

        # Lifecycle finalize
        finalize_affect = MTK.ImperativeAffect(fmiFinalize!; observed = (; wrapper = wrapper_param))

        # Build symbolic input vectors for CS (same variable parsing as ME)
        __mtk_internal_u = copy(diffvars)
        __mtk_internal_x = isempty(inputs) ? Float64[] : copy(inputs)
        __mtk_internal_p = isempty(params) ? Float64[] : copy(params)

        # Step callback type for lowering to PeriodicCallback
        step_cb = MTK.FMUStepCallback(wrapper_obj, Float64(communication_step_size))

        # Lifecycle callback with init + finalize
        lifecycle_cb = MTK.SymbolicDiscreteCallback(
            (t == t - 1), MTK.ImperativeAffect(Returns((;)));
            initialize = init_affect, finalize = finalize_affect,
            reinitializealg = SciMLBase.NoInit()
        )

        all_observed = observed
        all_params = SymT[MTK.unwrap.(params); MTK.unwrap(wrapper_param)]
        disc_events = Any[lifecycle_cb, step_cb]

        return MTK.FMUSystem{Mode}(;
            name = name,
            iv = t,
            states = Vector{SymT}(MTK.unwrap.(diffvars)),
            derivatives = SymT[],
            inputs = Vector{SymT}(MTK.unwrap.(inputs)),
            outputs = Vector{SymT}(MTK.unwrap.(outputs)),
            parameters = all_params,
            observed = all_observed,
            wrapper = wrapper_obj,
            capabilities = caps,
            value_references = vr_dict,
            default_values = default_dict,
            communication_step_size = Float64(communication_step_size),
            discrete_events = disc_events,
        )
    end
```

Note: The `fmiCSStep!` and `fmiCSInitialize!` functions already exist in the extension (lines 868-921 for FMI2, 960-1015 for FMI3). The `FMUStepCallback` is lowered to `PeriodicCallback` by the infrastructure from Task 3.

- [x] **Step 3: Implement the FMU step callback lowering stubs** _(commit 328893cd)_

The `lower_fmu_step_callback` in `src/systems/fmu_codegen.jl` calls `fmu_do_step!` and `fmu_read_outputs!`. These need implementations in the extension. However, for CS FMUs using the `FMI2CSFunctor`/`FMI3CSFunctor` approach with `ImperativeAffect`, the periodic callback uses the functor directly — so the step callback lowering needs to work with the wrapper.

In `ext/MTKFMIExt.jl`, at the end (before `end # module`), add:

```julia
function MTK.fmu_do_step!(wrapper::FMI2InstanceWrapper, t, dt)
    instance = wrapper.instance
    instance === nothing && error("FMU instance not initialized")
    @statuscheck FMI.fmi2DoStep(instance, t, dt, FMI.fmi2True)
end

function MTK.fmu_do_step!(wrapper::FMI3InstanceWrapper, t, dt)
    instance = wrapper.instance
    instance === nothing && error("FMU instance not initialized")
    eventEncountered = Ref(FMI.fmi3False)
    terminateSimulation = Ref(FMI.fmi3False)
    earlyReturn = Ref(FMI.fmi3False)
    lastSuccessfulTime = Ref(zero(FMI.fmi3Float64))
    @statuscheck FMI.fmi3DoStep!(
        instance, t, dt, FMI.fmi3True, eventEncountered,
        terminateSimulation, earlyReturn, lastSuccessfulTime
    )
end

function MTK.fmu_read_outputs!(wrapper::FMI2InstanceWrapper, integrator)
    instance = wrapper.instance
    instance === nothing && return
    if !isempty(wrapper.output_value_references)
        @statuscheck FMI.fmi2GetReal!(
            instance, wrapper.output_value_references, wrapper.outputs_buffer
        )
    end
end

function MTK.fmu_read_outputs!(wrapper::FMI3InstanceWrapper, integrator)
    instance = wrapper.instance
    instance === nothing && return
    if !isempty(wrapper.output_value_references)
        @statuscheck FMI.fmi3GetFloat64!(
            instance, wrapper.output_value_references, wrapper.outputs_buffer
        )
    end
end
```

- [x] **Step 4: Run test to verify it passes** _(commit 328893cd)_

Run: `cd /Users/anshul/temp/ModelingToolkit.jl && julia --project -e 'using Pkg; Pkg.activate("test/fmi"); include("test/fmi/fmu_events.jl")'`

The "CoSimulation FMU (Dahlquist)" test should pass.

- [x] **Step 5: Commit** _(commit 328893cd)_

```bash
git add ext/MTKFMIExt.jl test/fmi/fmu_events.jl
git commit -m "feat: add CoSimulation FMU support with periodic stepping callbacks"
```

---

### Task 5: FMU Event Handling (BouncingBall)

Create `FMUContinuousCallback` in `FMIComponent` when `n_event_indicators > 0`, and implement the FMI operation stubs for event indicator evaluation and event mode transitions.

**Files:**
- Modify: `ext/MTKFMIExt.jl:228-268` (ME path)
- Modify: `ext/MTKFMIExt.jl` (add FMI operation implementations)
- Modify: `test/fmi/fmu_events.jl`

- [x] **Step 1: Write the failing test** _(commit b868bed0)_

Add this test to `test/fmi/fmu_events.jl`, inside the `"FMU Pipeline - Reference FMUs"` testset:

```julia
@testset "BouncingBall (event indicators)" begin
    fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "BouncingBall.fmu"); type = :ME)
    fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :bb)

    @test fmu_sys isa MTKBase.FMUSystem{MTKBase.ModelExchange}
    @test MTKBase.get_fmu_capabilities(fmu_sys).n_event_indicators == 1

    parent = System(Equation[], t; systems = [fmu_sys], name = :sys)
    compiled = mtkcompile(parent)

    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        compiled,
        [compiled.bb.h => 1.0, compiled.bb.v => 0.0],
        (0.0, 5.0);
        build_initializeprob = false
    )
    sol = solve(prob, Tsit5(); reltol = 1e-8, abstol = 1e-8)
    @test SciMLBase.successful_retcode(sol)

    # Height should never go significantly below zero (bouncing)
    h_vals = sol[compiled.bb.h]
    @test all(h -> h >= -0.01, h_vals)

    # Ball should have bounced (velocity changed sign at least once)
    v_vals = sol[compiled.bb.v]
    sign_changes = count(i -> v_vals[i] * v_vals[i+1] < 0, 1:length(v_vals)-1)
    @test sign_changes >= 2
end
```

- [x] **Step 2: Create FMUContinuousCallback in ME path** _(commit b868bed0)_

In `ext/MTKFMIExt.jl`, in the ME branch of `FMIComponent` (around line 248, after `disc_events = Any[lifecycle_cb]`), add event indicator handling:

```julia
        cont_events = Any[]
        if caps.n_event_indicators > 0
            event_cb = MTK.FMUContinuousCallback(wrapper_obj, caps.n_event_indicators)
            push!(cont_events, event_cb)
        end
```

Then pass `cont_events` to the FMUSystem constructor. Change the `return MTK.FMUSystem{Mode}(;` call to include:

```julia
            continuous_events = cont_events,
```

- [x] **Step 3: Implement FMI event operation stubs** _(commit b868bed0)_

In `ext/MTKFMIExt.jl`, at the end (before `end # module`), add implementations for the event-related stubs:

```julia
# --- FMI Event Operations (used by lower_fmu_continuous_callback) ---

function MTK.fmu_get_event_indicators!(wrapper::FMI3InstanceWrapper, out, u, t)
    instance = wrapper.instance
    instance === nothing && error("FMU instance not initialized for event indicators")
    # Set current state and time
    @statuscheck FMI.fmi3SetTime(instance, t)
    if !isempty(u)
        @statuscheck FMI.fmi3SetContinuousStates(instance, collect(u))
    end
    @statuscheck FMI.fmi3GetEventIndicators!(instance, out)
    return nothing
end

function MTK.fmu_get_event_indicators!(wrapper::FMI2InstanceWrapper, out, u, t)
    instance = wrapper.instance
    instance === nothing && error("FMU instance not initialized for event indicators")
    @statuscheck FMI.fmi2SetTime(instance, t)
    if !isempty(u)
        @statuscheck FMI.fmi2SetContinuousStates(instance, collect(u))
    end
    @statuscheck FMI.fmi2GetEventIndicators!(instance, out)
    return nothing
end

function MTK.fmu_enter_event_mode!(wrapper::FMI3InstanceWrapper)
    @statuscheck FMI.fmi3EnterEventMode(wrapper.instance)
end

function MTK.fmu_enter_event_mode!(wrapper::FMI2InstanceWrapper)
    @statuscheck FMI.fmi2EnterEventMode(wrapper.instance)
end

function MTK.fmu_update_discrete_states!(wrapper::FMI3InstanceWrapper)
    result = FMI.fmi3UpdateDiscreteStates(wrapper.instance)
    # FMI3 returns a tuple: (newDiscreteStatesNeeded, terminateSimulation,
    #   nominalsOfContinuousStatesChanged, valuesOfContinuousStatesChanged,
    #   nextEventTimeDefined, nextEventTime)
    return (
        newDiscreteStatesNeeded = result[1] != FMI.fmi3False,
        terminateSimulation = result[2] != FMI.fmi3False,
        valuesOfContinuousStatesChanged = result[4] != FMI.fmi3False,
        nextEventTimeDefined = result[5] != FMI.fmi3False,
        nextEventTime = result[6]
    )
end

function MTK.fmu_update_discrete_states!(wrapper::FMI2InstanceWrapper)
    eventInfo = FMI.fmi2NewDiscreteStates(wrapper.instance)
    return (
        newDiscreteStatesNeeded = eventInfo.newDiscreteStatesNeeded != FMI.fmi2False,
        terminateSimulation = eventInfo.terminateSimulation != FMI.fmi2False,
        valuesOfContinuousStatesChanged = eventInfo.valuesOfContinuousStatesChanged != FMI.fmi2False,
        nextEventTimeDefined = eventInfo.nextEventTimeDefined != FMI.fmi2False,
        nextEventTime = eventInfo.nextEventTime
    )
end

function MTK.fmu_get_continuous_states!(wrapper::FMI3InstanceWrapper, u)
    @statuscheck FMI.fmi3GetContinuousStates!(wrapper.instance, u)
end

function MTK.fmu_get_continuous_states!(wrapper::FMI2InstanceWrapper, u)
    @statuscheck FMI.fmi2GetContinuousStates!(wrapper.instance, u)
end

function MTK.fmu_enter_continuous_time_mode!(wrapper::FMI3InstanceWrapper)
    @statuscheck FMI.fmi3EnterContinuousTimeMode(wrapper.instance)
end

function MTK.fmu_enter_continuous_time_mode!(wrapper::FMI2InstanceWrapper)
    @statuscheck FMI.fmi2EnterContinuousTimeMode(wrapper.instance)
end
```

- [x] **Step 4: Update ME get_instance to NOT auto-enter continuous time mode for event FMUs** _(commit b868bed0)_

The existing `get_instance_ME!` in `ext/MTKFMIExt.jl` (lines 746-758 for FMI3, 552-563 for FMI2) does initial event iteration and enters continuous time mode. This is fine — the initial event iteration happens once at instantiation. The `VectorContinuousCallback` from the lowering handles subsequent events during integration.

However, the `partiallyCompleteIntegratorStep` call in the callable (line 700) currently asserts `enterEventMode[] == fmi3False`. For FMUs with events, this assertion may fire. Remove the assertion and handle it:

In the callable `(wrapper::Union{FMI2InstanceWrapper, FMI3InstanceWrapper})(...)` (around line 700), the `partiallyCompleteIntegratorStep` call needs to be tolerant. For FMI3, change `partiallyCompleteIntegratorStep` (lines 787-795):

```julia
function partiallyCompleteIntegratorStep(wrapper::FMI3InstanceWrapper)
    enterEventMode = Ref(FMI.fmi3False)
    terminateSimulation = Ref(FMI.fmi3False)
    @statuscheck FMI.fmi3CompletedIntegratorStep!(
        wrapper.instance, FMI.fmi3False, enterEventMode, terminateSimulation
    )
    # Event mode entry is handled by VectorContinuousCallback, not here
    return @assert terminateSimulation[] == FMI.fmi3False
end
```

- [x] **Step 5: Run test to verify it passes** _(commit b868bed0)_

Run: `cd /Users/anshul/temp/ModelingToolkit.jl && julia --project -e 'using Pkg; Pkg.activate("test/fmi"); include("test/fmi/fmu_events.jl")'`

The "BouncingBall (event indicators)" test should pass — the ball bounces, height stays non-negative, velocity reverses.

- [x] **Step 6: Commit** _(commit b868bed0)_

```bash
git add ext/MTKFMIExt.jl test/fmi/fmu_events.jl
git commit -m "feat: add FMU event handling with BouncingBall support"
```

---

### Task 6: Integration Test Cleanup and Verification

Run all FMU tests together, verify existing tests still pass, and clean up the test file.

**Files:**
- Modify: `test/fmi/fmu_events.jl`

- [x] **Step 1: Verify all tests pass together** _(commit d9fd63fe)_

Run all FMU tests:
```bash
cd /Users/anshul/temp/ModelingToolkit.jl && julia --project -e 'using Pkg; Pkg.activate("test/fmi"); include("test/fmi/fmu_events.jl")'
```

All testsets should pass:
- Dahlquist (dx/dt = -kx)
- VanDerPol oscillator
- Multiple FMU subsystems
- Repeated solve
- Composed MTK+FMU equations
- AD-safe initialization
- CoSimulation FMU (Dahlquist)
- BouncingBall (event indicators)

- [x] **Step 2: Remove the composed MTK+FMU TODO comment** _(commit d9fd63fe)_

In `test/fmi/fmu_events.jl`, remove the TODO comment that was deferring composed tests:

```julia
    # TODO: Composed MTK+FMU test (FMU variables referenced in MTK equations
    # need additional integration work — the variable reference fmu.x is not
    # available during the symbolic compilation pass before FMU merge)
```

- [x] **Step 3: Commit** _(commit d9fd63fe)_

```bash
git add test/fmi/fmu_events.jl
git commit -m "test: verify all FMU pipeline tests pass, remove deferred TODO"
```

---

## Post-Implementation Notes

The implementation matches the plan very closely. The divergences below are all minor and were resolved while executing the plan.

### Task 3 — `RawCallbacksKey` lives in MTKBase, not MTK

The plan's Step 1 (and the accompanying note after Step 3) said `RawCallbacksKey` would be defined in MTK's `src/systems/fmu_compilation.jl` and referenced from MTKBase via "the type just needs to exist at runtime". In practice this was reversed: the struct was defined in `lib/ModelingToolkitBase/src/systems/callbacks.jl` (right above `process_events`, around line 1525) because that's where the only consumer lives and it keeps MTKBase self-contained.

As a consequence, `src/systems/fmu_compilation.jl` stores the callback set under `MTKBase.RawCallbacksKey` (not a local key), and the `FMUSubsystemsKey` struct is the only metadata key actually defined in `fmu_compilation.jl`.

### Task 3 — `process_events` uses the `SU` alias

The plan showed `hasmetadata(sys, RawCallbacksKey)` / `getmetadata(sys, RawCallbacksKey, nothing)`. The actual implementation uses the `SU.` prefix (`SU.hasmetadata` / `SU.getmetadata`) which is the SymbolicUtils alias used throughout MTKBase — this is a purely stylistic change needed because the unqualified functions aren't in scope at that call site.

### Task 3 — `process_events` pre-existing bug fix

The old `process_events` returned `CallbackSet(contin_cbs, discrete_cbs...)` when discrete callbacks were present, which silently dropped any callbacks merged into `cb` via `merge_cb`. The new implementation returns `CallbackSet(cb, discrete_cbs...)`, picking up both the user-supplied `callback` and the `RawCallbacksKey` raw callbacks. This fix was necessary for the `RawCallbacksKey` merge to take effect at all — the plan's "New" snippet happens to show the correct form, but the plan didn't call out that this was also fixing an existing bug.

### Task 3 — `namespace_callback` no-ops placement

The no-ops were added around line 276 of `lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl` (just after the `get_*` no-op accessors, before the `# ---- FMU-specific accessors ----` block), not strictly "at the end of the file" as the plan suggested. Functionally identical.

### Task 5 — ME path variable naming

The plan said to add `cont_events = Any[]` around line 248. In the actual code it sits just above `disc_events = Any[lifecycle_cb]` (around line 251) and is passed as `continuous_events = cont_events` in the `FMUSystem` constructor call — matching the plan's intent.

### Task 5 — `partiallyCompleteIntegratorStep` assertion removal

The plan asked to "remove the assertion" on `enterEventMode[] == fmi3False` and replace it with a comment. The actual implementation does exactly that, keeping only the `terminateSimulation[]` assertion (see `ext/MTKFMIExt.jl` around line 826).

### Not in the original plan — CI Reference FMU download

The plan did not cover CI infrastructure. During execution it became clear the Reference FMUs aren't vendored, so `test/fmi/fmu_events.jl` was updated to require the `REFERENCE_FMUS_DIR` env var explicitly (no local-path fallback) and `.github/workflows/Tests.yml` grew a "Download Reference FMUs" step (commit `66ddd02d`) that fetches pre-built FMUs from
`https://github.com/modelica/Reference-FMUs/releases/download/v0.0.39/Reference-FMUs-0.0.39.zip`
and sets `REFERENCE_FMUS_DIR` to the unpacked `3.0/` directory for FMI-tagged test jobs.
