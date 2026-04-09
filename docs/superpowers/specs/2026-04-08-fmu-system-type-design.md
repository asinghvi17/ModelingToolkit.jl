# FMUSystem Type Design

## Problem

The current `MTKFMIExt` extension wraps FMUs as regular `System` objects. This is
fundamentally wrong: a `System` is a declarative, stateless set of symbolic equations,
while an FMU is an imperative, stateful black box. The current approach:

- Fakes equations by wrapping the FMU callable in symbolic expressions
- Hides mutable FMU state inside registered callables where MTK can't see it
- Cannot support FMU events (explicitly asserted away)
- Cannot support FMU time-stepping influence (`nextEventTime`, `CompletedIntegratorStep`)
- Silently corrupts FMU state on solver step rejection
- Provides no FMU instance lifecycle management beyond lazy creation and a finalize callback

## Solution

Introduce `FMUSystem{Mode}`, a new `AbstractSystem` subtype purpose-built for FMUs. It
stores FMU-native data (wrapper, capabilities, value references) alongside symbolic
variables for composition. It does not carry equations — the FMU *is* the dynamics. The
`mtkcompile` pass lowers FMU subsystems into opaque callables and SciML callbacks,
producing a standard compiled `System` that the existing solver pipeline handles.

## Type Hierarchy

```
AbstractSystem (existing)
├── IntermediateDeprecationSystem (existing)
│   ├── System{SubSys} (existing, gains type parameter)
│   └── PDESystem (existing)
└── AbstractFMUSystem (new, abstract)
    └── FMUSystem{Mode, WrapperType, ValRefType} (new, concrete)
```

`AbstractFMUSystem` provides a dispatch target for "any FMU system" regardless of mode.
`FMUSystem` is parameterized on:

- `Mode <: FMUMode` — `ModelExchange` or `CoSimulation`
- `WrapperType` — `FMI2InstanceWrapper` or `FMI3InstanceWrapper` (inferred at construction)
- `ValRefType` — `fmi2ValueReference` or `fmi3ValueReference` (inferred at construction)

Users write `FMUSystem{ME}(; fmu, name)` — `WrapperType` and `ValRefType` are inferred.

## Mode Tags

```julia
abstract type FMUMode end
struct ModelExchange <: FMUMode end
struct CoSimulation <: FMUMode end
```

`ME` and `CS` are const aliases. Scheduled Execution (FMI 3.0) is deferred — the type
parameter means `FMUSystem{SE}` slots in later without breaking changes.

## FMU Capabilities

Extracted from `modelDescription.xml` at construction time.

```julia
struct FMUCapabilities
    can_get_and_set_fmu_state::Bool   # checkpoint/restore support
    has_event_mode::Bool              # FMI 3.0 CS only
    n_event_indicators::Int           # number of zero-crossing functions
    fmi_version::Int                  # 2 or 3
end
```

## FMU Callback Types

> **Post-implementation note:** The four custom callback types below (`FMUContinuousCallback`,
> `FMUTimeCallback`, `FMUStepCallback`, `FMUStepEventCallback`) still exist in `types.jl`
> but are **currently unused**. During implementation, the lowering pipeline that would
> convert these to SciML callbacks was dropped in favor of using standard
> `SymbolicDiscreteCallback` with `ImperativeAffect` for lifecycle and stepping callbacks
> (the same approach the old extension used). FMU callbacks are inherently non-symbolic —
> routing them through a custom symbolic callback pipeline was over-engineered.
>
> These types may become useful later for FMI event indicator support (opaque zero-crossings
> that cannot be expressed as symbolic equations), but are not part of the current pipeline.
> The extension builds callbacks directly and passes them as `discrete_events` /
> `continuous_events` keyword arguments to the `FMUSystem` constructor.

These are stored on the `FMUSystem` and returned by `get_continuous_events` /
`get_discrete_events`. They are lowered to standard SciML callbacks during `mtkcompile`.

```julia
# Lowered to VectorContinuousCallback
# Condition: calls fmi2GetEventIndicators, returns z[j]
# Affect: EnterEventMode → iterate UpdateDiscreteStates → EnterContinuousTimeMode
struct FMUContinuousCallback{W}
    wrapper::W
    n_event_indicators::Int
end

# Lowered to DiscreteCallback managing nextEventTime → add_tstop!
struct FMUTimeCallback{W}
    wrapper::W
end

# Lowered to PeriodicCallback for CS communication stepping
struct FMUStepCallback{W}
    wrapper::W
    communication_step_size::Float64
end

# Lowered to DiscreteCallback checking CompletedIntegratorStep → enterEventMode
struct FMUStepEventCallback{W}
    wrapper::W
end
```

## FMUSystem Fields

> **Post-implementation note:** The constructor was simplified. Auto-callback creation
> (from capabilities) was removed. The constructor now accepts `continuous_events` and
> `discrete_events` as keyword arguments (default empty). The extension is responsible
> for building appropriate callbacks (using `SymbolicDiscreteCallback` with
> `ImperativeAffect`) and passing them in. The event field types were also generalized
> to `Vector{Any}` to accept standard symbolic callbacks rather than the custom FMU
> callback types.

```julia
struct FMUSystem{Mode <: FMUMode, WrapperType, ValRefType} <: AbstractFMUSystem
    # Identity
    name::Symbol
    iv::Sym                                    # time variable

    # Symbolic interface (populated at construction from modelDescription.xml)
    states::Vector{Sym}                        # ME: continuous, CS: discrete
    derivatives::Vector{Sym}                   # ME only, empty for CS
    inputs::Vector{Sym}
    outputs::Vector{Sym}
    parameters::Vector{Sym}
    observed::Vector{Equation}                 # alias equations for multi-named FMU vars

    # FMU-native data
    wrapper::WrapperType
    capabilities::FMUCapabilities
    value_references::Dict{Sym, ValRefType}    # symbolic var → FMI value reference
    default_values::Dict{Sym, Any}             # parameter/state defaults from FMU

    # Events (passed in by caller, not auto-generated)
    continuous_events::Vector{Any}
    discrete_events::Vector{Any}

    # CS-specific
    communication_step_size::Union{Float64, Nothing}  # nothing for ME

    # Composition
    systems::Vector{AbstractSystem}
    parent::Any
    complete::Bool
end
```

## Change to System: Type Parameter for Subsystems

The `System` struct's `systems` field changes from `Vector{System}` to
`Vector{SubSys}` via a type parameter:

```julia
struct System{SubSys <: AbstractSystem} <: IntermediateDeprecationSystem
    # ... existing fields ...
    systems::Vector{SubSys}          # was Vector{System}
    # ...
end
```

- Default: `System{System}` — backward compatible, fully concrete, no performance loss
- With FMU subsystem: `System{AbstractSystem}` or `System{Union{System, FMUSystem{...}}}`
- Existing dispatch: `f(::System)` matches any `System{T}` since unparameterized
  `System` is the union of all `System{T}`

Approximately 15 locations in the codebase that create `System[]` vectors or have
`::System` / `::Vector{System}` type assertions need updating. Key locations:

- `system.jl` line 162: field definition
- `systemstructure.jl` lines 10-44: `extract_top_level_statemachines`
- `connectors.jl` lines 733, 826, 966, 1097: type assertions in connection handling
- `connectors.jl` lines 26, 66: `connect` / `domain_connect`

## AbstractSystem Interface Implementation

### Direct field access

| Method | Returns |
|---|---|
| `get_iv(sys::FMUSystem)` | `sys.iv` |
| `get_unknowns(sys::FMUSystem)` | `sys.states` |
| `get_ps(sys::FMUSystem)` | `sys.parameters` |
| `get_observed(sys::FMUSystem)` | `sys.observed` |
| `get_systems(sys::FMUSystem)` | `sys.systems` |
| `get_inputs(sys::FMUSystem)` | `sys.inputs` |
| `get_outputs(sys::FMUSystem)` | `sys.outputs` |
| `get_continuous_events(sys::FMUSystem)` | `sys.continuous_events` |
| `get_discrete_events(sys::FMUSystem)` | `sys.discrete_events` |
| `nameof(sys::FMUSystem)` | `sys.name` |
| `iscomplete(sys::FMUSystem)` | `sys.complete` |

### Constants / not-applicable

| Method | Returns |
|---|---|
| `is_time_dependent(::FMUSystem)` | `true` |
| `has_iv(::FMUSystem)` | `true` |
| `has_ps(::FMUSystem)` | `true` |
| `has_observed(::FMUSystem)` | `true` |
| `does_namespacing(::FMUSystem)` | `true` |
| `isscheduled(::FMUSystem)` | `false` |
| `get_eqs(::FMUSystem)` | `Equation[]` |
| `get_initialization_eqs(::FMUSystem)` | `Equation[]` |
| `get_noise_eqs(::FMUSystem)` | `nothing` |
| `get_jumps(::FMUSystem)` | `[]` |
| `get_brownians(::FMUSystem)` | `[]` |
| `get_poissonians(::FMUSystem)` | `[]` |
| `get_costs(::FMUSystem)` | `[]` |
| `get_constraints(::FMUSystem)` | `[]` |

### Composition

`getproperty(::AbstractSystem, ::Symbol)` works generically — it searches subsystems,
unknowns, parameters, then observed. Since `FMUSystem` implements the getters above,
`sys.mass__s`, `sys.spring__stiffness`, etc. work with no custom `getproperty`.

`compose(parent, fmu_subsystem)` works because the parent `System`'s `systems` field
now accepts `<:AbstractSystem`. Connection equations live on the parent, not the FMU.

## Construction and Validation

Validation happens at construction time, not at `mtkcompile` time.

### `FMUSystem{ME}` construction

> **Post-implementation note:** Steps 9-11 (auto-creating FMU callback types) were removed
> from the constructor. The extension now builds standard `SymbolicDiscreteCallback` with
> `ImperativeAffect` for lifecycle management and passes them as `discrete_events`. The
> extension also uses `@register_array_symbolic` on the wrapper type to create a symbolic
> callable, stores it as a parameter, and builds observed equations
> `deriv_var ~ wrapper(states, inputs, params, t)[i]` that get namespaced and merged
> automatically by `collect_fmu_variables`/`merge_fmu_data`.

1. Verify FMU declares Model Exchange support
2. Extract `modelDescription.xml` metadata: states, derivatives, inputs, outputs, parameters
3. Verify all state derivatives are provided
4. Create symbolic variables for each category
5. Build `value_references` mapping (symbolic var → FMI value reference)
6. Extract `default_values` from FMU start values
7. Extract capabilities (`canGetAndSetFMUstate`, `numberOfEventIndicators`, etc.)
8. Build wrapper struct (do NOT instantiate the FMU instance — that happens at solve time)
9. Build observed equations for derivatives via `@register_array_symbolic` wrapper callable
10. Build lifecycle `SymbolicDiscreteCallback` with `ImperativeAffect`
11. Pass events as `continuous_events`/`discrete_events` kwargs to `FMUSystem` constructor

### `FMUSystem{CS}` construction

> **Post-implementation note:** CS FMU support is deferred. The extension creates an
> `FMUSystem{CoSimulation}` but does not yet build stepping callbacks. Only the ME path
> is fully implemented.

Same as ME, plus:

1. Verify FMU declares Co-Simulation support
2. Require `communication_step_size` parameter
3. Create stepping callbacks and pass as `discrete_events`
4. For FMI 3.0 with `hasEventMode`: create event indicator callbacks if `n_event_indicators > 0`
5. Create states as discrete variables (not continuous)

## mtkcompile Compilation Pass

### Before mtkcompile

Mixed hierarchy: `System` containing `FMUSystem` subsystems (or standalone `FMUSystem`).

### After mtkcompile

A single compiled `System` with FMU calls embedded as opaque callables and FMU events
lowered to standard SciML callbacks. The existing `ODEFunction`/`ODEProblem` pipeline
handles it unchanged.

### Pass behavior for System subsystems

No change from today: collect equations, unknowns, parameters; run alias elimination,
tearing, index reduction.

### Pass behavior for FMUSystem subsystems

1. **Skip symbolic manipulation** — no tearing, no alias elimination on FMU internals
2. **Collect symbolic variables** — states, parameters, observed (aliases) are namespaced
   and merged into the parent
3. **Do NOT collect equations** — there are none
4. **Collect and namespace events** — `continuous_events` and `discrete_events` from FMU
   subsystems are namespaced via `namespace_callback` during collection, then merged into
   the compiled system during `merge_fmu_data`
5. **Mark FMU outputs as irreducible** — tearing cannot eliminate or substitute through them
6. **FMU inputs become algebraic constraints** — satisfied by connection equations on the parent

> **Post-implementation note:** The original `collect_fmu_variables` forgot to namespace
> events. Since FMU subsystems are extracted before the symbolic pipeline, their events
> need explicit namespacing. This was fixed to use `namespace_callback`. The
> `merge_fmu_data` function was also updated to merge events into the compiled system
> (the original implementation omitted this).

### Code generation: ME FMUs

> **Post-implementation note:** The derivative evaluation approach changed. The plan
> originally had `merge_fmu_data` create `D(state) ~ ns_deriv` where `ns_deriv` was an
> undefined variable. The fix: the extension restores `@register_array_symbolic` for the
> wrapper types, creates the wrapper as a symbolic callable parameter, and builds observed
> equations `deriv_var ~ wrapper(states, inputs, params, t)[i]` on the `FMUSystem`. These
> observed equations are namespaced and merged by `collect_fmu_variables`/`merge_fmu_data`
> automatically. The `merge_fmu_data` function then creates `D(ns_state) ~ ns_deriv`
> equations that reference the now-properly-defined namespaced derivative variables.

**RHS contribution:**

```julia
# Observed equations on FMUSystem (built by extension):
fmu₊der_state_i ~ fmu₊wrapper_callable(fmu₊states, fmu₊inputs, fmu₊params, t)[i]

# Derivative equations (built by merge_fmu_data):
D(parent₊fmu₊state_i) ~ parent₊fmu₊der_state_i
```

The wrapper callable is registered via `@register_array_symbolic` in the extension and
stored as a symbolic parameter on the `FMUSystem`.

**Event callbacks lowered to:**

```julia
VectorContinuousCallback(
    (out, u, t, integrator) -> begin
        fmu_set_state!(wrapper, u, t)
        fmi_get_event_indicators!(wrapper, out)
    end,
    (integrator, idx) -> begin
        fmi_enter_event_mode!(wrapper)
        while true
            info = fmi_update_discrete_states!(wrapper)
            info.newDiscreteStatesNeeded || break
            info.terminateSimulation && (terminate!(integrator); return)
        end
        if info.valuesOfContinuousStatesChanged
            fmi_get_continuous_states!(wrapper, integrator.u)
            u_modified!(integrator, true)
        end
        if info.nextEventTimeDefined
            add_tstop!(integrator, info.nextEventTime)
        end
        fmi_enter_continuous_time_mode!(wrapper)
    end,
    n_event_indicators
)
```

**Lifecycle callbacks:**

```julia
# Initialize: instantiate FMU at solve start
initialize = (integrator) -> begin
    instantiate!(wrapper)
    enter_initialization_mode!(wrapper, integrator.u, integrator.p, integrator.t)
    exit_initialization_mode!(wrapper)
    # initial event iteration
    while info.newDiscreteStatesNeeded ...
    enter_continuous_time_mode!(wrapper)
    if info.nextEventTimeDefined
        add_tstop!(integrator, info.nextEventTime)
    end
end

# Finalize: clean up at solve end
finalize = (integrator) -> begin
    terminate_instance!(wrapper)
    free_instance!(wrapper)
end
```

### Code generation: CS FMUs

**No RHS contribution.** CS FMU states are discrete variables.

**Periodic stepping callback:**

```julia
PeriodicCallback(
    (integrator) -> begin
        fmi_do_step!(wrapper, t_prev, communication_step_size)
        # For FMI 3.0: check eventOccurred, handle event mode if needed
        read_fmu_states_and_outputs!(wrapper, integrator)
        u_modified!(integrator, true)
    end,
    communication_step_size
)
```

**Lifecycle callbacks:** Same pattern as ME, but enters Step Mode instead of
Continuous-Time Mode.

### Standalone compilation output

| Input | Output |
|---|---|
| `FMUSystem{ME}` standalone | Compiled `System`: FMU states as unknowns, `fmu_eval` in RHS, all callbacks |
| `FMUSystem{CS}` standalone | Compiled `System`: zero continuous unknowns, FMU discrete vars, stepping callback |
| `System` with `FMUSystem` subsystems | Compiled `System`: FMU calls embedded, all callbacks merged with parent's |

## Step Rejection Handling

Capability-adaptive, based on `FMUCapabilities.can_get_and_set_fmu_state`:

### When FMU supports get/set state (preferred)

After each accepted step, checkpoint the FMU state via `fmi2GetFMUstate`. On step
rejection (solver resets `u` and retries with smaller `dt`), detect the time regression
and restore the FMU state via `fmi2SetFMUstate`.

### When FMU does not support get/set state (fallback)

On step rejection, re-instantiate the FMU from the last accepted `(u, t)`. This is
heavier but always works. The wrapper tracks the last accepted state for this purpose.

### Implementation

A `DiscreteCallback` at every accepted step:

```julia
DiscreteCallback(
    (u, t, integrator) -> true,
    (integrator) -> begin
        if capabilities.can_get_and_set_fmu_state
            save_fmu_state!(wrapper)
        else
            save_accepted_state!(wrapper, integrator.u, integrator.t)
        end
        # Also handle CompletedIntegratorStep
        if completed_integrator_step!(wrapper)
            # enter event mode
        end
    end
)
```

## FMI Event Model Mapping

| FMI Concept | SciML Equivalent |
|---|---|
| Event indicators (zero-crossings) | `VectorContinuousCallback` conditions |
| Time events (`nextEventTime`) | `add_tstop!(integrator, t)` via `DiscreteCallback` |
| Step events (`CompletedIntegratorStep`) | `DiscreteCallback` at each accepted step |
| Event iteration (`newDiscreteStatesNeeded` loop) | Inside callback affect function |
| `valuesOfContinuousStatesChanged` | `u_modified!(integrator, true)` after state read-back |
| `nominalsOfContinuousStatesChanged` | Log warning (solver reconfiguration not straightforward) |
| `terminateSimulation` | `terminate!(integrator)` |

## Time-Stepping Influence

FMUs communicate time-stepping needs via:

1. **`nextEventTime`** — after each event iteration, if `nextEventTimeDefined`, call
   `add_tstop!(integrator, nextEventTime)`. The solver steps exactly to this time.
2. **`CompletedIntegratorStep` returning `enterEventMode`** — checked at each accepted
   step via `DiscreteCallback`. If true, enter event mode.
3. **FMI 3.0 CS early return** — `fmi3DoStep` returns `earlyReturn=true` with
   `lastSuccessfulTime`. Accept the partial step and adjust timing.

All use standard SciML mechanisms (`tstops`, callbacks). No custom step-size control.

## FMU Instance Lifecycle

The FMU instance is tied to the integrator, not the problem:

- **Created** in an `initialize` callback when `solve()` starts
- **Freed** in a `finalize` callback when `solve()` ends (or errors)
- **GC safety net** via Julia finalizer on the wrapper struct
- Re-solving the same problem creates a fresh instance
- Concurrent solves on the same problem get independent instances

## Scope

### In scope

- `FMUSystem{ME}` and `FMUSystem{CS}`
- FMI 2.0 and FMI 3.0
- Event handling (state events, time events, step events)
- Capability-adaptive step rejection handling
- Full composition with MTK systems via `compose`/`connect`
- Instance lifecycle management

### Out of scope (deferred)

- FMI 3.0 Scheduled Execution (`FMUSystem{SE}` — slots in via type parameter later)
- FMI 3.0 Intermediate Update callbacks
- Non-floating-point FMU variables
- Array variable time derivatives
- FMU-to-FMU direct connections without a parent MTK system

## File Layout

| What | Where |
|---|---|
| `AbstractFMUSystem`, `FMUMode`, `ModelExchange`, `CoSimulation` | `lib/ModelingToolkitBase/src/systems/fmu/types.jl` |
| `FMUCapabilities`, `FMUContinuousCallback`, `FMUTimeCallback`, `FMUStepCallback`, `FMUStepEventCallback` | `lib/ModelingToolkitBase/src/systems/fmu/types.jl` |
| `FMUSystem` struct + `AbstractSystem` interface methods | `lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl` |
| `System{SubSys}` type parameter change | `lib/ModelingToolkitBase/src/systems/system.jl` |
| `mtkcompile` FMU pass | `src/systems/fmu_compilation.jl` |
| FMU callback lowering (codegen) | `src/systems/fmu_codegen.jl` |
| FMI2/FMI3 wrapper structs, instance management, `fmu_eval` | `ext/MTKFMIExt.jl` (rewritten) |
| `::System` / `::Vector{System}` assertion updates | `src/systems/systemstructure.jl`, `lib/ModelingToolkitBase/src/systems/connectors.jl` |
