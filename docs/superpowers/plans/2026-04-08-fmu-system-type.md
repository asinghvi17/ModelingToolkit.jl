# FMUSystem Type Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `FMUSystem{Mode}` as a first-class `AbstractSystem` subtype that properly represents FMUs, replacing the current hack-based approach in `MTKFMIExt`.

**Architecture:** New types (`AbstractFMUSystem`, `FMUSystem`) live in `ModelingToolkitBase` so the base system knows about them. `System` gains a type parameter on its `systems` field to hold heterogeneous subsystems. The `mtkcompile` pipeline is extended to extract FMU subsystems, skip symbolic manipulation on them, and lower FMU-specific callbacks to standard SciML callbacks. The actual FMI library interaction remains in the `MTKFMIExt` extension.

**Tech Stack:** Julia, ModelingToolkit.jl, Symbolics.jl, SciMLBase.jl, FMI.jl/FMIImport.jl, DiffEqCallbacks.jl

---

## File Structure

### New files

| File | Responsibility |
|---|---|
| `lib/ModelingToolkitBase/src/systems/fmu/types.jl` | `FMUMode`, `ModelExchange`, `CoSimulation`, `ME`/`CS` aliases, `FMUCapabilities`, `AbstractFMUSystem`, FMU callback types |
| `lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl` | `FMUSystem` struct, constructor, `AbstractSystem` interface methods |
| `src/systems/fmu_compilation.jl` | `mtkcompile` pass for extracting/handling FMU subsystems |
| `src/systems/fmu_codegen.jl` | Lowering FMU callback types to SciML callbacks |
| `test/fmi/fmusystem_unit.jl` | Unit tests for FMUSystem type and interface |
| `test/fmi/fmu_compilation.jl` | Unit tests for the compilation pass |

### Modified files

| File | Change |
|---|---|
| `lib/ModelingToolkitBase/src/systems/system.jl` | Add type parameter `SubSys` to `System`; update constructors |
| `lib/ModelingToolkitBase/src/systems/abstractsystem.jl` | `is_time_dependent` for `AbstractFMUSystem` |
| `lib/ModelingToolkitBase/src/systems/connectors.jl` | Replace `System[]` with `AbstractSystem[]` in ~7 locations |
| `src/systems/systemstructure.jl` | Replace `System[]` with `AbstractSystem[]` in `extract_top_level_statemachines` |
| `src/systems/systems.jl` | Hook FMU pass into `__mtkcompile` |
| `lib/ModelingToolkitBase/src/ModelingToolkitBase.jl` | Include new FMU files, export types |
| `src/ModelingToolkit.jl` | Include new compilation/codegen files |
| `ext/MTKFMIExt.jl` | Rewrite to construct `FMUSystem` instead of `System` |
| `test/runtests.jl` | Add new FMU unit test groups |

---

### Task 1: FMU Foundation Types

**Files:**
- Create: `lib/ModelingToolkitBase/src/systems/fmu/types.jl`
- Test: `test/fmi/fmusystem_unit.jl`

- [ ] **Step 1: Create the FMU types file with mode tags and capabilities**

Create the directory and file:

```julia
# lib/ModelingToolkitBase/src/systems/fmu/types.jl

# ---- Mode tags ----

"""
Abstract supertype for FMU mode tags. Subtypes: `ModelExchange`, `CoSimulation`.
"""
abstract type FMUMode end

"""
FMU Model Exchange mode. The FMU provides derivatives; the external solver integrates.
"""
struct ModelExchange <: FMUMode end

"""
FMU Co-Simulation mode. The FMU has its own internal solver.
"""
struct CoSimulation <: FMUMode end

"""Alias for `ModelExchange()`."""
const ME = ModelExchange()

"""Alias for `CoSimulation()`."""
const CS = CoSimulation()

# ---- Capabilities ----

"""
    FMUCapabilities

Capabilities extracted from the FMU's `modelDescription.xml` at construction time.
"""
struct FMUCapabilities
    "Whether the FMU supports checkpoint/restore via get/set FMU state"
    can_get_and_set_fmu_state::Bool
    "Whether the FMU has event mode (FMI 3.0 CS only)"
    has_event_mode::Bool
    "Number of event indicator (zero-crossing) functions"
    n_event_indicators::Int
    "FMI version: 2 or 3"
    fmi_version::Int
end

function FMUCapabilities(; can_get_and_set_fmu_state::Bool, has_event_mode::Bool = false,
        n_event_indicators::Int, fmi_version::Int)
    fmi_version in (2, 3) || throw(ArgumentError("fmi_version must be 2 or 3, got $fmi_version"))
    n_event_indicators >= 0 || throw(ArgumentError("n_event_indicators must be >= 0"))
    return FMUCapabilities(can_get_and_set_fmu_state, has_event_mode, n_event_indicators, fmi_version)
end

# ---- Abstract FMU system type ----

"""
    AbstractFMUSystem <: AbstractSystem

Abstract supertype for all FMU system types. Provides a dispatch target for
"any FMU system" regardless of mode.
"""
abstract type AbstractFMUSystem <: AbstractSystem end

# ---- FMU Callback types ----
# These are stored on FMUSystem and lowered to SciML callbacks during mtkcompile.

"""
    FMUContinuousCallback{W}

Represents FMU event indicators (zero-crossings). Lowered to a
`VectorContinuousCallback` during compilation.

The condition calls `fmi_get_event_indicators!`, and the affect enters event mode
and iterates `UpdateDiscreteStates`.
"""
struct FMUContinuousCallback{W}
    wrapper::W
    n_event_indicators::Int
end

"""
    FMUTimeCallback{W}

Represents FMU time events (`nextEventTime`). Lowered to a `DiscreteCallback`
that manages `add_tstop!` calls.
"""
struct FMUTimeCallback{W}
    wrapper::W
end

"""
    FMUStepCallback{W}

Represents CS FMU periodic communication stepping. Lowered to a
`PeriodicCallback` that calls `fmi_do_step!`.
"""
struct FMUStepCallback{W}
    wrapper::W
    communication_step_size::Float64
end

"""
    FMUStepEventCallback{W}

Represents ME FMU step events (`CompletedIntegratorStep`). Lowered to a
`DiscreteCallback` that checks whether the FMU wants to enter event mode
after each accepted step.
"""
struct FMUStepEventCallback{W}
    wrapper::W
end
```

- [ ] **Step 2: Write tests for the foundation types**

```julia
# test/fmi/fmusystem_unit.jl
using Test
using ModelingToolkit
import ModelingToolkitBase as MTKBase

@testset "FMU Foundation Types" begin
    @testset "Mode tags" begin
        @test MTKBase.ModelExchange <: MTKBase.FMUMode
        @test MTKBase.CoSimulation <: MTKBase.FMUMode
        @test MTKBase.ME isa MTKBase.ModelExchange
        @test MTKBase.CS isa MTKBase.CoSimulation
    end

    @testset "FMUCapabilities" begin
        caps = MTKBase.FMUCapabilities(
            can_get_and_set_fmu_state = true,
            has_event_mode = false,
            n_event_indicators = 2,
            fmi_version = 2
        )
        @test caps.can_get_and_set_fmu_state == true
        @test caps.has_event_mode == false
        @test caps.n_event_indicators == 2
        @test caps.fmi_version == 2

        @test_throws ArgumentError MTKBase.FMUCapabilities(
            can_get_and_set_fmu_state = false,
            n_event_indicators = 0,
            fmi_version = 4
        )
        @test_throws ArgumentError MTKBase.FMUCapabilities(
            can_get_and_set_fmu_state = false,
            n_event_indicators = -1,
            fmi_version = 2
        )
    end

    @testset "AbstractFMUSystem hierarchy" begin
        @test MTKBase.AbstractFMUSystem <: MTKBase.AbstractSystem
    end

    @testset "FMU callback types" begin
        # Use a dummy wrapper type for testing
        wrapper = :dummy_wrapper
        cc = MTKBase.FMUContinuousCallback(wrapper, 3)
        @test cc.wrapper === :dummy_wrapper
        @test cc.n_event_indicators == 3

        tc = MTKBase.FMUTimeCallback(wrapper)
        @test tc.wrapper === :dummy_wrapper

        sc = MTKBase.FMUStepCallback(wrapper, 0.001)
        @test sc.wrapper === :dummy_wrapper
        @test sc.communication_step_size == 0.001

        sec = MTKBase.FMUStepEventCallback(wrapper)
        @test sec.wrapper === :dummy_wrapper
    end
end
```

- [ ] **Step 3: Include the types file in ModelingToolkitBase**

In `lib/ModelingToolkitBase/src/ModelingToolkitBase.jl`, add the include **after** the `abstractsystem.jl` include (line 186) but **before** `connectors.jl` (line 188):

```julia
# After line 186: include("systems/abstractsystem.jl")
include("systems/fmu/types.jl")
# Line 187 continues: include("systems/connectiongraph.jl")
```

Also add exports near the existing exports (around line 91):

```julia
export FMUMode, ModelExchange, CoSimulation, ME, CS
export FMUCapabilities, AbstractFMUSystem
export FMUContinuousCallback, FMUTimeCallback, FMUStepCallback, FMUStepEventCallback
```

- [ ] **Step 4: Run the tests**

Run: `julia --project -e 'using Test; include("test/fmi/fmusystem_unit.jl")'`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add lib/ModelingToolkitBase/src/systems/fmu/types.jl \
        lib/ModelingToolkitBase/src/ModelingToolkitBase.jl \
        test/fmi/fmusystem_unit.jl
git commit -m "feat: add FMU foundation types (FMUMode, FMUCapabilities, callback types)"
```

---

### Task 2: FMUSystem Struct and AbstractSystem Interface

**Files:**
- Create: `lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl`
- Modify: `lib/ModelingToolkitBase/src/systems/fmu/types.jl` (if needed)
- Modify: `lib/ModelingToolkitBase/src/ModelingToolkitBase.jl`
- Modify: `lib/ModelingToolkitBase/src/systems/abstractsystem.jl`
- Modify: `lib/ModelingToolkitBase/src/systems/system.jl` (for `is_time_dependent`)
- Test: `test/fmi/fmusystem_unit.jl`

- [ ] **Step 1: Write failing tests for FMUSystem construction and interface**

Append to `test/fmi/fmusystem_unit.jl`:

```julia
using ModelingToolkit: t_nounits as t
using Symbolics: unwrap

@testset "FMUSystem struct" begin
    @testset "ME construction" begin
        iv = unwrap(t)
        s1 = unwrap(only(@variables mass__s(t)))
        s2 = unwrap(only(@variables mass__v(t)))
        ds1 = unwrap(only(@variables mass__der_s(t)))
        ds2 = unwrap(only(@variables mass__der_v(t)))
        p1 = unwrap(only(@parameters spring__c))

        caps = MTKBase.FMUCapabilities(
            can_get_and_set_fmu_state = true,
            has_event_mode = false,
            n_event_indicators = 2,
            fmi_version = 2
        )

        wrapper = :mock_wrapper
        valrefs = Dict(s1 => UInt32(0), s2 => UInt32(1),
                       ds1 => UInt32(2), ds2 => UInt32(3), p1 => UInt32(4))

        sys = MTKBase.FMUSystem{MTKBase.ModelExchange}(;
            name = :fmu,
            iv = iv,
            states = [s1, s2],
            derivatives = [ds1, ds2],
            inputs = Symbolics.SymbolicT[],
            outputs = Symbolics.SymbolicT[],
            parameters = [p1],
            observed = Equation[],
            wrapper = wrapper,
            capabilities = caps,
            value_references = valrefs,
            default_values = Dict{Symbolics.SymbolicT, Any}(s1 => 0.5, s2 => 0.0),
            communication_step_size = nothing
        )

        @test nameof(sys) === :fmu
        @test MTKBase.get_iv(sys) === iv
        @test MTKBase.get_unknowns(sys) == [s1, s2]
        @test MTKBase.get_ps(sys) == [p1]
        @test MTKBase.get_observed(sys) == Equation[]
        @test MTKBase.get_eqs(sys) == Equation[]
        @test MTKBase.get_systems(sys) == MTKBase.AbstractSystem[]
        @test MTKBase.is_time_dependent(sys) == true
        @test MTKBase.does_namespacing(sys) == true
        @test MTKBase.iscomplete(sys) == false
        @test MTKBase.isscheduled(sys) == false
        @test MTKBase.get_continuous_events(sys) isa Vector
        @test MTKBase.get_discrete_events(sys) isa Vector
        @test MTKBase.get_initialization_eqs(sys) == Equation[]
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `julia --project -e 'using Test; include("test/fmi/fmusystem_unit.jl")'`
Expected: FAIL — `FMUSystem` not defined yet.

- [ ] **Step 3: Write the FMUSystem struct and constructor**

```julia
# lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl

using Symbolics: SymbolicT, Equation

"""
    FMUSystem{Mode, WrapperType, ValRefType} <: AbstractFMUSystem

A system type purpose-built for FMUs. Stores FMU-native data alongside symbolic
variables for composition. Does not carry equations — the FMU *is* the dynamics.

# Type Parameters
- `Mode <: FMUMode`: `ModelExchange` or `CoSimulation`
- `WrapperType`: The FMI wrapper struct type (inferred at construction)
- `ValRefType`: The FMI value reference type (inferred at construction)

# Construction
```julia
FMUSystem{ModelExchange}(; name, iv, states, derivatives, inputs, outputs,
    parameters, observed, wrapper, capabilities, value_references,
    default_values, communication_step_size = nothing)
```
"""
struct FMUSystem{Mode <: FMUMode, WrapperType, ValRefType} <: AbstractFMUSystem
    # Identity
    name::Symbol
    iv::SymbolicT

    # Symbolic interface
    states::Vector{SymbolicT}
    derivatives::Vector{SymbolicT}
    inputs::Vector{SymbolicT}
    outputs::Vector{SymbolicT}
    parameters::Vector{SymbolicT}
    observed::Vector{Equation}

    # FMU-native data
    wrapper::WrapperType
    capabilities::FMUCapabilities
    value_references::Dict{SymbolicT, ValRefType}
    default_values::Dict{SymbolicT, Any}

    # Events (populated at construction based on capabilities)
    continuous_events::Vector{FMUContinuousCallback{WrapperType}}
    discrete_events::Vector{Union{FMUTimeCallback{WrapperType},
                                  FMUStepCallback{WrapperType},
                                  FMUStepEventCallback{WrapperType}}}

    # CS-specific
    communication_step_size::Union{Float64, Nothing}

    # Composition
    systems::Vector{AbstractSystem}
    parent::Union{Nothing, AbstractSystem}
    namespacing::Bool
    complete::Bool
end

"""
    FMUSystem{Mode}(; name, iv, states, derivatives, inputs, outputs,
        parameters, observed, wrapper, capabilities, value_references,
        default_values, communication_step_size = nothing)

Construct an `FMUSystem`. Validation is performed at construction time:
- ME FMUs must have derivatives for all states
- CS FMUs must provide `communication_step_size`
- Event callbacks are auto-populated from capabilities
"""
function FMUSystem{Mode}(;
        name::Symbol,
        iv,
        states::Vector{SymbolicT},
        derivatives::Vector{SymbolicT},
        inputs::Vector{SymbolicT},
        outputs::Vector{SymbolicT},
        parameters::Vector{SymbolicT},
        observed::Vector{Equation} = Equation[],
        wrapper::W,
        capabilities::FMUCapabilities,
        value_references::Dict{SymbolicT, VR},
        default_values::Dict{SymbolicT, Any} = Dict{SymbolicT, Any}(),
        communication_step_size::Union{Float64, Nothing} = nothing
    ) where {Mode <: FMUMode, W, VR}
    iv_unwrapped = Symbolics.unwrap(iv)

    # --- Validation ---
    if Mode === ModelExchange
        if length(derivatives) != length(states)
            throw(ArgumentError(
                "ME FMUSystem requires one derivative per state. " *
                "Got $(length(states)) states and $(length(derivatives)) derivatives."
            ))
        end
        if communication_step_size !== nothing
            throw(ArgumentError("communication_step_size must be nothing for ME FMUs."))
        end
    elseif Mode === CoSimulation
        if communication_step_size === nothing
            throw(ArgumentError("communication_step_size is required for CS FMUs."))
        end
        if !isempty(derivatives)
            throw(ArgumentError("CS FMUs should not have derivatives (states are discrete)."))
        end
    end

    # --- Build event callbacks from capabilities ---
    FMUDiscreteEvent = Union{FMUTimeCallback{W}, FMUStepCallback{W}, FMUStepEventCallback{W}}
    continuous_events = FMUContinuousCallback{W}[]
    discrete_events = FMUDiscreteEvent[]

    if capabilities.n_event_indicators > 0
        push!(continuous_events,
            FMUContinuousCallback(wrapper, capabilities.n_event_indicators))
    end

    if Mode === ModelExchange
        # ME always gets a step event callback (CompletedIntegratorStep)
        push!(discrete_events, FMUStepEventCallback(wrapper))
        # ME always gets a time callback (nextEventTime discovered at runtime)
        push!(discrete_events, FMUTimeCallback(wrapper))
    elseif Mode === CoSimulation
        # CS gets a periodic step callback
        push!(discrete_events, FMUStepCallback(wrapper, communication_step_size))
        # FMI 3.0 CS with event mode may also have time callbacks
        if capabilities.has_event_mode
            push!(discrete_events, FMUTimeCallback(wrapper))
        end
    end

    return FMUSystem{Mode, W, VR}(
        name, iv_unwrapped,
        states, derivatives, inputs, outputs, parameters, observed,
        wrapper, capabilities, value_references, default_values,
        continuous_events, discrete_events,
        communication_step_size,
        AbstractSystem[],  # no subsystems
        nothing,           # parent = nothing
        true,              # namespacing = true
        false              # complete = false
    )
end

# ---- AbstractSystem interface implementation ----

# Direct field access
get_iv(sys::AbstractFMUSystem) = getfield(sys, :iv)
has_iv(::AbstractFMUSystem) = true
get_unknowns(sys::AbstractFMUSystem) = getfield(sys, :states)
has_unknowns(::AbstractFMUSystem) = true
get_ps(sys::AbstractFMUSystem) = getfield(sys, :parameters)
has_ps(::AbstractFMUSystem) = true
get_observed(sys::AbstractFMUSystem) = getfield(sys, :observed)
has_observed(::AbstractFMUSystem) = true
get_systems(sys::AbstractFMUSystem) = getfield(sys, :systems)
has_systems(::AbstractFMUSystem) = true
get_inputs(sys::AbstractFMUSystem) = getfield(sys, :inputs)
has_inputs(::AbstractFMUSystem) = true
get_outputs(sys::AbstractFMUSystem) = getfield(sys, :outputs)
has_outputs(::AbstractFMUSystem) = true
get_continuous_events(sys::AbstractFMUSystem) = getfield(sys, :continuous_events)
has_continuous_events(::AbstractFMUSystem) = true
get_discrete_events(sys::AbstractFMUSystem) = getfield(sys, :discrete_events)
has_discrete_events(::AbstractFMUSystem) = true
get_name(sys::AbstractFMUSystem) = getfield(sys, :name)
has_name(::AbstractFMUSystem) = true
Base.nameof(sys::AbstractFMUSystem) = getfield(sys, :name)
iscomplete(sys::AbstractFMUSystem) = getfield(sys, :complete)

function does_namespacing(sys::AbstractFMUSystem)
    return getfield(sys, :namespacing)
end

# Constants / not-applicable
SymbolicIndexingInterface.is_time_dependent(::AbstractFMUSystem) = true
isscheduled(::AbstractFMUSystem) = false
get_eqs(::AbstractFMUSystem) = Equation[]
has_eqs(::AbstractFMUSystem) = true
get_initialization_eqs(::AbstractFMUSystem) = Equation[]
has_initialization_eqs(::AbstractFMUSystem) = true
get_noise_eqs(::AbstractFMUSystem) = nothing
has_noise_eqs(::AbstractFMUSystem) = false
get_jumps(::AbstractFMUSystem) = []
has_jumps(::AbstractFMUSystem) = false
get_brownians(::AbstractFMUSystem) = SymbolicT[]
has_brownians(::AbstractFMUSystem) = false
get_poissonians(::AbstractFMUSystem) = SymbolicT[]
has_poissonians(::AbstractFMUSystem) = false
get_costs(::AbstractFMUSystem) = SymbolicT[]
has_costs(::AbstractFMUSystem) = false
get_constraints(::AbstractFMUSystem) = Union{Equation, Symbolics.Inequality}[]
has_constraints(::AbstractFMUSystem) = false
get_connector_type(::AbstractFMUSystem) = nothing
has_connector_type(::AbstractFMUSystem) = false
get_var_to_name(::AbstractFMUSystem) = Dict{Symbol, SymbolicT}()
has_var_to_name(::AbstractFMUSystem) = false
get_guesses(::AbstractFMUSystem) = Dict{SymbolicT, Any}()
has_guesses(::AbstractFMUSystem) = false
get_bindings(::AbstractFMUSystem) = Dict{SymbolicT, Any}()
has_bindings(::AbstractFMUSystem) = false
get_initial_conditions(::AbstractFMUSystem) = Dict{SymbolicT, Any}()
has_initial_conditions(::AbstractFMUSystem) = false
get_tag(::AbstractFMUSystem) = UInt(0)
has_tag(::AbstractFMUSystem) = false
get_description(::AbstractFMUSystem) = ""
has_description(::AbstractFMUSystem) = true
get_metadata(::AbstractFMUSystem) = Base.ImmutableDict{DataType, Any}()
has_metadata(::AbstractFMUSystem) = false
get_parent(sys::AbstractFMUSystem) = getfield(sys, :parent)
has_parent(::AbstractFMUSystem) = true
get_is_dde(::AbstractFMUSystem) = false
has_is_dde(::AbstractFMUSystem) = true
get_tstops(::AbstractFMUSystem) = []
has_tstops(::AbstractFMUSystem) = true
get_index_cache(::AbstractFMUSystem) = nothing
has_index_cache(::AbstractFMUSystem) = false
get_isscheduled(::AbstractFMUSystem) = false
has_isscheduled(::AbstractFMUSystem) = true

# FMU-specific accessors
"""Get the FMU wrapper struct."""
get_fmu_wrapper(sys::AbstractFMUSystem) = getfield(sys, :wrapper)
"""Get FMU capabilities."""
get_fmu_capabilities(sys::AbstractFMUSystem) = getfield(sys, :capabilities)
"""Get the mapping from symbolic variables to FMI value references."""
get_value_references(sys::AbstractFMUSystem) = getfield(sys, :value_references)
"""Get FMU default values."""
get_default_values(sys::AbstractFMUSystem) = getfield(sys, :default_values)
"""Get FMU derivative variables (ME only)."""
get_derivatives(sys::AbstractFMUSystem) = getfield(sys, :derivatives)
"""Get FMU communication step size (CS only, nothing for ME)."""
get_communication_step_size(sys::AbstractFMUSystem) = getfield(sys, :communication_step_size)
```

- [ ] **Step 4: Include the fmusystem file in ModelingToolkitBase**

In `lib/ModelingToolkitBase/src/ModelingToolkitBase.jl`, add after the `types.jl` include:

```julia
include("systems/fmu/fmusystem.jl")
```

Add exports:

```julia
export FMUSystem
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `julia --project -e 'using Test; include("test/fmi/fmusystem_unit.jl")'`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add lib/ModelingToolkitBase/src/systems/fmu/fmusystem.jl \
        lib/ModelingToolkitBase/src/ModelingToolkitBase.jl \
        test/fmi/fmusystem_unit.jl
git commit -m "feat: add FMUSystem struct with AbstractSystem interface"
```

---

### Task 3: System Type Parameter for Subsystems

**Files:**
- Modify: `lib/ModelingToolkitBase/src/systems/system.jl:38` (struct definition)
- Modify: `lib/ModelingToolkitBase/src/systems/system.jl:300-378` (inner constructor)
- Modify: `lib/ModelingToolkitBase/src/systems/system.jl:449-643` (outer constructor)
- Test: `test/fmi/fmusystem_unit.jl`

This is the most delicate refactor. The goal is to make `System` accept heterogeneous subsystems
while keeping existing code working without changes.

**Key insight:** In Julia, `System` without parameters refers to `System` where `SubSys` can be
anything — so existing dispatch `f(::System)` continues to work for all `System{T}`.

- [ ] **Step 1: Write a failing test for heterogeneous subsystems**

Append to `test/fmi/fmusystem_unit.jl`:

```julia
@testset "System with FMUSystem subsystem" begin
    using ModelingToolkit: t_nounits as t, D_nounits as D

    iv = unwrap(t)
    s1 = unwrap(only(@variables fmu_state(t)))
    ds1 = unwrap(only(@variables fmu_der_state(t)))
    p1 = unwrap(only(@parameters fmu_param))

    caps = MTKBase.FMUCapabilities(
        can_get_and_set_fmu_state = false,
        n_event_indicators = 0,
        fmi_version = 2
    )

    fmu_sys = MTKBase.FMUSystem{MTKBase.ModelExchange}(;
        name = :fmu,
        iv = iv,
        states = [s1],
        derivatives = [ds1],
        inputs = Symbolics.SymbolicT[],
        outputs = Symbolics.SymbolicT[],
        parameters = [p1],
        wrapper = :mock,
        capabilities = caps,
        value_references = Dict{Symbolics.SymbolicT, UInt32}(
            s1 => UInt32(0), ds1 => UInt32(1), p1 => UInt32(2)),
        default_values = Dict{Symbolics.SymbolicT, Any}()
    )

    @variables x(t) = 1.0
    parent = System([D(x) ~ x], t; systems = [fmu_sys], name = :parent)
    @test length(MTKBase.get_systems(parent)) == 1
    @test MTKBase.get_systems(parent)[1] === fmu_sys
end
```

- [ ] **Step 2: Run tests to verify it fails**

Run: `julia --project -e 'using Test; include("test/fmi/fmusystem_unit.jl")'`
Expected: FAIL — `System` constructor rejects `FMUSystem` in the `systems` vector because of `Vector{System}(systems)` coercion at line 481-482.

- [ ] **Step 3: Add type parameter to System struct**

In `lib/ModelingToolkitBase/src/systems/system.jl`, change line 38:

Old:
```julia
struct System <: IntermediateDeprecationSystem
```

New:
```julia
struct System{SubSys <: AbstractSystem} <: IntermediateDeprecationSystem
```

Change line 162:

Old:
```julia
    systems::Vector{System}
```

New:
```julia
    systems::Vector{SubSys}
```

Change line 266:

Old:
```julia
    parent::Union{Nothing, System}
```

New:
```julia
    parent::Union{Nothing, AbstractSystem}
```

Change line 271:

Old:
```julia
    initializesystem::Union{Nothing, System}
```

New:
```julia
    initializesystem::Union{Nothing, AbstractSystem}
```

- [ ] **Step 4: Update the inner constructor**

In `lib/ModelingToolkitBase/src/systems/system.jl`, the inner constructor at line 300 needs to
accept the parameterized systems vector. Change:

Old (line 300):
```julia
    function System(
            tag, eqs, noise_eqs, jumps, constraints, costs, consolidate, unknowns, ps,
            brownians, poissonians, iv, observed, var_to_name, name, description, bindings,
            initial_conditions, guesses, systems, initialization_eqs, continuous_events,
            discrete_events, connector_type, assertions = Dict{SymbolicT, String}(),
```

New:
```julia
    function System(
            tag, eqs, noise_eqs, jumps, constraints, costs, consolidate, unknowns, ps,
            brownians, poissonians, iv, observed, var_to_name, name, description, bindings,
            initial_conditions, guesses, systems::Vector{S}, initialization_eqs, continuous_events,
            discrete_events, connector_type, assertions = Dict{SymbolicT, String}(),
```
(add `::Vector{S}` annotation)

And change the `return new(` at line 366 to:

Old:
```julia
        return new(
```

New:
```julia
        return new{S}(
```

- [ ] **Step 5: Update the outer constructor to accept heterogeneous subsystems**

In `lib/ModelingToolkitBase/src/systems/system.jl`, change lines 481-483:

Old:
```julia
    if !(systems isa Vector{System})
        systems = Vector{System}(systems)
    end
```

New:
```julia
    if !(systems isa Vector{<:AbstractSystem})
        systems = collect(AbstractSystem, systems)
    end
    if all(s -> s isa System, systems) && !(systems isa Vector{System})
        systems = Vector{System}(systems)
    end
```

This logic: if all subsystems are `System`, keep `Vector{System}` for backward compatibility.
Otherwise, widen to `Vector{AbstractSystem}`.

Also update the default in the outer constructor signature at line 457. Keep the default
as `System[]` for backward compatibility — the coercion logic below handles the widening:

Old:
```julia
        guesses = SymmapT(), systems = System[], initialization_eqs = Equation[],
```

No change needed here. The default stays `System[]`. When a user passes a vector containing
`FMUSystem`, the coercion logic in the next step widens it to `Vector{AbstractSystem}`.

- [ ] **Step 6: Run tests**

Run: `julia --project -e 'using Test; include("test/fmi/fmusystem_unit.jl")'`
Expected: All tests pass, including the new "System with FMUSystem subsystem" test.

Then run the existing test suite to check nothing breaks:

Run: `julia --project test/runtests.jl` (or the relevant subset)
Expected: No regressions.

- [ ] **Step 7: Commit**

```bash
git add lib/ModelingToolkitBase/src/systems/system.jl \
        test/fmi/fmusystem_unit.jl
git commit -m "feat: add type parameter to System for heterogeneous subsystems"
```

---

### Task 4: Fix Hard-coded System[] Across Codebase

**Files:**
- Modify: `lib/ModelingToolkitBase/src/systems/connectors.jl:26,66,559,826,964,966,1097`
- Modify: `src/systems/systemstructure.jl:10,23,24,27,40`

Each change replaces `System[]` with `AbstractSystem[]` or relaxes `::System` type assertions.

- [ ] **Step 1: Fix connectors.jl — connect() and domain_connect()**

In `lib/ModelingToolkitBase/src/systems/connectors.jl`:

**Line 26:** Change `_syss = System[]` to `_syss = AbstractSystem[]`

**Line 66:** Change `_syss = System[]` to `_syss = AbstractSystem[]`

- [ ] **Step 2: Fix connectors.jl — _generate_connectionsets!()**

**Line 558:** Change `systems = systems::Vector{System}` to:
```julia
systems = systems::Vector{<:AbstractSystem}
```

**Line 559:** Change `regular_systems = System[]` to `regular_systems = AbstractSystem[]`

- [ ] **Step 3: Fix connectors.jl — generate_connection_set!()**

**Line 826:** Change `new_systems = System[]` to `new_systems = AbstractSystem[]`

- [ ] **Step 4: Fix connectors.jl — get_domain_bindings()**

**Line 964:** Change `systems = System[]` to `systems = AbstractSystem[]`

**Line 966:** Change `push!(systems, variable_from_vertex(sys, cvar)::System)` to:
```julia
push!(systems, variable_from_vertex(sys, cvar)::AbstractSystem)
```

- [ ] **Step 5: Fix connectors.jl — get_flowvar()**

**Line 1097:** Change `parent_sys = iterative_getproperty(sys, cvert.name)::System` to:
```julia
parent_sys = iterative_getproperty(sys, cvert.name)::AbstractSystem
```

- [ ] **Step 6: Fix systemstructure.jl — extract_top_level_statemachines()**

In `src/systems/systemstructure.jl`:

**Line 10:** Change function signature from `function extract_top_level_statemachines(sys::System)`
to `function extract_top_level_statemachines(sys::AbstractSystem)`.

**Line 23-24:** Change:
```julia
        newsubsystems = System[]
        statemachines = System[]
```
to:
```julia
        newsubsystems = AbstractSystem[]
        statemachines = AbstractSystem[]
```

**Line 40:** Change `function remove_child_equations(sys::System)` to
`function remove_child_equations(sys::AbstractSystem)`.

Note: `remove_child_equations` uses `@set! sys.eqs = Equation[]` which requires `Setfield`
support. For `FMUSystem`, this function should be a no-op (FMUSystem has no equations to remove).
Add a dispatch:

```julia
remove_child_equations(sys::AbstractFMUSystem) = sys
```

- [ ] **Step 7: Run the full test suite**

Run: `julia --project test/runtests.jl`
Expected: No regressions. All existing tests pass.

- [ ] **Step 8: Commit**

```bash
git add lib/ModelingToolkitBase/src/systems/connectors.jl \
        src/systems/systemstructure.jl
git commit -m "refactor: replace hard-coded System[] with AbstractSystem[] for FMU support"
```

---

### Task 5: mtkcompile FMU Pass

**Files:**
- Create: `src/systems/fmu_compilation.jl`
- Modify: `src/systems/systems.jl:23-30` (hook into `__mtkcompile`)
- Modify: `src/ModelingToolkit.jl` (include new file)
- Test: `test/fmi/fmu_compilation.jl`

The key idea: before the existing `expand_connections` + `TearingState` pipeline, extract
FMU subsystems from the hierarchy and collect their symbolic variables. FMU internals are
opaque — no tearing, no alias elimination. FMU states/parameters/observed are namespaced
and merged into the parent. FMU events are collected for later lowering.

- [ ] **Step 1: Write tests for FMU extraction from system hierarchy**

```julia
# test/fmi/fmu_compilation.jl
using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
import ModelingToolkitBase as MTKBase
using Symbolics: unwrap, SymbolicT

@testset "FMU Compilation Pass" begin
    @testset "extract_fmu_subsystems" begin
        iv = unwrap(t)
        s1 = unwrap(only(@variables fmu_state(t)))
        ds1 = unwrap(only(@variables fmu_der_state(t)))

        caps = MTKBase.FMUCapabilities(
            can_get_and_set_fmu_state = false,
            n_event_indicators = 0,
            fmi_version = 2
        )

        fmu_sys = MTKBase.FMUSystem{MTKBase.ModelExchange}(;
            name = :fmu,
            iv = iv,
            states = [s1],
            derivatives = [ds1],
            inputs = SymbolicT[],
            outputs = SymbolicT[],
            parameters = SymbolicT[],
            wrapper = :mock,
            capabilities = caps,
            value_references = Dict{SymbolicT, UInt32}(
                s1 => UInt32(0), ds1 => UInt32(1)),
            default_values = Dict{SymbolicT, Any}()
        )

        @variables x(t) = 1.0
        parent = System([D(x) ~ x], t; systems = [fmu_sys], name = :parent)

        # Test that extraction identifies FMU subsystems
        fmu_subsystems, regular_sys = ModelingToolkit.extract_fmu_subsystems(parent)
        @test length(fmu_subsystems) == 1
        @test fmu_subsystems[1] isa MTKBase.FMUSystem
        @test length(MTKBase.get_systems(regular_sys)) == 0
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `julia --project -e 'using Test; include("test/fmi/fmu_compilation.jl")'`
Expected: FAIL — `extract_fmu_subsystems` not defined.

- [ ] **Step 3: Implement extract_fmu_subsystems**

```julia
# src/systems/fmu_compilation.jl

"""
    extract_fmu_subsystems(sys::System)

Walk the system hierarchy and separate FMU subsystems from regular subsystems.
Returns `(fmu_subsystems, sys_without_fmus)` where:
- `fmu_subsystems` is a vector of `(namespaced_name, FMUSystem)` pairs
- `sys_without_fmus` is the system with FMU subsystems removed

FMU subsystems are extracted at the first level they are found. They do not
participate in symbolic manipulation (tearing, alias elimination).
"""
function extract_fmu_subsystems(sys)
    subsystems = get_systems(sys)
    fmu_subsystems = AbstractFMUSystem[]
    regular_subsystems = AbstractSystem[]

    for s in subsystems
        if s isa AbstractFMUSystem
            push!(fmu_subsystems, s)
        else
            push!(regular_subsystems, s)
        end
    end

    if !isempty(fmu_subsystems)
        sys = Setfield.@set sys.systems = regular_subsystems
    end

    return fmu_subsystems, sys
end

"""
    collect_fmu_variables(sys, fmu_subsystems)

Namespace and collect symbolic variables from FMU subsystems into vectors that
can be merged into the parent system's compilation.

Returns a named tuple of:
- `unknowns`: namespaced FMU state variables
- `parameters`: namespaced FMU parameters
- `observed`: namespaced FMU observed equations
- `continuous_events`: FMU continuous callbacks
- `discrete_events`: FMU discrete callbacks
"""
function collect_fmu_variables(sys, fmu_subsystems)
    fmu_unknowns = SymbolicT[]
    fmu_parameters = SymbolicT[]
    fmu_observed = Equation[]
    fmu_continuous_events = []
    fmu_discrete_events = []

    for fmu in fmu_subsystems
        # Namespace FMU variables into parent context
        for v in get_unknowns(fmu)
            push!(fmu_unknowns, renamespace(fmu, v))
        end
        for p in get_ps(fmu)
            push!(fmu_parameters, renamespace(fmu, p))
        end
        for eq in get_observed(fmu)
            push!(fmu_observed, namespace_equation(eq, fmu))
        end
        append!(fmu_continuous_events, get_continuous_events(fmu))
        append!(fmu_discrete_events, get_discrete_events(fmu))
    end

    return (;
        unknowns = fmu_unknowns,
        parameters = fmu_parameters,
        observed = fmu_observed,
        continuous_events = fmu_continuous_events,
        discrete_events = fmu_discrete_events
    )
end
```

- [ ] **Step 4: Include the file in ModelingToolkit.jl**

In `src/ModelingToolkit.jl`, add after line 147 (`include("systems/systems.jl")`):

```julia
include("systems/fmu_compilation.jl")
```

- [ ] **Step 5: Run tests**

Run: `julia --project -e 'using Test; include("test/fmi/fmu_compilation.jl")'`
Expected: All tests pass.

- [ ] **Step 6: Hook into __mtkcompile**

In `src/systems/systems.jl`, modify `__mtkcompile` to extract FMU subsystems before the existing
pipeline. After line 30 (`sort_eqs = true, kwargs...`) and before line 31 (`sys, statemachines = ...`):

```julia
    # Extract FMU subsystems — they bypass symbolic manipulation
    fmu_subsystems, sys = extract_fmu_subsystems(sys)
    fmu_data = isempty(fmu_subsystems) ? nothing : collect_fmu_variables(sys, fmu_subsystems)
```

Then after the `mtkcompile!` call returns the compiled system (line 49-51), merge FMU data.
The `mtkcompile!` function returns a `System`. We post-process it:

```julia
    # Existing code (line 49):
    compiled_sys = mtkcompile!(state; inputs, outputs, disturbance_inputs, kwargs...)

    # NEW: merge FMU data back into compiled system
    if fmu_data !== nothing
        compiled_sys = merge_fmu_data(compiled_sys, fmu_data, fmu_subsystems)
    end

    return compiled_sys
```

- [ ] **Step 7: Implement merge_fmu_data**

Add to `src/systems/fmu_compilation.jl`:

```julia
"""
    merge_fmu_data(compiled_sys, fmu_data, fmu_subsystems)

Merge FMU symbolic variables and events into the compiled system. This runs
after `mtkcompile!` has processed the regular (non-FMU) part of the system.

For ME FMUs: adds FMU states as unknowns with `fmu_eval`-based derivative equations.
For CS FMUs: adds FMU states as discrete parameters (not unknowns).

FMU events are appended to the system's continuous/discrete events for later
lowering during problem construction.
"""
function merge_fmu_data(compiled_sys, fmu_data, fmu_subsystems)
    # Append FMU parameters
    new_ps = vcat(get_ps(compiled_sys), fmu_data.parameters)

    # Append FMU observed equations
    new_observed = vcat(get_observed(compiled_sys), fmu_data.observed)

    # Handle unknowns and equations per FMU mode
    new_unknowns = copy(get_unknowns(compiled_sys))
    new_eqs = copy(get_eqs(compiled_sys))

    for fmu in fmu_subsystems
        if fmu isa FMUSystem{ModelExchange}
            # ME: FMU states become unknowns, derivatives become equations
            # The fmu_eval callable is stored as a parameter
            for (state, deriv) in zip(get_unknowns(fmu), get_derivatives(fmu))
                ns_state = renamespace(fmu, state)
                ns_deriv = renamespace(fmu, deriv)
                push!(new_unknowns, ns_state)
                # D(state) ~ fmu_eval(...)[i] — the actual equation references
                # the wrapper callable, which is registered during codegen
                push!(new_eqs, Differential(get_iv(compiled_sys))(ns_state) ~ ns_deriv)
            end
        elseif fmu isa FMUSystem{CoSimulation}
            # CS: states are discrete — they don't appear as unknowns.
            # They are updated by the periodic callback, not by the ODE solver.
            # Add them as discrete parameters instead.
            for state in get_unknowns(fmu)
                push!(new_ps, renamespace(fmu, state))
            end
        end
    end

    # Store FMU subsystem references in metadata for codegen access
    fmu_metadata = Base.ImmutableDict(get_metadata(compiled_sys),
        FMUSubsystemsKey => fmu_subsystems)

    # Rebuild the compiled system with merged data
    @set! compiled_sys.eqs = new_eqs
    @set! compiled_sys.unknowns = new_unknowns
    @set! compiled_sys.ps = new_ps
    @set! compiled_sys.observed = new_observed
    @set! compiled_sys.metadata = fmu_metadata

    return compiled_sys
end

"""Metadata key for storing FMU subsystem references on the compiled system."""
struct FMUSubsystemsKey end
```

- [ ] **Step 7: Run the full test suite**

Run: `julia --project test/runtests.jl`
Expected: No regressions. FMU compilation tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/systems/fmu_compilation.jl \
        src/systems/systems.jl \
        src/ModelingToolkit.jl \
        test/fmi/fmu_compilation.jl
git commit -m "feat: add mtkcompile pass for FMU subsystem extraction"
```

---

### Task 6: FMU Callback Lowering (Codegen)

**Files:**
- Create: `src/systems/fmu_codegen.jl`
- Modify: `src/ModelingToolkit.jl` (include new file)
- Test: `test/fmi/fmu_compilation.jl` (extend)

This task implements the lowering of FMU-specific callback types to standard SciML callbacks.
The actual FMI API calls (e.g., `fmi2GetEventIndicators`) are dispatched through the wrapper,
which is populated in the extension (Task 7). Here we define the lowering structure.

- [ ] **Step 1: Write tests for callback lowering**

Append to `test/fmi/fmu_compilation.jl`:

```julia
@testset "FMU Callback Lowering" begin
    @testset "FMUContinuousCallback lowering" begin
        wrapper = :mock
        fmu_cb = MTKBase.FMUContinuousCallback(wrapper, 3)
        # lower_fmu_callback should return a VectorContinuousCallback
        cb = ModelingToolkit.lower_fmu_continuous_callback(fmu_cb)
        @test cb isa SciMLBase.VectorContinuousCallback
        @test cb.len == 3
    end

    @testset "FMUStepCallback lowering" begin
        wrapper = :mock
        fmu_cb = MTKBase.FMUStepCallback(wrapper, 0.01)
        cb = ModelingToolkit.lower_fmu_step_callback(fmu_cb)
        @test cb isa DiffEqCallbacks.PeriodicCallback
    end

    @testset "FMUTimeCallback lowering" begin
        wrapper = :mock
        fmu_cb = MTKBase.FMUTimeCallback(wrapper)
        cb = ModelingToolkit.lower_fmu_time_callback(fmu_cb)
        @test cb isa SciMLBase.DiscreteCallback
    end

    @testset "FMUStepEventCallback lowering" begin
        wrapper = :mock
        fmu_cb = MTKBase.FMUStepEventCallback(wrapper)
        cb = ModelingToolkit.lower_fmu_step_event_callback(fmu_cb)
        @test cb isa SciMLBase.DiscreteCallback
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `julia --project -e 'using Test; include("test/fmi/fmu_compilation.jl")'`
Expected: FAIL — lowering functions not defined.

- [ ] **Step 3: Implement callback lowering functions**

```julia
# src/systems/fmu_codegen.jl

using SciMLBase: VectorContinuousCallback, DiscreteCallback
using DiffEqCallbacks: PeriodicCallback

"""
    lower_fmu_continuous_callback(cb::FMUContinuousCallback)

Lower an FMU continuous callback to a `VectorContinuousCallback`.

The condition function calls the wrapper's `get_event_indicators!` method.
The affect function enters event mode, iterates discrete state updates,
and handles state changes and time events.
"""
function lower_fmu_continuous_callback(cb::FMUContinuousCallback)
    wrapper = cb.wrapper
    n = cb.n_event_indicators

    condition = function (out, u, t, integrator)
        fmu_get_event_indicators!(wrapper, out, u, t)
    end

    affect = function (integrator, idx)
        fmu_enter_event_mode!(wrapper)
        while true
            info = fmu_update_discrete_states!(wrapper)
            info.newDiscreteStatesNeeded || break
            if info.terminateSimulation
                SciMLBase.terminate!(integrator)
                return
            end
        end
        if info.valuesOfContinuousStatesChanged
            fmu_get_continuous_states!(wrapper, integrator.u)
            SciMLBase.u_modified!(integrator, true)
        end
        if info.nextEventTimeDefined
            SciMLBase.add_tstop!(integrator, info.nextEventTime)
        end
        fmu_enter_continuous_time_mode!(wrapper)
    end

    return VectorContinuousCallback(condition, affect, n)
end

"""
    lower_fmu_time_callback(cb::FMUTimeCallback)

Lower an FMU time callback to a `DiscreteCallback` that fires at FMU-requested
time stops. The callback condition is always false (it only fires at tstops
added by other callbacks). The affect handles time event processing.
"""
function lower_fmu_time_callback(cb::FMUTimeCallback)
    wrapper = cb.wrapper

    # This callback fires at tstops that were added by the continuous callback
    # or initialization. The condition detects when we've hit an FMU time event.
    condition = function (u, t, integrator)
        fmu_has_pending_time_event(wrapper, t)
    end

    affect = function (integrator)
        fmu_enter_event_mode!(wrapper)
        while true
            info = fmu_update_discrete_states!(wrapper)
            info.newDiscreteStatesNeeded || break
            if info.terminateSimulation
                SciMLBase.terminate!(integrator)
                return
            end
        end
        if info.valuesOfContinuousStatesChanged
            fmu_get_continuous_states!(wrapper, integrator.u)
            SciMLBase.u_modified!(integrator, true)
        end
        if info.nextEventTimeDefined
            SciMLBase.add_tstop!(integrator, info.nextEventTime)
        end
        fmu_enter_continuous_time_mode!(wrapper)
    end

    return DiscreteCallback(condition, affect)
end

"""
    lower_fmu_step_callback(cb::FMUStepCallback)

Lower a CS FMU step callback to a `PeriodicCallback` that calls `fmi_do_step!`
at each communication point.
"""
function lower_fmu_step_callback(cb::FMUStepCallback)
    wrapper = cb.wrapper
    dt = cb.communication_step_size

    affect = function (integrator)
        fmu_do_step!(wrapper, integrator.t - dt, dt)
        fmu_read_outputs!(wrapper, integrator)
        SciMLBase.u_modified!(integrator, true)
    end

    return PeriodicCallback(affect, dt)
end

"""
    lower_fmu_step_event_callback(cb::FMUStepEventCallback)

Lower an ME FMU step event callback to a `DiscreteCallback` that checks
`CompletedIntegratorStep` at each accepted solver step.
"""
function lower_fmu_step_event_callback(cb::FMUStepEventCallback)
    wrapper = cb.wrapper

    # Fires at every accepted step
    condition = (u, t, integrator) -> true

    affect = function (integrator)
        enter_event_mode = fmu_completed_integrator_step!(wrapper)
        if enter_event_mode
            fmu_enter_event_mode!(wrapper)
            while true
                info = fmu_update_discrete_states!(wrapper)
                info.newDiscreteStatesNeeded || break
                if info.terminateSimulation
                    SciMLBase.terminate!(integrator)
                    return
                end
            end
            if info.valuesOfContinuousStatesChanged
                fmu_get_continuous_states!(wrapper, integrator.u)
                SciMLBase.u_modified!(integrator, true)
            end
            if info.nextEventTimeDefined
                SciMLBase.add_tstop!(integrator, info.nextEventTime)
            end
            fmu_enter_continuous_time_mode!(wrapper)
        end
    end

    return DiscreteCallback(condition, affect)
end

"""
    lower_fmu_lifecycle_callbacks(wrapper, capabilities::FMUCapabilities)

Create the initialization and finalization callbacks for an FMU instance.
Returns `(initialize_cb, finalize_cb, step_rejection_cb)`.
"""
function lower_fmu_lifecycle_callbacks(wrapper, capabilities::FMUCapabilities)
    initialize = function (integrator)
        fmu_instantiate!(wrapper)
        fmu_enter_initialization_mode!(wrapper, integrator.u, integrator.p, integrator.t)
        fmu_exit_initialization_mode!(wrapper)
        # Initial event iteration
        info = fmu_initial_event_iteration!(wrapper)
        if info.nextEventTimeDefined
            SciMLBase.add_tstop!(integrator, info.nextEventTime)
        end
        fmu_enter_continuous_time_mode!(wrapper)
    end

    finalize = function (integrator)
        fmu_terminate_instance!(wrapper)
        fmu_free_instance!(wrapper)
    end

    # Step rejection handling
    step_rejection = if capabilities.can_get_and_set_fmu_state
        # Checkpoint/restore approach
        DiscreteCallback(
            (u, t, integrator) -> true,
            function (integrator)
                fmu_save_state!(wrapper)
            end
        )
    else
        # Re-instantiate approach
        DiscreteCallback(
            (u, t, integrator) -> true,
            function (integrator)
                fmu_save_accepted_state!(wrapper, integrator.u, integrator.t)
            end
        )
    end

    return (; initialize, finalize, step_rejection)
end

# ---- Stub functions ----
# These are overridden by the FMI extension (MTKFMIExt) with actual FMI calls.
# They exist here so the codegen can reference them without the extension loaded.

function fmu_get_event_indicators! end
function fmu_enter_event_mode! end
function fmu_update_discrete_states! end
function fmu_get_continuous_states! end
function fmu_enter_continuous_time_mode! end
function fmu_has_pending_time_event end
function fmu_do_step! end
function fmu_read_outputs! end
function fmu_completed_integrator_step! end
function fmu_instantiate! end
function fmu_enter_initialization_mode! end
function fmu_exit_initialization_mode! end
function fmu_initial_event_iteration! end
function fmu_terminate_instance! end
function fmu_free_instance! end
function fmu_save_state! end
function fmu_save_accepted_state! end
```

- [ ] **Step 4: Include in ModelingToolkit.jl**

In `src/ModelingToolkit.jl`, add after `fmu_compilation.jl`:

```julia
include("systems/fmu_codegen.jl")
```

- [ ] **Step 5: Run tests**

Run: `julia --project -e 'using Test; include("test/fmi/fmu_compilation.jl")'`
Expected: All tests pass.

Note: The stub functions mean the callbacks will be created but can't actually be *called*
without the extension. The tests verify structural correctness (right callback types returned).

- [ ] **Step 6: Commit**

```bash
git add src/systems/fmu_codegen.jl \
        src/ModelingToolkit.jl \
        test/fmi/fmu_compilation.jl
git commit -m "feat: add FMU callback lowering to SciML callback types"
```

---

### Task 7: Rewrite MTKFMIExt Extension

**Files:**
- Modify: `ext/MTKFMIExt.jl` (major rewrite)
- Test: `test/fmi/fmi.jl` (update existing tests)

This task rewrites the extension to construct `FMUSystem` instead of faking equations in a
regular `System`. The wrapper structs (`FMI2InstanceWrapper`, `FMI3InstanceWrapper`) are
retained and enhanced. The `FMIComponent` function now returns an `FMUSystem`.

This is the largest task. The existing wrappers and variable parsing logic are preserved
where possible, but the output changes from `System` to `FMUSystem`.

- [ ] **Step 1: Update FMIComponent to return FMUSystem**

The existing `FMIComponent` function at `ext/MTKFMIExt.jl:100` returns a `System`. Rewrite it
to return an `FMUSystem{Mode}` instead. The key changes:

1. **Keep** the variable parsing logic (`fmi_variables_to_mtk_variables!`, lines 128-207)
2. **Keep** the wrapper construction logic
3. **Remove** the equation construction (fake `D(state) ~ wrapper(...)` equations)
4. **Remove** the `@register_array_symbolic` registration (no longer needed for the system itself)
5. **Replace** the `System(eqs, ...)` return with `FMUSystem{Mode}(; ...)` return

The core structure becomes:

```julia
function MTK.FMIComponent(
        ::Val{Ver}; fmu = nothing, tolerance = 1.0e-6,
        communication_step_size = nothing, type, name
    ) where {Ver}
    # ... existing validation ...
    # ... existing variable parsing (states, derivatives, inputs, outputs, params) ...
    # ... existing wrapper construction ...

    # Extract capabilities from modelDescription
    caps = _extract_capabilities(fmu, Ver)

    # Build value_references dict
    # (already exists as `value_references` in current code)

    Mode = type == :ME ? MTKBase.ModelExchange : MTKBase.CoSimulation

    return MTKBase.FMUSystem{Mode}(;
        name = name,
        iv = t,
        states = SymbolicT.(diffvars),
        derivatives = type == :ME ? SymbolicT.(dervars) : SymbolicT[],
        inputs = SymbolicT.(inputs),
        outputs = SymbolicT.(alloutputs),
        parameters = SymbolicT.(params),
        observed = observed,
        wrapper = wrapper,
        capabilities = caps,
        value_references = Dict{SymbolicT, valref_type}(
            unwrap(k) => v for (k, v) in value_references),
        default_values = Dict{SymbolicT, Any}(
            unwrap(k) => v for (k, v) in defs),
        communication_step_size = communication_step_size
    )
end
```

- [ ] **Step 2: Add _extract_capabilities helper**

```julia
function _extract_capabilities(fmu, ver::Int)
    if ver == 2
        md = fmu.modelDescription
        MTKBase.FMUCapabilities(
            can_get_and_set_fmu_state = md.coSimulation !== nothing ?
                md.coSimulation.canGetAndSetFMUstate :
                (md.modelExchange !== nothing && hasfield(typeof(md.modelExchange), :canGetAndSetFMUstate) ?
                    md.modelExchange.canGetAndSetFMUstate : false),
            has_event_mode = false,  # FMI 2.0 doesn't have explicit event mode for CS
            n_event_indicators = md.numberOfEventIndicators,
            fmi_version = 2
        )
    else  # ver == 3
        md = fmu.modelDescription
        MTKBase.FMUCapabilities(
            can_get_and_set_fmu_state = md.coSimulation !== nothing ?
                md.coSimulation.canGetAndSetFMUstate :
                (md.modelExchange !== nothing ? md.modelExchange.canGetAndSetFMUstate : false),
            has_event_mode = md.coSimulation !== nothing ?
                md.coSimulation.hasEventMode : false,
            n_event_indicators = md.numberOfEventIndicators,
            fmi_version = 3
        )
    end
end
```

- [ ] **Step 3: Implement the FMI API stub functions for FMI2**

Override the stub functions from `fmu_codegen.jl` for `FMI2InstanceWrapper`:

```julia
function MTK.fmu_get_event_indicators!(wrapper::FMI2InstanceWrapper, out, u, t)
    inst = wrapper.instance
    inst === nothing && error("FMU instance not initialized")
    FMI.fmi2SetTime(inst, t)
    FMI.fmi2SetContinuousStates(inst, u[wrapper.state_indices])
    FMI.fmi2GetEventIndicators!(inst, out)
end

function MTK.fmu_enter_event_mode!(wrapper::FMI2InstanceWrapper)
    FMI.fmi2EnterEventMode(wrapper.instance)
end

function MTK.fmu_update_discrete_states!(wrapper::FMI2InstanceWrapper)
    FMI.fmi2NewDiscreteStates(wrapper.instance)
end

function MTK.fmu_get_continuous_states!(wrapper::FMI2InstanceWrapper, u)
    states = similar(u, length(wrapper.state_value_references))
    FMI.fmi2GetContinuousStates!(wrapper.instance, states)
    u[wrapper.state_indices] .= states
end

function MTK.fmu_enter_continuous_time_mode!(wrapper::FMI2InstanceWrapper)
    FMI.fmi2EnterContinuousTimeMode(wrapper.instance)
end

function MTK.fmu_completed_integrator_step!(wrapper::FMI2InstanceWrapper)
    (enterEventMode, terminateSimulation) = FMI.fmi2CompletedIntegratorStep(
        wrapper.instance, FMI.fmi2True)
    return enterEventMode == FMI.fmi2True
end

function MTK.fmu_instantiate!(wrapper::FMI2InstanceWrapper)
    wrapper.instance = FMI.fmi2Instantiate!(wrapper.fmu)
end

function MTK.fmu_enter_initialization_mode!(wrapper::FMI2InstanceWrapper, u, p, t)
    FMI.fmi2SetupExperiment(wrapper.instance, wrapper.tolerance; startTime = t)
    # Set initial parameter values
    if !isempty(wrapper.param_value_references)
        FMI.fmi2SetReal(wrapper.instance, wrapper.param_value_references,
            p[wrapper.param_indices])
    end
    # Set initial state values
    if !isempty(wrapper.state_value_references)
        FMI.fmi2SetContinuousStates(wrapper.instance, u[wrapper.state_indices])
    end
    FMI.fmi2EnterInitializationMode(wrapper.instance)
end

function MTK.fmu_exit_initialization_mode!(wrapper::FMI2InstanceWrapper)
    FMI.fmi2ExitInitializationMode(wrapper.instance)
end

function MTK.fmu_terminate_instance!(wrapper::FMI2InstanceWrapper)
    wrapper.instance === nothing && return
    FMI.fmi2Terminate(wrapper.instance)
end

function MTK.fmu_free_instance!(wrapper::FMI2InstanceWrapper)
    wrapper.instance === nothing && return
    FMI.fmi2FreeInstance!(wrapper.instance)
    wrapper.instance = nothing
end

function MTK.fmu_save_state!(wrapper::FMI2InstanceWrapper)
    wrapper.saved_state = FMI.fmi2GetFMUstate(wrapper.instance)
end

function MTK.fmu_save_accepted_state!(wrapper::FMI2InstanceWrapper, u, t)
    wrapper.last_accepted_u = copy(u)
    wrapper.last_accepted_t = t
end
```

- [ ] **Step 4: Add state_indices and param_indices fields to wrappers**

The wrappers need to know which indices in the global `u` and `p` vectors correspond to
their variables. These are set during the codegen/problem construction phase.

Add to `FMI2InstanceWrapper`:

```julia
mutable struct FMI2InstanceWrapper
    # ... existing fields ...

    # Index mapping (set during problem construction)
    state_indices::Vector{Int}
    param_indices::Vector{Int}

    # Step rejection state
    saved_state::Any  # fmi2FMUstate for checkpoint/restore
    last_accepted_u::Union{Nothing, Vector{Float64}}
    last_accepted_t::Float64
end
```

Similarly for `FMI3InstanceWrapper`.

- [ ] **Step 5: Implement FMI3 stub functions**

Follow the same pattern as Step 3 but using `fmi3*` API calls. The structure is identical;
only the FMI API function names change (`fmi3SetTime`, `fmi3GetEventIndicators`, etc.).

- [ ] **Step 6: Update existing tests**

Update `test/fmi/fmi.jl` to use the new `FMUSystem` API. The test structure is similar,
but now `FMIComponent` returns an `FMUSystem` rather than a `System`:

```julia
@testset "v2, ME" begin
    fmu = loadFMU("SpringPendulum1D", "Dymola", "2022x"; type = :ME)
    fmu_sys = MTK.FMIComponent(Val(2); fmu, type = :ME, name = :pendulum)

    # FMIComponent now returns an FMUSystem
    @test fmu_sys isa MTKBase.FMUSystem{MTKBase.ModelExchange}

    # Compose with a parent system for standalone usage
    @mtkcompile sys = System(Equation[], t; systems = [fmu_sys], name = :sys)

    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        sys, [sys.pendulum.mass__s => 0.5, sys.pendulum.mass__v => 0.0], (0.0, 8.0)
    )
    sol = solve(prob, Tsit5(); reltol = 1.0e-8, abstol = 1.0e-8)
    @test SciMLBase.successful_retcode(sol)
end
```

Note: The exact test changes depend on how the mtkcompile pass (Task 5) integrates the FMU
into the compiled system. The key behavioral difference: the user no longer wraps FMU in
`@mtkcompile sys = MTK.FMIComponent(...)`. Instead they create the FMUSystem and compose it.

- [ ] **Step 7: Run the full FMI test suite**

Run: `julia --project -e 'using Test; include("test/fmi/fmi.jl")'`
Expected: All FMI tests pass with the new FMUSystem-based API.

- [ ] **Step 8: Commit**

```bash
git add ext/MTKFMIExt.jl test/fmi/fmi.jl
git commit -m "feat: rewrite MTKFMIExt to construct FMUSystem instead of faking System equations"
```

---

### Task 8: Integration Tests — Event Handling and Step Rejection

**Files:**
- Create: `test/fmi/fmu_events.jl`
- Modify: `test/runtests.jl`

This task adds integration tests for the new capabilities that the old extension couldn't handle:
FMU events and step rejection.

- [ ] **Step 1: Write event handling integration tests**

```julia
# test/fmi/fmu_events.jl
using Test
using ModelingToolkit, FMI, FMIZoo, OrdinaryDiffEq, SciMLBase
using ModelingToolkit: t_nounits as t, D_nounits as D
import ModelingToolkit as MTK
import ModelingToolkitBase as MTKBase

@testset "FMU Event Handling" begin
    @testset "ME FMU with events" begin
        # Use an FMU that has event indicators (bouncing ball, etc.)
        # The exact FMU depends on what FMIZoo provides
        # BouncingBall has event indicators for the floor contact

        fmu = loadFMU("BouncingBall", "Dymola", "2022x"; type = :ME)
        fmu_sys = MTK.FMIComponent(Val(2); fmu, type = :ME, name = :ball)

        # Verify capabilities were extracted
        caps = MTKBase.get_fmu_capabilities(fmu_sys)
        @test caps.n_event_indicators > 0

        # Verify event callbacks were created
        @test !isempty(MTKBase.get_continuous_events(fmu_sys))

        # Compile and solve
        @mtkcompile sys = System(Equation[], t; systems = [fmu_sys], name = :sys)
        prob = ODEProblem(sys, [], (0.0, 3.0))
        sol = solve(prob, Tsit5())
        @test SciMLBase.successful_retcode(sol)
    end
end
```

- [ ] **Step 2: Write step rejection tests**

```julia
@testset "FMU Step Rejection" begin
    @testset "checkpoint/restore when supported" begin
        fmu = loadFMU("SpringPendulum1D", "Dymola", "2022x"; type = :ME)
        fmu_sys = MTK.FMIComponent(Val(2); fmu, type = :ME, name = :pendulum)

        caps = MTKBase.get_fmu_capabilities(fmu_sys)
        if caps.can_get_and_set_fmu_state
            # Test with a stiff solver that may reject steps
            @mtkcompile sys = System(Equation[], t; systems = [fmu_sys], name = :sys)
            prob = ODEProblem(sys,
                [sys.pendulum.mass__s => 0.5, sys.pendulum.mass__v => 0.0],
                (0.0, 8.0))

            # Use an adaptive solver that will reject some steps
            sol = solve(prob, Rosenbrock23(); reltol = 1e-6, abstol = 1e-6)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end
```

- [ ] **Step 3: Write composed system tests**

```julia
@testset "FMU Composed with MTK System" begin
    @testset "ME FMU driving an MTK oscillator" begin
        fmu = loadFMU("SpringPendulum1D", "Dymola", "2022x"; type = :ME)
        fmu_sys = MTK.FMIComponent(Val(2); fmu, type = :ME, name = :fmu)

        # Create an MTK system that uses the FMU output
        @variables y(t) = 0.0
        parent = System(
            [D(y) ~ -y + fmu_sys.mass__s],
            t;
            systems = [fmu_sys],
            name = :composed
        )
        @mtkcompile sys = parent
        prob = ODEProblem(sys,
            [sys.fmu.mass__s => 0.5, sys.fmu.mass__v => 0.0],
            (0.0, 4.0))
        sol = solve(prob, Tsit5())
        @test SciMLBase.successful_retcode(sol)
    end
end
```

- [ ] **Step 4: Add to runtests.jl**

In `test/runtests.jl`, add the new test files to the FMI group:

```julia
if GROUP == "All" || GROUP == "FMI"
    @safetestset "FMU System Unit Tests" include("fmi/fmusystem_unit.jl")
    @safetestset "FMU Compilation Tests" include("fmi/fmu_compilation.jl")
    @safetestset "FMI Extension Test" include("fmi/fmi.jl")
    @safetestset "FMU Events Test" include("fmi/fmu_events.jl")
end
```

- [ ] **Step 5: Run all FMI tests**

Run: `julia --project -e 'ENV["GROUP"] = "FMI"; include("test/runtests.jl")'`
Expected: All FMI tests pass.

- [ ] **Step 6: Commit**

```bash
git add test/fmi/fmu_events.jl test/runtests.jl
git commit -m "test: add integration tests for FMU events and step rejection"
```

---

## Dependency Graph

```
Task 1 (types) ──→ Task 2 (FMUSystem) ──→ Task 3 (System type param) ──→ Task 4 (fix System[])
                                                                              │
                                                                              ▼
                                                               Task 5 (mtkcompile pass) ──→ Task 6 (codegen)
                                                                                                │
                                                                                                ▼
                                                                                    Task 7 (extension rewrite)
                                                                                                │
                                                                                                ▼
                                                                                    Task 8 (integration tests)
```

Tasks 1-4 are foundational and must be done in order. Tasks 5 and 6 can potentially be
developed in parallel. Task 7 depends on both 5 and 6. Task 8 depends on 7.

## Notes for Implementer

1. **Julia module loading order matters.** The `include` order in `ModelingToolkitBase.jl` and
   `ModelingToolkit.jl` determines what's available. FMU types must be defined before `system.jl`
   if `System` needs to reference `AbstractSystem` (which is already defined in `abstractsystem.jl`).

2. **The `@set!` macro from Setfield.jl** is used extensively for immutable struct updates.
   `FMUSystem` is immutable, so any "mutation" during compilation uses `@set!`.

3. **The `SYS_PROPS` loop** in `abstractsystem.jl` generates `get_*`/`has_*` methods for
   `::AbstractSystem`. Your `FMUSystem`-specific methods must be **more specific** (dispatch on
   `::AbstractFMUSystem`) to override them. Since `AbstractFMUSystem <: AbstractSystem`, this
   works naturally.

4. **The existing FMI tests use `@mtkcompile sys = MTK.FMIComponent(...)`** which compiles
   inline. With the new design, `FMIComponent` returns an `FMUSystem` that then gets composed
   and compiled. The test patterns change accordingly.

5. **FMI library access** (`FMI.fmi2Instantiate!`, etc.) happens only in the extension. The
   codegen stubs in `fmu_codegen.jl` use function stubs that the extension overrides. This
   keeps the dependency chain clean: `ModelingToolkit` doesn't depend on `FMI`.

6. **Step rejection** is the trickiest part. The wrapper needs mutable state for checkpoint
   tracking. The `DiscreteCallback` that handles step rejection must detect time regression
   (solver went back in time after rejection) and restore the FMU state appropriately.

---

## Post-Implementation Notes

The following sections document divergences from the original plan that emerged during
implementation. Reference commits: `ba01c9c1` through `d4b158a7`.

### 1. Custom FMU callback types dropped from active pipeline

**Plan said:** Tasks 1, 2, and 6 defined four custom callback types (`FMUContinuousCallback`,
`FMUTimeCallback`, `FMUStepCallback`, `FMUStepEventCallback`) with a lowering pipeline in
`fmu_codegen.jl` that converted them to standard SciML callbacks.

**What happened:** The lowering pipeline was dropped. FMU callbacks are inherently
non-symbolic (they wrap opaque FMI API calls), so routing them through a custom symbolic
callback pipeline was over-engineered. Instead, the extension builds standard
`SymbolicDiscreteCallback` with `ImperativeAffect` for lifecycle and stepping — the same
approach the old extension used, which integrates naturally with the existing MTK callback
infrastructure.

The four custom types still exist in `types.jl` but are unused. They could be useful later
for FMI event indicator support (opaque zero-crossings that cannot be expressed as symbolic
equations). The `fmu_codegen.jl` stub functions are also unused.

### 2. ME derivative evaluation uses symbolic wrapper callable

**Plan said (Task 5, Step 7):** `merge_fmu_data` would create equations
`D(ns_state) ~ ns_deriv` where `ns_deriv` was the namespaced derivative variable — but
nothing defined what `ns_deriv` actually evaluates to.

**What happened (commit `d4b158a7`):** The extension restores `@register_array_symbolic`
for the wrapper types (`FMI2InstanceWrapper`, `FMI3InstanceWrapper`), creates the wrapper
as a symbolic callable parameter on the `FMUSystem`, and builds observed equations:
```julia
deriv_var ~ wrapper_callable(states, inputs, params, t)[i]
```
These observed equations live on the `FMUSystem` and get namespaced/merged automatically
by `collect_fmu_variables`/`merge_fmu_data`. The `merge_fmu_data` function then creates
`D(ns_state) ~ ns_deriv` which now references properly-defined observed variables.

### 3. FMUSystem constructor simplified

**Plan said (Task 2, Step 3):** The constructor auto-populated event callbacks from
`FMUCapabilities` (creating `FMUContinuousCallback` if `n_event_indicators > 0`,
`FMUStepEventCallback` always for ME, etc.).

**What happened (commit `d4b158a7`):** Auto-callback creation was removed from the
constructor. It now accepts `continuous_events` and `discrete_events` as keyword arguments
(default empty vectors). The extension is responsible for building appropriate callbacks
and passing them in. This is cleaner because callback construction requires extension-specific
knowledge (FMI API calls, wrapper references) that doesn't belong in the base type.

### 4. merge_fmu_data now merges events

**Plan said (Task 5, Step 7):** The `merge_fmu_data` function merged unknowns, parameters,
observed, and equations — but did not merge events.

**What happened (commit `d4b158a7`):** The function was updated to also merge
`continuous_events` and `discrete_events` from FMU subsystems into the compiled system.
Without this, FMU lifecycle callbacks were silently dropped.

### 5. collect_fmu_variables now namespaces callbacks

**Plan said (Task 5, Step 3):** The `collect_fmu_variables` function used bare `append!`
for events, which did not namespace them.

**What happened (commit `d4b158a7`):** Since FMU subsystems are extracted before the
symbolic pipeline runs, their events need explicit namespacing. The collection was fixed
to use `namespace_callback` so that callback variable references are properly scoped.

### 6. CS FMU support deferred

**Plan said (Task 7):** Both ME and CS paths would be fully implemented in the extension.

**What happened:** The extension currently only fully implements the ME path. The CS path
creates an `FMUSystem{CoSimulation}` but does not yet have stepping callbacks. This is
deferred to a follow-up.

### 7. Task 8 (integration tests) partially implemented

**Plan said:** Full event handling and step rejection integration tests.

**What happened (commit `a39af10f`):** Basic integration tests were added, but event
indicator and step rejection tests are deferred pending access to suitable test FMUs
(e.g., BouncingBall with event indicators).
