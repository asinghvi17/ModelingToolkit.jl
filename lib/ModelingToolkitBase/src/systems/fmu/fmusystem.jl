# ---- FMUSystem: concrete AbstractFMUSystem implementation ----

"""
    FMUSystem{Mode <: FMUMode, WrapperType, ValRefType}

A symbolic system backed by an FMU (Functional Mock-up Unit). This struct implements
the `AbstractSystem` interface, enabling FMU-backed components to participate in
ModelingToolkit's hierarchical system composition and compilation pipeline.

# Type parameters
- `Mode`: `ModelExchange` or `CoSimulation`
- `WrapperType`: the type of the FMU wrapper object
- `ValRefType`: the type of the value reference mapping

# Fields
$(TYPEDFIELDS)
"""
struct FMUSystem{Mode <: FMUMode, WrapperType, ValRefType} <: AbstractFMUSystem
    "The name of the system."
    name::Symbol
    "The independent variable (time)."
    iv::SymbolicT
    "State (unknown) variables of the FMU."
    states::Vector{SymbolicT}
    "Derivative variables (ME only)."
    derivatives::Vector{SymbolicT}
    "Input variables fed into the FMU."
    inputs::Vector{SymbolicT}
    "Output variables read from the FMU."
    outputs::Vector{SymbolicT}
    "Parameters of the FMU."
    parameters::Vector{SymbolicT}
    "Observed equations (algebraic relations derived from FMU outputs)."
    observed::Vector{Equation}
    "The FMU wrapper object (e.g., FMU2 or FMU3 handle)."
    wrapper::WrapperType
    "Capabilities extracted from the FMU model description."
    capabilities::FMUCapabilities
    "Mapping from symbolic variables to FMU value references."
    value_references::ValRefType
    "Default values for variables and parameters."
    default_values::Dict{SymbolicT, Any}
    "Continuous event callbacks."
    continuous_events::Vector{Any}
    "Discrete event callbacks."
    discrete_events::Vector{Any}
    "Communication step size (CS only; nothing for ME)."
    communication_step_size::Union{Nothing, Float64}
    "Subsystems."
    systems::Vector{AbstractSystem}
    "Parent system after simplification."
    parent::Union{Nothing, AbstractSystem}
    "Whether the system performs namespacing."
    namespacing::Bool
    "Whether the system has been marked complete."
    complete::Bool
    "Initialization equations."
    initialization_eqs::Vector{Equation}
end

"""
    FMUSystem{Mode}(; name, iv, states, derivatives, inputs, outputs, parameters,
                      observed, wrapper, capabilities, value_references,
                      default_values, communication_step_size, kwargs...)

Construct an `FMUSystem` with validation based on the FMU mode.

For `ModelExchange`:
- `length(derivatives)` must equal `length(states)`
- `communication_step_size` must be `nothing`

For `CoSimulation`:
- `derivatives` must be empty
- `communication_step_size` is required (must not be `nothing`)
"""
function FMUSystem{Mode}(;
        name::Symbol,
        iv,
        states::Vector{<:SymbolicT},
        derivatives::Vector{<:SymbolicT},
        inputs::Vector{<:SymbolicT},
        outputs::Vector{<:SymbolicT},
        parameters::Vector{<:SymbolicT},
        observed::Vector{Equation} = Equation[],
        wrapper,
        capabilities::FMUCapabilities,
        value_references,
        default_values::Dict{<:SymbolicT, Any} = Dict{SymbolicT, Any}(),
        communication_step_size::Union{Nothing, Float64} = nothing,
        systems::Vector{<:AbstractSystem} = AbstractSystem[],
        parent::Union{Nothing, AbstractSystem} = nothing,
        namespacing::Bool = true,
        complete::Bool = false,
        initialization_eqs::Vector{Equation} = Equation[],
        continuous_events::Vector{Any} = Any[],
        discrete_events::Vector{Any} = Any[],
    ) where {Mode <: FMUMode}

    # Mode-specific validation
    if Mode === ModelExchange
        length(derivatives) == length(states) ||
            throw(ArgumentError(
                "ModelExchange FMU requires length(derivatives) == length(states), " *
                "got $(length(derivatives)) != $(length(states))"))
        communication_step_size === nothing ||
            throw(ArgumentError(
                "ModelExchange FMU must have communication_step_size = nothing"))
    elseif Mode === CoSimulation
        isempty(derivatives) ||
            throw(ArgumentError(
                "CoSimulation FMU must have empty derivatives, got $(length(derivatives))"))
        communication_step_size !== nothing ||
            throw(ArgumentError(
                "CoSimulation FMU requires a communication_step_size"))
    end

    # Events are built by the extension and passed in

    return FMUSystem{Mode, typeof(wrapper), typeof(value_references)}(
        name,
        iv,
        convert(Vector{SymbolicT}, states),
        convert(Vector{SymbolicT}, derivatives),
        convert(Vector{SymbolicT}, inputs),
        convert(Vector{SymbolicT}, outputs),
        convert(Vector{SymbolicT}, parameters),
        observed,
        wrapper,
        capabilities,
        value_references,
        convert(Dict{SymbolicT, Any}, default_values),
        continuous_events,
        discrete_events,
        communication_step_size,
        convert(Vector{AbstractSystem}, systems),
        parent,
        namespacing,
        complete,
        initialization_eqs,
    )
end

# ---- AbstractSystem interface: required get_* / has_* methods ----

# Fields that FMUSystem actually has
Base.nameof(sys::AbstractFMUSystem) = getfield(sys, :name)

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

get_continuous_events(sys::AbstractFMUSystem) = getfield(sys, :continuous_events)
has_continuous_events(::AbstractFMUSystem) = true

get_discrete_events(sys::AbstractFMUSystem) = getfield(sys, :discrete_events)
has_discrete_events(::AbstractFMUSystem) = true

get_initialization_eqs(sys::AbstractFMUSystem) = getfield(sys, :initialization_eqs)
has_initialization_eqs(::AbstractFMUSystem) = true

get_name(sys::AbstractFMUSystem) = getfield(sys, :name)
has_name(::AbstractFMUSystem) = true

get_parent(sys::AbstractFMUSystem) = getfield(sys, :parent)
has_parent(::AbstractFMUSystem) = true

get_inputs(sys::AbstractFMUSystem) = getfield(sys, :inputs)
has_inputs(::AbstractFMUSystem) = true

get_outputs(sys::AbstractFMUSystem) = getfield(sys, :outputs)
has_outputs(::AbstractFMUSystem) = true

# Equations: FMU systems have no symbolic equations
get_eqs(::AbstractFMUSystem) = Equation[]
has_eqs(::AbstractFMUSystem) = true

# Completeness and scheduling
iscomplete(sys::AbstractFMUSystem) = getfield(sys, :complete)
does_namespacing(sys::AbstractFMUSystem) = getfield(sys, :namespacing)
get_isscheduled(::AbstractFMUSystem) = false
has_isscheduled(::AbstractFMUSystem) = true

# Time dependence
SymbolicIndexingInterface.is_time_dependent(::AbstractFMUSystem) = true

# ---- Properties that don't apply to FMU systems: has_* returns false ----

has_tag(::AbstractFMUSystem) = false
has_noise_eqs(::AbstractFMUSystem) = false
has_tspan(::AbstractFMUSystem) = false
has_brownians(::AbstractFMUSystem) = false
has_poissonians(::AbstractFMUSystem) = false
has_jumps(::AbstractFMUSystem) = false
has_description(::AbstractFMUSystem) = false
has_var_to_name(::AbstractFMUSystem) = false
has_bindings(::AbstractFMUSystem) = false
has_initial_conditions(::AbstractFMUSystem) = false
has_guesses(::AbstractFMUSystem) = false
has_constraints(::AbstractFMUSystem) = false
has_bcs(::AbstractFMUSystem) = false
has_domain(::AbstractFMUSystem) = false
has_ivs(::AbstractFMUSystem) = false
has_dvs(::AbstractFMUSystem) = false
has_connector_type(::AbstractFMUSystem) = false
has_preface(::AbstractFMUSystem) = false
has_initializesystem(::AbstractFMUSystem) = false
has_schedule(::AbstractFMUSystem) = false
has_tearing_state(::AbstractFMUSystem) = false
has_metadata(::AbstractFMUSystem) = false
has_gui_metadata(::AbstractFMUSystem) = false
has_is_initializesystem(::AbstractFMUSystem) = false
has_is_discrete(::AbstractFMUSystem) = false
has_state_priorities(::AbstractFMUSystem) = false
has_irreducibles(::AbstractFMUSystem) = false
has_maybe_zeros(::AbstractFMUSystem) = false
has_assertions(::AbstractFMUSystem) = false
has_ignored_connections(::AbstractFMUSystem) = false
has_is_dde(::AbstractFMUSystem) = false
has_tstops(::AbstractFMUSystem) = false
has_index_cache(::AbstractFMUSystem) = false
has_parameter_bindings_graph(::AbstractFMUSystem) = false
has_costs(::AbstractFMUSystem) = false
has_consolidate(::AbstractFMUSystem) = false

# ---- Sensible default get_* for inapplicable properties ----
# These are needed because some generic code paths call get_* without checking has_*.

get_tag(::AbstractFMUSystem) = UInt(0)
get_noise_eqs(::AbstractFMUSystem) = nothing
get_tspan(::AbstractFMUSystem) = nothing
get_brownians(::AbstractFMUSystem) = SymbolicT[]
get_poissonians(::AbstractFMUSystem) = SymbolicT[]
get_jumps(::AbstractFMUSystem) = Any[]
get_description(::AbstractFMUSystem) = ""
get_var_to_name(::AbstractFMUSystem) = Dict{Symbol, SymbolicT}()
get_bindings(::AbstractFMUSystem) = ROSymmapT(SymmapT())
get_initial_conditions(::AbstractFMUSystem) = SymmapT()
get_guesses(::AbstractFMUSystem) = SymmapT()
get_constraints(::AbstractFMUSystem) = Union{Equation, Inequality}[]
get_bcs(::AbstractFMUSystem) = Equation[]
get_domain(::AbstractFMUSystem) = nothing
get_ivs(::AbstractFMUSystem) = SymbolicT[]
get_dvs(::AbstractFMUSystem) = SymbolicT[]
get_connector_type(::AbstractFMUSystem) = nothing
get_preface(::AbstractFMUSystem) = nothing
get_initializesystem(::AbstractFMUSystem) = nothing
get_schedule(::AbstractFMUSystem) = nothing
get_tearing_state(::AbstractFMUSystem) = nothing
get_metadata(::AbstractFMUSystem) = Base.ImmutableDict{DataType, Any}()
get_gui_metadata(::AbstractFMUSystem) = nothing
get_is_initializesystem(::AbstractFMUSystem) = false
get_is_discrete(::AbstractFMUSystem) = false
get_state_priorities(::AbstractFMUSystem) = AtomicMapT{Int}()
get_irreducibles(::AbstractFMUSystem) = AtomicSetT()
get_maybe_zeros(::AbstractFMUSystem) = AtomicSetT()
get_assertions(::AbstractFMUSystem) = Dict{SymbolicT, String}()
get_ignored_connections(::AbstractFMUSystem) = nothing
get_is_dde(::AbstractFMUSystem) = false
get_tstops(::AbstractFMUSystem) = Any[]
get_index_cache(::AbstractFMUSystem) = nothing
get_parameter_bindings_graph(::AbstractFMUSystem) = nothing
get_costs(::AbstractFMUSystem) = SymbolicT[]
get_consolidate(::AbstractFMUSystem) = nothing

# FMU callback types don't have symbolic variables to namespace — pass through as-is
namespace_callback(cb::FMUContinuousCallback, s) = cb
namespace_callback(cb::FMUTimeCallback, s) = cb
namespace_callback(cb::FMUStepCallback, s) = cb
namespace_callback(cb::FMUStepEventCallback, s) = cb

# ---- FMU-specific accessors ----

"""Get the FMU wrapper object."""
get_fmu_wrapper(sys::AbstractFMUSystem) = getfield(sys, :wrapper)

"""Get the FMU capabilities."""
get_fmu_capabilities(sys::AbstractFMUSystem) = getfield(sys, :capabilities)

"""Get the value reference mapping."""
get_value_references(sys::AbstractFMUSystem) = getfield(sys, :value_references)

"""Get the default values dictionary."""
get_default_values(sys::AbstractFMUSystem) = getfield(sys, :default_values)

"""Get the derivative variables (ME only)."""
get_derivatives(sys::AbstractFMUSystem) = getfield(sys, :derivatives)

"""Get the communication step size (CS only)."""
get_communication_step_size(sys::AbstractFMUSystem) = getfield(sys, :communication_step_size)
