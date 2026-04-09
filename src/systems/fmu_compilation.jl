# src/systems/fmu_compilation.jl

"""
    extract_fmu_subsystems(sys)

Walk the system hierarchy and separate FMU subsystems from regular subsystems.
Returns `(fmu_subsystems, sys_without_fmus)` where:
- `fmu_subsystems` is a vector of `AbstractFMUSystem`
- `sys_without_fmus` is the system with FMU subsystems removed
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
"""
function collect_fmu_variables(sys, fmu_subsystems)
    fmu_unknowns = SymbolicT[]
    fmu_parameters = SymbolicT[]
    fmu_observed = Equation[]
    fmu_continuous_events = []
    fmu_discrete_events = []
    fmu_defaults = Dict{SymbolicT, Any}()

    for fmu in fmu_subsystems
        for v in get_unknowns(fmu)
            push!(fmu_unknowns, renamespace(fmu, v))
        end
        for p in get_ps(fmu)
            push!(fmu_parameters, renamespace(fmu, p))
        end
        for eq in get_observed(fmu)
            push!(fmu_observed, namespace_equation(eq, fmu))
        end
        for cb in get_continuous_events(fmu)
            push!(fmu_continuous_events, namespace_callback(cb, fmu))
        end
        for cb in get_discrete_events(fmu)
            push!(fmu_discrete_events, namespace_callback(cb, fmu))
        end
        # Collect defaults (parameter values, state initial conditions)
        for (var, val) in get_default_values(fmu)
            fmu_defaults[renamespace(fmu, var)] = val
        end
    end

    return (;
        unknowns = fmu_unknowns,
        parameters = fmu_parameters,
        observed = fmu_observed,
        continuous_events = fmu_continuous_events,
        discrete_events = fmu_discrete_events,
        defaults = fmu_defaults
    )
end

"""
    merge_fmu_data(compiled_sys, fmu_data, fmu_subsystems)

Merge FMU symbolic variables and events into the compiled system.
"""
function merge_fmu_data(compiled_sys, fmu_data, fmu_subsystems)
    # Filter out FMU states that were temporarily injected as parameters
    fmu_state_set = Set(fmu_data.unknowns)
    compiled_ps = filter(p -> p ∉ fmu_state_set, get_ps(compiled_sys))
    new_ps = vcat(compiled_ps, fmu_data.parameters)
    new_observed = vcat(get_observed(compiled_sys), fmu_data.observed)
    new_unknowns = copy(get_unknowns(compiled_sys))
    new_eqs = copy(get_eqs(compiled_sys))

    for fmu in fmu_subsystems
        if fmu isa FMUSystem{ModelExchange}
            for (state, deriv) in zip(get_unknowns(fmu), get_derivatives(fmu))
                ns_state = renamespace(fmu, state)
                ns_deriv = renamespace(fmu, deriv)
                push!(new_unknowns, ns_state)
                push!(new_eqs, Differential(get_iv(compiled_sys))(ns_state) ~ ns_deriv)
            end
        elseif fmu isa FMUSystem{CoSimulation}
            for state in get_unknowns(fmu)
                push!(new_ps, renamespace(fmu, state))
            end
        end
    end

    fmu_metadata = Base.ImmutableDict(get_metadata(compiled_sys),
        FMUSubsystemsKey => fmu_subsystems)

    @set! compiled_sys.eqs = new_eqs
    @set! compiled_sys.unknowns = new_unknowns
    @set! compiled_sys.ps = new_ps
    @set! compiled_sys.observed = new_observed

    # Merge FMU events into compiled system
    if !isempty(fmu_data.continuous_events)
        new_cont = vcat(get_continuous_events(compiled_sys), fmu_data.continuous_events)
        @set! compiled_sys.continuous_events = new_cont
    end
    if !isempty(fmu_data.discrete_events)
        new_disc = vcat(get_discrete_events(compiled_sys), fmu_data.discrete_events)
        @set! compiled_sys.discrete_events = new_disc
    end

    @set! compiled_sys.metadata = fmu_metadata

    # Merge FMU defaults into compiled system's initial_conditions
    if !isempty(fmu_data.defaults)
        ic = copy(initial_conditions(compiled_sys).dict)
        merge!(ic, fmu_data.defaults)
        @set! compiled_sys.initial_conditions = MTKBase.AtomicArrayDict(ic)
    end

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

    return compiled_sys
end

"""Metadata key for storing FMU subsystem references on the compiled system."""
struct FMUSubsystemsKey end
