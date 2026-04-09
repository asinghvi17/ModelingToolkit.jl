# src/systems/fmu_codegen.jl

"""
    lower_fmu_continuous_callback(cb::FMUContinuousCallback)

Lower an FMU continuous callback to a `VectorContinuousCallback`.
The condition calls `fmu_get_event_indicators!`, and the affect enters event mode
and iterates `UpdateDiscreteStates`.
"""
function lower_fmu_continuous_callback(cb::FMUContinuousCallback)
    wrapper = cb.wrapper
    n = cb.n_event_indicators

    condition = function (out, u, t, integrator)
        fmu_get_event_indicators!(wrapper, out, u, t)
    end

    affect = function (integrator, idx)
        fmu_enter_event_mode!(wrapper)
        info = nothing
        while true
            info = fmu_update_discrete_states!(wrapper)
            info.newDiscreteStatesNeeded || break
            if info.terminateSimulation
                SciMLBase.terminate!(integrator)
                return
            end
        end
        if info !== nothing && info.valuesOfContinuousStatesChanged
            fmu_get_continuous_states!(wrapper, integrator.u)
            SciMLBase.u_modified!(integrator, true)
        end
        if info !== nothing && info.nextEventTimeDefined
            SciMLBase.add_tstop!(integrator, info.nextEventTime)
        end
        fmu_enter_continuous_time_mode!(wrapper)
    end

    return SciMLBase.VectorContinuousCallback(condition, affect, n)
end

"""
    lower_fmu_time_callback(cb::FMUTimeCallback)

Lower an FMU time callback to a `DiscreteCallback` that fires at FMU-requested
time stops.
"""
function lower_fmu_time_callback(cb::FMUTimeCallback)
    wrapper = cb.wrapper

    condition = function (u, t, integrator)
        fmu_has_pending_time_event(wrapper, t)
    end

    affect = function (integrator)
        fmu_enter_event_mode!(wrapper)
        info = nothing
        while true
            info = fmu_update_discrete_states!(wrapper)
            info.newDiscreteStatesNeeded || break
            if info.terminateSimulation
                SciMLBase.terminate!(integrator)
                return
            end
        end
        if info !== nothing && info.valuesOfContinuousStatesChanged
            fmu_get_continuous_states!(wrapper, integrator.u)
            SciMLBase.u_modified!(integrator, true)
        end
        if info !== nothing && info.nextEventTimeDefined
            SciMLBase.add_tstop!(integrator, info.nextEventTime)
        end
        fmu_enter_continuous_time_mode!(wrapper)
    end

    return SciMLBase.DiscreteCallback(condition, affect)
end

"""
    lower_fmu_step_callback(cb::FMUStepCallback)

Lower a CS FMU step callback to a `PeriodicCallback`.
"""
function lower_fmu_step_callback(cb::FMUStepCallback)
    wrapper = cb.wrapper
    dt = cb.communication_step_size

    affect = function (integrator)
        fmu_do_step!(wrapper, integrator.t - dt, dt)
        fmu_read_outputs!(wrapper, integrator)
        SciMLBase.u_modified!(integrator, true)
    end

    return MTKBase.DiffEqCallbacks.PeriodicCallback(affect, dt)
end

"""
    lower_fmu_step_event_callback(cb::FMUStepEventCallback)

Lower an ME FMU step event callback to a `DiscreteCallback` that checks
`CompletedIntegratorStep` at each accepted solver step.
"""
function lower_fmu_step_event_callback(cb::FMUStepEventCallback)
    wrapper = cb.wrapper

    condition = (u, t, integrator) -> true

    affect = function (integrator)
        enter_event_mode = fmu_completed_integrator_step!(wrapper)
        if enter_event_mode
            fmu_enter_event_mode!(wrapper)
            info = nothing
            while true
                info = fmu_update_discrete_states!(wrapper)
                info.newDiscreteStatesNeeded || break
                if info.terminateSimulation
                    SciMLBase.terminate!(integrator)
                    return
                end
            end
            if info !== nothing && info.valuesOfContinuousStatesChanged
                fmu_get_continuous_states!(wrapper, integrator.u)
                SciMLBase.u_modified!(integrator, true)
            end
            if info !== nothing && info.nextEventTimeDefined
                SciMLBase.add_tstop!(integrator, info.nextEventTime)
            end
            fmu_enter_continuous_time_mode!(wrapper)
        end
    end

    return SciMLBase.DiscreteCallback(condition, affect)
end

# ---- Stub functions ----
# These are overridden by the FMI extension (MTKFMIExt) with actual FMI calls.

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
