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
    fmi_version in (2, 3) ||
        throw(ArgumentError("fmi_version must be 2 or 3, got $fmi_version"))
    n_event_indicators >= 0 ||
        throw(ArgumentError("n_event_indicators must be >= 0"))
    return FMUCapabilities(
        can_get_and_set_fmu_state, has_event_mode, n_event_indicators, fmi_version)
end

# ---- Abstract FMU system type ----

"""
    AbstractFMUSystem <: AbstractSystem

Abstract supertype for all FMU system types. Provides a dispatch target for
"any FMU system" regardless of mode.
"""
abstract type AbstractFMUSystem <: AbstractSystem end

# ---- FMU Callback types ----

struct FMUContinuousCallback{W}
    wrapper::W
    n_event_indicators::Int
end

struct FMUTimeCallback{W}
    wrapper::W
end

struct FMUStepCallback{W}
    wrapper::W
    communication_step_size::Float64
end

struct FMUStepEventCallback{W}
    wrapper::W
end
