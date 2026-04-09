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
        parent_sys = System([D(x) ~ x], t; systems = [fmu_sys], name = :parent)

        fmu_subsystems, regular_sys = ModelingToolkit.extract_fmu_subsystems(parent_sys)
        @test length(fmu_subsystems) == 1
        @test fmu_subsystems[1] isa MTKBase.FMUSystem
        @test length(MTKBase.get_systems(regular_sys)) == 0
    end

    @testset "extract_fmu_subsystems - no FMUs" begin
        @variables x(t) = 1.0
        parent_sys = System([D(x) ~ x], t; name = :parent)

        fmu_subsystems, regular_sys = ModelingToolkit.extract_fmu_subsystems(parent_sys)
        @test isempty(fmu_subsystems)
        # System should be unchanged when no FMUs present
        @test length(MTKBase.get_systems(regular_sys)) == 0
    end

    @testset "collect_fmu_variables" begin
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
            inputs = SymbolicT[],
            outputs = SymbolicT[],
            parameters = [p1],
            wrapper = :mock,
            capabilities = caps,
            value_references = Dict{SymbolicT, UInt32}(
                s1 => UInt32(0), ds1 => UInt32(1), p1 => UInt32(2)),
            default_values = Dict{SymbolicT, Any}()
        )

        @variables x(t) = 1.0
        parent_sys = System([D(x) ~ x], t; name = :parent)

        fmu_data = ModelingToolkit.collect_fmu_variables(parent_sys, [fmu_sys])
        @test length(fmu_data.unknowns) == 1
        @test length(fmu_data.parameters) == 1
        @test isempty(fmu_data.observed)
        @test isempty(fmu_data.continuous_events)
        @test isempty(fmu_data.discrete_events)
    end

    @testset "FMUSubsystemsKey struct exists" begin
        @test isdefined(ModelingToolkit, :FMUSubsystemsKey)
    end
end
