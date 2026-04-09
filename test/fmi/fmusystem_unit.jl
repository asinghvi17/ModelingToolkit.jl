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
            inputs = typeof(s1)[],
            outputs = typeof(s1)[],
            parameters = [p1],
            observed = Equation[],
            wrapper = wrapper,
            capabilities = caps,
            value_references = valrefs,
            default_values = Dict{typeof(s1), Any}(s1 => 0.5, s2 => 0.0),
            communication_step_size = nothing
        )

        @test nameof(sys) === :fmu
        @test MTKBase.get_iv(sys) === iv
        @test isequal(MTKBase.get_unknowns(sys), [s1, s2])
        @test isequal(MTKBase.get_ps(sys), [p1])
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
        inputs = typeof(s1)[],
        outputs = typeof(s1)[],
        parameters = [p1],
        wrapper = :mock,
        capabilities = caps,
        value_references = Dict{typeof(s1), UInt32}(
            s1 => UInt32(0), ds1 => UInt32(1), p1 => UInt32(2)),
        default_values = Dict{typeof(s1), Any}()
    )

    @variables x(t) = 1.0
    parent_sys = System([D(x) ~ x], t; systems = [fmu_sys], name = :parent)
    @test length(MTKBase.get_systems(parent_sys)) == 1
    @test MTKBase.get_systems(parent_sys)[1] isa MTKBase.AbstractFMUSystem
end
