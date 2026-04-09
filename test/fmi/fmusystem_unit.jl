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
