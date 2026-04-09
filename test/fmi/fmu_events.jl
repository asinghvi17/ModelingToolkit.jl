# test/fmi/fmu_events.jl
using Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
import ModelingToolkit as MTK
import ModelingToolkitBase as MTKBase
using FMI, FMIZoo

@testset "FMU Event Handling" begin
    @testset "ME FMU construction returns FMUSystem" begin
        fmu = loadFMU("SpringPendulum1D", "Dymola", "2022x"; type = :ME)
        fmu_sys = MTK.FMIComponent(Val(2); fmu, type = :ME, name = :pendulum)

        @test fmu_sys isa MTKBase.FMUSystem{MTKBase.ModelExchange}
        @test nameof(fmu_sys) === :pendulum
        @test MTKBase.is_time_dependent(fmu_sys)

        # Verify capabilities were extracted
        caps = MTKBase.get_fmu_capabilities(fmu_sys)
        @test caps.fmi_version == 2
        @test caps.n_event_indicators >= 0

        # Verify states and derivatives
        @test !isempty(MTKBase.get_unknowns(fmu_sys))
        @test !isempty(MTKBase.get_derivatives(fmu_sys))
        @test length(MTKBase.get_unknowns(fmu_sys)) == length(MTKBase.get_derivatives(fmu_sys))
    end

    @testset "CS FMU construction returns FMUSystem" begin
        fmu = loadFMU("SpringPendulum1D", "Dymola", "2022x"; type = :CS)
        fmu_sys = MTK.FMIComponent(
            Val(2); fmu, type = :CS, communication_step_size = 1e-3, name = :pendulum_cs
        )

        @test fmu_sys isa MTKBase.FMUSystem{MTKBase.CoSimulation}
        @test MTKBase.get_communication_step_size(fmu_sys) == 1e-3
        @test isempty(MTKBase.get_derivatives(fmu_sys))
    end

    @testset "FMU as subsystem of System" begin
        fmu = loadFMU("SpringPendulum1D", "Dymola", "2022x"; type = :ME)
        fmu_sys = MTK.FMIComponent(Val(2); fmu, type = :ME, name = :fmu)

        @variables x(t) = 1.0
        parent = System([D(x) ~ x], t; systems = [fmu_sys], name = :parent)

        @test length(MTKBase.get_systems(parent)) == 1
        @test MTKBase.get_systems(parent)[1] isa MTKBase.AbstractFMUSystem

        # Test FMU extraction pass
        fmu_subsystems, regular_sys = MTK.extract_fmu_subsystems(parent)
        @test length(fmu_subsystems) == 1
        @test fmu_subsystems[1] === fmu_sys
        @test length(MTKBase.get_systems(regular_sys)) == 0
    end

    @testset "Multiple FMU subsystems" begin
        fmu = loadFMU("SpringPendulum1D", "Dymola", "2022x"; type = :ME)
        fmu1 = MTK.FMIComponent(Val(2); fmu, type = :ME, name = :fmu1)
        fmu2 = MTK.FMIComponent(Val(2); fmu, type = :ME, name = :fmu2)

        @variables x(t) = 1.0
        parent = System([D(x) ~ x], t; systems = [fmu1, fmu2], name = :parent)

        fmu_subsystems, regular_sys = MTK.extract_fmu_subsystems(parent)
        @test length(fmu_subsystems) == 2
        @test length(MTKBase.get_systems(regular_sys)) == 0
    end

    @testset "FMU callback lowering structure" begin
        # Test that callbacks are correctly created for ME FMUs with event indicators
        fmu = loadFMU("SpringPendulum1D", "Dymola", "2022x"; type = :ME)
        fmu_sys = MTK.FMIComponent(Val(2); fmu, type = :ME, name = :fmu)

        # Check discrete events (step event callback, time callback) are present for ME
        disc_events = MTKBase.get_discrete_events(fmu_sys)
        @test !isempty(disc_events)
        @test any(e -> e isa MTKBase.FMUStepEventCallback, disc_events)
        @test any(e -> e isa MTKBase.FMUTimeCallback, disc_events)

        # If event indicators exist, continuous callback should be present
        caps = MTKBase.get_fmu_capabilities(fmu_sys)
        cont_events = MTKBase.get_continuous_events(fmu_sys)
        if caps.n_event_indicators > 0
            @test !isempty(cont_events)
            @test cont_events[1] isa MTKBase.FMUContinuousCallback
            @test cont_events[1].n_event_indicators == caps.n_event_indicators
        end
    end

    @testset "v3 FMU construction" begin
        fmu = loadFMU("SpringPendulum1D", "Dymola", "2023x", "3.0"; type = :ME)
        fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :pendulum_v3)

        @test fmu_sys isa MTKBase.FMUSystem{MTKBase.ModelExchange}
        caps = MTKBase.get_fmu_capabilities(fmu_sys)
        @test caps.fmi_version == 3
        @test !isempty(MTKBase.get_unknowns(fmu_sys))
    end
end
