# test/fmi/fmu_events.jl
#
# Integration tests for the FMUSystem end-to-end pipeline using Reference FMUs.

using Test
using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
import ModelingToolkit as MTK
import ModelingToolkitBase as MTKBase
import FMI

const REF_FMU_DIR = joinpath(@__DIR__, "..", "..", "..", "Reference-FMUs",
    "build", "fmi3-aarch64-darwin", "install")

# Only run if Reference FMUs are available
if isdir(REF_FMU_DIR)

@testset "FMU Pipeline - Reference FMUs" begin
    @testset "Dahlquist (dx/dt = -kx)" begin
        fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "Dahlquist.fmu"); type = :ME)
        fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :dahlquist)

        @test fmu_sys isa MTKBase.FMUSystem{MTKBase.ModelExchange}
        @test length(MTKBase.get_unknowns(fmu_sys)) == 1
        @test length(MTKBase.get_derivatives(fmu_sys)) == 1

        parent = System(Equation[], t; systems = [fmu_sys], name = :sys)
        compiled = mtkcompile(parent)

        @test length(unknowns(compiled)) == 1
        @test length(equations(compiled)) == 1

        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            compiled, [compiled.dahlquist.x => 1.0], (0.0, 1.0);
            build_initializeprob = false
        )
        sol = solve(prob, Tsit5(); reltol = 1e-8, abstol = 1e-8)
        @test SciMLBase.successful_retcode(sol)

        # Analytical solution: x(t) = exp(-k*t), k=1 (default)
        @test sol[end, end] ≈ exp(-1.0) atol = 1e-6
    end

    @testset "VanDerPol oscillator" begin
        fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "VanDerPol.fmu"); type = :ME)
        fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :vdp)

        @test fmu_sys isa MTKBase.FMUSystem{MTKBase.ModelExchange}
        @test length(MTKBase.get_unknowns(fmu_sys)) == 2

        parent = System(Equation[], t; systems = [fmu_sys], name = :sys)
        compiled = mtkcompile(parent)

        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            compiled,
            [compiled.vdp.x0 => 2.0, compiled.vdp.x1 => 0.0],
            (0.0, 5.0);
            build_initializeprob = false
        )
        sol = solve(prob, Tsit5(); reltol = 1e-6, abstol = 1e-6)
        @test SciMLBase.successful_retcode(sol)
        @test length(sol.t) > 10  # should take multiple steps
    end

    # TODO: Composed MTK+FMU test (FMU variables referenced in MTK equations
    # need additional integration work — the variable reference fmu.x is not
    # available during the symbolic compilation pass before FMU merge)

    @testset "Multiple FMU subsystems" begin
        fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "Dahlquist.fmu"); type = :ME)
        fmu1 = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :fmu1)
        fmu2 = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :fmu2)

        parent = System(Equation[], t; systems = [fmu1, fmu2], name = :sys)
        compiled = mtkcompile(parent)

        @test length(unknowns(compiled)) == 2  # one state from each FMU

        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            compiled,
            [compiled.fmu1.x => 1.0, compiled.fmu2.x => 2.0],
            (0.0, 1.0);
            build_initializeprob = false
        )
        sol = solve(prob, Tsit5(); reltol = 1e-8, abstol = 1e-8)
        @test SciMLBase.successful_retcode(sol)

        # Both should decay exponentially with k=1
        @test sol[compiled.fmu1.x][end] ≈ exp(-1.0) atol = 1e-6
        @test sol[compiled.fmu2.x][end] ≈ 2.0 * exp(-1.0) atol = 1e-6
    end

    @testset "Repeated solve" begin
        fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "Dahlquist.fmu"); type = :ME)
        fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :dahlquist)
        parent = System(Equation[], t; systems = [fmu_sys], name = :sys)
        compiled = mtkcompile(parent)

        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            compiled, [compiled.dahlquist.x => 1.0], (0.0, 1.0);
            build_initializeprob = false
        )
        sol1 = solve(prob, Tsit5())
        @test SciMLBase.successful_retcode(sol1)

        # Second solve should also work
        sol2 = solve(prob, Tsit5())
        @test SciMLBase.successful_retcode(sol2)
    end

    @testset "Composed MTK+FMU equations" begin
        fmu = FMI.loadFMU(joinpath(REF_FMU_DIR, "Dahlquist.fmu"); type = :ME)
        fmu_sys = MTK.FMIComponent(Val(3); fmu, type = :ME, name = :fmu)

        @variables y(t)
        eqs = [D(y) ~ -y + fmu_sys.x]
        parent = System(eqs, t; systems = [fmu_sys], name = :sys)
        compiled = mtkcompile(parent)

        @test length(unknowns(compiled)) == 2  # y and fmu.x

        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            compiled,
            [compiled.y => 0.0, compiled.fmu.x => 1.0],
            (0.0, 1.0);
            build_initializeprob = false
        )
        sol = solve(prob, Tsit5(); reltol = 1e-8, abstol = 1e-8)
        @test SciMLBase.successful_retcode(sol)

        # fmu.x decays as exp(-t), y is driven by it
        # fmu.x(1) ≈ exp(-1)
        @test sol[compiled.fmu.x][end] ≈ exp(-1.0) atol = 1e-6
    end
end

else
    @info "Skipping Reference FMU tests: directory not found at $REF_FMU_DIR"
end
