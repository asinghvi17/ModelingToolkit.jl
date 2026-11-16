using ModelingToolkitBase, Symbolics, ForwardDiff, Test
using ModelingToolkitBase: t_nounits as t, D_nounits as D
using Symbolics: unwrap
import SymbolicUtils as SU
using LinearAlgebra: dot

# `f.tgrad` must agree with the forward-mode derivative of `f` in `t`.
function tgrad_matches_fd(sys, op)
    prob = ODEProblem(sys, op, (0.0, 1.0); tgrad = true)
    u, p = prob.u0, prob.p
    return all((0.3, 0.7)) do τ
        g = similar(u)
        prob.f.tgrad(g, u, p, τ)
        isapprox(g, ForwardDiff.derivative(s -> prob.f(u, p, s), τ); rtol = 1.0e-10)
    end
end

@testset "time-varying (discrete) parameters are constants of the t-derivative" begin
    # `k(t)` is detimed like an unknown; the round trip must restore it, or the
    # generated code references an unbound `k`.
    @discretes k(t)
    @variables A(t)
    @named sys = System([D(A) ~ -k * A + t^2 * A], t, [A], [k])
    sys = mtkcompile(sys)
    @test isequal(calculate_tgrad(sys), [2t * A])
    @test tgrad_matches_fd(sys, [A => 1.5, k => 0.3])
end

@testset "array unknowns inside `dot`/broadcast terms" begin
    @variables x(t)[1:3]
    @parameters k[1:3]
    @named sys = System([D(x) ~ -x .+ dot(k, x) * t], t)
    sys = mtkcompile(sys)
    @test tgrad_matches_fd(sys, [x => [1.0, 2.0, 3.0], k => [0.1, 0.2, 0.3]])
end

@testset "symbolically indexed array unknown" begin
    # `x[i]` keeps the parent array `x(t)` through scalarization; it must be
    # restored to the term of the equations, not left as the detimed `x`.
    @variables x(t)[1:3] y(t)
    @parameters i::Int
    @named sys = System([D(x) ~ -x .* t, D(y) ~ x[i] * t^2], t)
    sys = complete(sys)
    tg = calculate_tgrad(sys)
    @test isequal(tg[4], 2t * x[i])
    xi = only(v for v in SU.search_variables(unwrap(tg[4])) if SU.iscall(v))
    xt = only(v for v in SU.search_variables(unwrap(full_equations(sys)[4].rhs)) if SU.iscall(v))
    @test SU.arguments(xi)[1] === SU.arguments(xt)[1]
end

@testset "`_tgrad_const_indexed`" begin
    @variables x(t)[1:3]
    @parameters i::Int
    @test ModelingToolkitBase._tgrad_const_indexed(unwrap(x[2]))
    @test !ModelingToolkitBase._tgrad_const_indexed(unwrap(x[i]))
end

@testset "tgrad of a deep shared expression DAG" begin
    @variables A(t)
    e = A * t
    for _ in 1:30
        e = sin(e) + cos(e) * t
    end
    @named sys = System([D(A) ~ e], t)
    sys = complete(sys)
    @test @elapsed(calculate_tgrad(sys)) < 60
end
