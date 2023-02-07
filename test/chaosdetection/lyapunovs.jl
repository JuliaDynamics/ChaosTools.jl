using ChaosTools, Test

# Copy linear regression from FractalDimensions.jl
import Statistics
function _linreg(x::AbstractVector, y::AbstractVector)
    mx = Statistics.mean(x)
    my = Statistics.mean(y)
    b = Statistics.covm(x, mx, y, my)/Statistics.varm(x, mx)
    a = my - b*mx
    return a, b
end

# Creation of a trivial system with one coordinate going to infinity
# and the other to zero. Lyapunov exponents are known analytically
trivial_rule(x, p, n) = SVector{2}(p[1]*x[1], p[2]*x[2])
function trivial_rule_iip(dx, x, p, n)
    dx .= trivial_rule(x, p, n)
    return
end
trivial_jac(x, p, n) = SMatrix{2, 2}(p[1], 0, 0, p[2])
trivial_jac_iip(J, x, p, n) = (J[1,1] = p[1]; J[2,2] = p[2]; nothing)

u0 = ones(2)
p0_disc = [1.1, 0.8]
p0_cont = [0.1, -0.4]

# We do not have to test the autodiff version; that's done in DynamicalSystemsBase.jl!
@testset "analytic IDT=$(IDT), IIP=$(IIP)" for IDT in (true, false), IIP in (false, true)
    SystemType = IDT ? DeterministicIteratedMap : CoupledODEs
    rule = IIP ? trivial_rule_iip : trivial_rule
    p0 = IDT ? p0_disc : p0_cont
    lyapunovs = IDT ? log.(p0) : p0
    Jf = IIP ? trivial_jac_iip : trivial_jac

    ds = SystemType(rule, u0, p0)

    # Be careful here to not integrate too long because states become infinite
    λmax = lyapunov(ds, 100; Δt = 1, d0 = 1e-3)

    @test isapprox(λmax, lyapunovs[1]; atol = 0, rtol = 0.05)

    # tands = TangentDynamicalSystem(ds; J = Jf)

    # test_dynamical_system(tands, u0, p0; idt=IDT, iip=IIP, test_trajectory = false)
    # tangent_space_test(tands, lyapunovs)
end

@testset "Negative λ, continuous" begin
    f(u, p, t) = -0.9u
    ds = ContinuousDynamicalSystem(f, rand(SVector{3}), nothing)
    λ1 = lyapunov(ds, 1000)
    @test λ1 < 0
    @testset "Lorenz stable" begin
        ds = Systems.lorenz(ρ = 20.0)
        @test lyapunov(ds, 1000, Ttr = 100) ≈ 0 atol = 1e-3
    end
end

@testset "Negative λ, discrete" begin
    f(u, p, t) = 0.9u
    ds = DiscreteDynamicalSystem(f, rand(SVector{3}), nothing)
    λ1 = lyapunov(ds, 10000)
    @test isapprox(λ1, log(0.9); rtol = 1e-3)
end


@testset "Lyapunov from data" begin
    ds = Systems.henon()
    X, tvec = trajectory(ds, 50_000)
    x = X[:, 1] # some "recorded" timeseries
    ks = 1:20
    @testset "$meth, $di" for meth in [NeighborNumber(1), NeighborNumber(4), WithinRange(0.01)], di in [Euclidean(), FirstElement()]
        R = embed(x, 4, 1)
        E = lyapunov_from_data(R, ks,
        refstates = 1:1000, distance=di, ntype=meth)
        λ = _linreg(ks, E)[2]
        @test 0.3 < λ < 0.5
    end
end

#=

using ChaosTools
using ChaosTools.DelayEmbeddings
using Test, ChaosTools.StaticArrays
using DynamicalSystemsBase: CDS, DDS
using DynamicalSystemsBase.Systems: hoop, hoop_jac, hiip, hiip_jac
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac
using OrdinaryDiffEq
using Statistics

println("\nTesting lyapunov exponents...")
@testset "Lyapunov exponents" begin
u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0h = ones(2)
ph = [1.4, 0.3]

FUNCTIONS = [liip, liip_jac, loop, loop_jac, hiip, hiip_jac, hoop, hoop_jac]
INITCOD = [u0, u0h]
PARAMS = [p, ph]
MLE = [[0.75, 0.95], [0.41, 0.43]]
SLE = [[-0.1, 0.1], [-1.63, -1.61]]

@testset "Lyapunov spectrum" begin
for i in 1:8
    @testset "combination $i" begin
        sysindx = i < 5 ? 1 : 2
        if i < 5
            if isodd(i)
                ds = CDS(FUNCTIONS[i], INITCOD[sysindx], PARAMS[sysindx])
            else
                ds = CDS(FUNCTIONS[i-1], INITCOD[sysindx], PARAMS[sysindx], FUNCTIONS[i])
            end
        else
            if isodd(i)
                ds = DDS(FUNCTIONS[i], INITCOD[sysindx], PARAMS[sysindx])
            else
                ds = DDS(FUNCTIONS[i-1], INITCOD[sysindx], PARAMS[sysindx], FUNCTIONS[i])
            end
        end

        λ = lyapunovspectrum(ds, 2000)
        if i < 5
            λ2 = lyapunovspectrum(ds, 2000; Δt = 2.0, Ttr = 10.0,
            diffeq = (abstol = 1e-9, alg = Tsit5()))
        else
            λ2 = lyapunovspectrum(ds, 1000; Δt = 5, Ttr = 20)
        end

        @test MLE[sysindx][1] < λ[1] < MLE[sysindx][2]
        @test SLE[sysindx][1] < λ[2] < SLE[sysindx][2]
        @test MLE[sysindx][1] < λ2[1] < MLE[sysindx][2]
        @test SLE[sysindx][1] < λ2[2] < SLE[sysindx][2]

        if isodd(i) # Jacobians dont matter for MLE
            T = i < 5 ? 10000 : 1000
            λ = lyapunov(ds, T)
            @test MLE[sysindx][1] < λ < MLE[sysindx][2]
            if i < 5
                λ2 = lyapunov(ds, 2000; Δt = 1.0, Ttr = 10.0,
                diffeq = (abstol = 1e-9, alg = Tsit5()))
                @test MLE[sysindx][1] < λ2 < MLE[sysindx][2]
            end
        end
    end
end
end

@testset "1D Lyapunovs" begin
    ds = Systems.logistic(;r = 4.0)
    λ = lyapunov(ds, 10000; Ttr = 100)
    @test 0.692 < λ < 0.694
end

@testset "Negative λ, continuous" begin
    f(u, p, t) = -0.9u
    g(du, u, p, t) = (du .= -0.9u)

    ds = ContinuousDynamicalSystem(f, rand(SVector{3}), nothing)
    λ1 = lyapunov(ds, 10000)
    @test λ1 < 0
    ds = ContinuousDynamicalSystem(g, rand(3), nothing)
    λ2 = lyapunov(ds, 10000)
    @test λ2 < 0

    @testset "Lorenz stable" begin
        ds = Systems.lorenz(ρ = 20.0)
        @test lyapunov(ds, 2000, Ttr = 100) ≈ 0 atol = 1e-4
    end
end

@testset "Negative λ, discrete" begin
    f(u, p, t) = 0.9u
    g(du, u, p, t) = (du .= 0.9u)

    ds = DiscreteDynamicalSystem(f, rand(SVector{3}), nothing)
    λ1 = lyapunov(ds, 100000)
    @test isapprox(λ1, log(0.9); rtol = 1e-4)
    ds = DiscreteDynamicalSystem(g, rand(3), nothing)
    λ2 = lyapunov(ds, 100000)
    @test isapprox(λ1, log(0.9); rtol = 1e-4)
end

@testset "Lyapunov convergence" begin

ds = Systems.towel()
tinteg = tangent_integrator(ds)
ls, t = ChaosTools.lyapunovspectrum_convergence(tinteg, 20000, 1, 0)
l1 = [x[1] for x in ls]
@test 0.434 < l1[end] < 0.436
end


@testset "Lyapunov convergence, discrete" begin
ds = Systems.logistic(; r=4.0)
λs, t = ChaosTools.lyapunovspectrum_convergence(ds, 5000)
@test 0.692 < λs[end] < 0.694
end

@testset "Local growth rates" begin
    # input arguments
    ds = Systems.henon()
    points = trajectory(ds, 2000; Ttr = 100)
    λ = lyapunov(ds, 100000)

    λlocal = local_growth_rates(ds, points; Δt = 20, S = 20, e = 1e-12)

    @test size(λlocal) == (2001, 20)
    @test all(λlocal .< 1.0)
    mean_local = mean(λlocal[findall(!isinf, λlocal)])
    @test λ-0.1 ≤ mean_local ≤ λ+0.1
end



end
=#