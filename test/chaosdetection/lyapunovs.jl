using ChaosTools, Test
import Statistics

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

    ds = SystemType(rule, u0, p0)

    # Be careful here to not integrate too long because states become infinite
    λmax = lyapunov(ds, 100; Δt = 1, d0 = 1e-3)

    @test isapprox(λmax, lyapunovs[1]; atol = 0, rtol = 0.05)

    @testset "tangent IAD=$(IAD)" for IAD in (true, false)
        if IAD
            Jf = IIP ? trivial_jac_iip : trivial_jac
            tands = TangentDynamicalSystem(ds; J = Jf)
            spec = lyapunovspectrum(tands, 100; Δt = 1)
        else
            spec = lyapunovspectrum(ds, 100; Δt = 1)
        end

        @test isapprox(spec[1], lyapunovs[1]; atol = 0, rtol = 1e-3)
        @test isapprox(spec[2], lyapunovs[2]; atol = 0, rtol = 1e-3)
    end

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
        λ = ChaosTools.linreg(ks, E)[2]
        @test 0.3 < λ < 0.5
    end
end

@testset "Local growth rates" begin
    # input arguments
    ds = Systems.henon()
    points, tvec = trajectory(ds, 2000; Ttr = 100)
    λ = lyapunov(ds, 100000)

    λlocal = local_growth_rates(ds, points; Δt = 20, S = 20, e = 1e-12)

    @test size(λlocal) == (2001, 20)
    @test all(λlocal .< 1.0)
    mean_local = Statistics.mean(λlocal)
    @test λ-0.1 ≤ mean_local ≤ λ+0.1
end
