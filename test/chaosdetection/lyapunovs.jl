using ChaosTools, Test
using DelayEmbeddings: embed
import Statistics

# Creation of a trivial system with one coordinate going to infinity
# and the other to zero. Lyapunov exponents are known analytically
trivial_rule(x, p, n) = SVector{2}(p[1] * x[1], p[2] * x[2])
function trivial_rule_iip(dx, x, p, n)
    dx .= trivial_rule(x, p, n)
    return
end
trivial_jac(x, p, n) = SMatrix{2,2}(p[1], 0, 0, p[2])
trivial_jac_iip(J, x, p, n) = (J[1, 1] = p[1]; J[2, 2] = p[2]; nothing)

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
    λmax = lyapunov(ds, 100; Δt=1, d0=1e-3)

    @test isapprox(λmax, lyapunovs[1]; atol=0, rtol=0.05)

    @testset "tangent IAD=$(IAD)" for IAD in (true, false)
        if IAD
            Jf = IIP ? trivial_jac_iip : trivial_jac
            tands = TangentDynamicalSystem(ds; J=Jf)
            spec = lyapunovspectrum(tands, 100; Δt=1)
            spec1 = lyapunovspectrum(TangentDynamicalSystem(ds; k=1, J=Jf), 100; Δt=1)
        else
            spec = lyapunovspectrum(ds, 100; Δt=1)
            spec1 = lyapunovspectrum(ds, 100, 1; Δt=1)
        end

        @test isapprox(spec[1], lyapunovs[1]; atol=0, rtol=1e-3)
        @test isapprox(spec1[1], lyapunovs[1]; atol=0, rtol=1e-3)
        @test isapprox(spec[2], lyapunovs[2]; atol=0, rtol=1e-3)
    end

end

@testset "Negative λ, continuous" begin
    f(u, p, t) = -0.9u
    ds = ContinuousDynamicalSystem(f, rand(SVector{3}), nothing)
    λ1 = lyapunov(ds, 10000)
    @test λ1 ≈ -0.9 atol = 1e-2 rtol = 1e-2

    @testset "Lorenz stable" begin
        function lorenz_rule(u, p, t)
            σ = p[1]
            ρ = p[2]
            β = p[3]
            du1 = σ * (u[2] - u[1])
            du2 = u[1] * (ρ - u[3]) - u[2]
            du3 = u[1] * u[2] - β * u[3]
            return SVector{3}(du1, du2, du3)
        end

        ds = CoupledODEs(lorenz_rule, fill(10.0, 3), [10, 20, 8 / 3])
        @test lyapunov(ds, 2000, Ttr=100) ≈ 0 atol = 1e-3
    end
end

@testset "Negative λ, discrete" begin
    f(u, p, t) = 0.9u
    ds = DiscreteDynamicalSystem(f, rand(SVector{3}), nothing)
    λ1 = lyapunov(ds, 10000)
    @test isapprox(λ1, log(0.9); rtol=1e-3)
end

henon_rule(x, p, n) = SVector{2}(1.0 - p[1] * x[1]^2 + x[2], p[2] * x[1])
henon() = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])

@testset "Lyapunov from data" begin
    ds = henon()
    X, tvec = trajectory(ds, 50_000)
    x = X[:, 1] # some "recorded" timeseries
    ks = 1:20
    @testset "$meth, $di" for meth in [NeighborNumber(1), NeighborNumber(4), WithinRange(0.01)], di in [Euclidean(), FirstElement()]
        R = embed(x, 4, 1)
        E = lyapunov_from_data(R, ks,
            refstates=1:1000, distance=di, ntype=meth)
        λ = ChaosTools.linreg(ks, E)[2]
        @test 0.3 < λ < 0.5
    end
end

@testset "Local growth rates" begin
    # input arguments
    ds = henon()
    points, tvec = trajectory(ds, 2000; Ttr=100)
    λ = lyapunov(ds, 10_000)
    using Random: seed!
    seed!(1234151)

    λlocal = local_growth_rates(ds, points; Δt=20, S=20, e=1e-12)

    @test size(λlocal) == (2001, 20)
    @test all(λlocal .< 1.0)
    mean_local = Statistics.mean(λlocal)
    @test λ - 0.1 ≤ mean_local ≤ λ + 0.1
end

@testset "testSuccessfulStep" begin
    u0 = [NaN, 0]
    p0 = [0.1, -0.4]
    ds = CoupledODEs(trivial_rule, u0, p0)
    @test isnan(lyapunov(ds, 10; Ttr=0, Δt=0.5))
end

# --- Covariant Lyapunov Vectors (CLV) tests ---

@testset "CLV diagonal system" begin
    # For diagonal systems, CLVs should be coordinate axis vectors
    @testset "IDT=$(IDT), IIP=$(IIP)" for IDT in (true, false), IIP in (false, true)
        SystemType = IDT ? DeterministicIteratedMap : CoupledODEs
        rule = IIP ? trivial_rule_iip : trivial_rule
        p0 = IDT ? p0_disc : p0_cont

        ds = SystemType(rule, u0, p0)
        result = clv(ds, 20; Ttr=50, Ttr_bkw=50)

        # Check that final CLVs are close to coordinate axes
        V_final = result.V[end]
        for j in 1:2
            v = V_final[:, j]
            # CLV should be approximately a unit vector along one axis
            @test maximum(abs.(v)) ≈ 1.0 atol = 1e-4
            # Only one component should be significant
            @test count(x -> abs(x) > 0.5, v) == 1
        end
    end
end

@testset "CLV Lyapunov consistency" begin
    # CLV-computed Lyapunov exponents should match lyapunovspectrum
    @testset "Hénon map" begin
        ds = henon()
        N = 100
        result = clv(ds, N; Ttr=200, Ttr_bkw=100)
        λ_spec = lyapunovspectrum(ds, N + 100; Ttr=200)

        @test result.λ[1] ≈ λ_spec[1] rtol = 0.05
        @test result.λ[2] ≈ λ_spec[2] rtol = 0.05
    end

    @testset "Lorenz system" begin
        function lorenz_rule(u, p, t)
            σ, ρ, β = p
            return SVector{3}(
                σ * (u[2] - u[1]),
                u[1] * (ρ - u[3]) - u[2],
                u[1] * u[2] - β * u[3]
            )
        end
        ds = CoupledODEs(lorenz_rule, [1.0, 0.0, 0.0], [10.0, 28.0, 8 / 3])

        N = 50
        result = clv(ds, N; Ttr=20, Ttr_bkw=50, Δt=0.5)
        λ_spec = lyapunovspectrum(ds, N + 50; Ttr=20, Δt=0.5)

        @test result.λ[1] ≈ λ_spec[1] rtol = 0.1
        @test result.λ[2] ≈ λ_spec[2] atol = 0.1  # Near-zero exponent
        @test result.λ[3] ≈ λ_spec[3] rtol = 0.1
    end
end

@testset "CLV covariance property" begin
    import LinearAlgebra: dot, norm

    @testset "Hénon map covariance" begin
        ds = henon()
        N = 50
        result = clv(ds, N; Ttr=500, Ttr_bkw=200)

        # Get the Jacobian function (note: SMatrix uses column-major order)
        # Hénon Jacobian: [[-2ax, 1], [b, 0]], but SMatrix fills column-by-column
        henon_jac(x, p, n) = SMatrix{2,2}(-2 * p[1] * x[1], p[2], 1.0, 0.0)

        # Use the states returned by clv()
        p = current_parameters(ds)

        # Check covariance for the first CLV (most expanding direction)
        # The second CLV requires much longer transients to converge
        num_tests_pass = 0
        for i in 1:(N-1)
            J = henon_jac(result.x[i], p, 0)
            v_evolved = J * result.V[i][:, 1]
            v_next = result.V[i+1][:, 1]
            # Check they are parallel: |cos(angle)| ≈ 1
            cosθ = abs(dot(v_evolved, v_next) / (norm(v_evolved) * norm(v_next)))
            if cosθ > 0.999
                num_tests_pass += 1
            end
        end
        # All covariance checks should pass with high precision
        @test num_tests_pass == N - 1
    end
end

@testset "CLV neutral direction (flow alignment)" begin
    # This test is very important for ensuring that the CLVs are being computed correctly.
    # If this test fails, you're probably getting bogus CLVs.
    import LinearAlgebra: dot, norm

    # For continuous-time systems, the neutral CLV (λ ≈ 0) should align with the flow direction
    @testset "Lorenz neutral CLV aligns with flow" begin
        function lorenz_rule(u, p, t)
            σ, ρ, β = p
            return SVector{3}(σ * (u[2] - u[1]), u[1] * (ρ - u[3]) - u[2], u[1] * u[2] - β * u[3])
        end

        ds = CoupledODEs(lorenz_rule, [1.0, 0.0, 0.0], [10.0, 28.0, 8 / 3])
        p = current_parameters(ds)

        # Compute CLVs with sufficient transients for convergence
        result = clv(ds, 100; Ttr=1_000, Ttr_bkw=1_000, Δt=0.1)

        # Check that the neutral CLV (second one) aligns with the flow vector
        num_aligned = 0
        for i in 1:length(result.x)
            x = result.x[i]
            # Flow direction = f(x) / |f(x)|
            flow = lorenz_rule(x, p, 0.0)
            flow_normalized = flow / norm(flow)
            # Neutral CLV is the second one (λ ≈ 0)
            v_neutral = result.V[i][:, 2]
            v_neutral_normalized = v_neutral / norm(v_neutral)
            # Check alignment: |cos(angle)| ≈ 1
            cosθ = abs(dot(flow_normalized, v_neutral_normalized))
            if cosθ > 0.99
                num_aligned += 1
            end
        end
        # At least 95% should be well-aligned
        @test num_aligned >= 0.95 * length(result.x)
    end
end