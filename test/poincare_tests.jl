using ChaosTools
using Test, StaticArrays, OrdinaryDiffEq, LinearAlgebra

println("\nTesting poincare sections...")

@testset "Poincare SOS" begin
    @testset "Henon Heiles" begin
        ds = Systems.henonheiles([0, .483000, .278980390, 0] )
        psos = poincaresos(ds, (2, 0.0), 1000.0, direction = +1)
        xcross = psos[:, 2]
        @test length(xcross) > 1
        for x in xcross
            @test abs(x) < 1e-3
        end

        # @inline Vhh(q1, q2) = 1//2 * (q1^2 + q2^2 + 2q1^2 * q2 - 2//3 * q2^3)
        # @inline Thh(p1, p2) = 1//2 * (p1^2 + p2^2)
        # @inline Hhh(q1, q2, p1, p2) = Thh(p1, p2) + Vhh(q1, q2)
        # @inline Hhh(u::AbstractVector) = Hhh(u...)
        #
        # E = [Hhh(p) for p in psos]
        #
        # @test std(E) < 1e-10
        # @test maximum(@. E - E[1]) < 1e-10
    end
    @testset "Gissinger crazy plane" begin
        gis = Systems.gissinger([2.32865, 2.02514, 1.98312])
        # Define appropriate hyperplane for gissinger system
        ν = 0.1
        Γ = 0.9 # default parameters of the system

        # I want hyperperplane defined by these two points:
        Np(μ) = SVector{3}(sqrt(ν + Γ*sqrt(ν/μ)), -sqrt(μ + Γ*sqrt(μ/ν)), -sqrt(μ*ν))
        Nm(μ) = SVector{3}(-sqrt(ν + Γ*sqrt(ν/μ)), sqrt(μ + Γ*sqrt(μ/ν)), -sqrt(μ*ν))

        # Create hyperplane using normal vector to vector connecting points:
        gis_plane(μ) = [cross(Np(μ), Nm(μ))..., 0]

        μ = 0.12
        set_parameter!(gis, 1, μ)
        psos = poincaresos(gis, gis_plane(μ), 5000.0, Ttr = 200.0, direction = +1)
        @test length(psos) > 1
        @test generalized_dim(2, psos) < 1
    end
    @testset "beginning on the plane" begin

        @inbounds @inline function ż(z, p, t)
            A, B, D = p
            p₀, p₂ = z[1:2]
            q₀, q₂ = z[3:4]

            return SVector{4}(
                -A * q₀ - 3 * B / √2 * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2),
                -q₂ * (A + 3 * √2 * B * q₀ + D * (q₀^2 + q₂^2)),
                A * p₀,
                A * p₂
            )
        end
        z0 = SVector{4}(2.3499921651423565, -9.801351029039825, 0.0, -2.7230316965872268)

        ds = ContinuousDynamicalSystem(ż, z0, (A=1, B=0.55, D=0.4))
        psos = poincaresos(ds, (3, 0), 10., direction=+1, idxs=[2,4])
        @test length(psos) > 0
        @test size(psos)[2] == 2
    end
end

@testset "Produce OD" begin
  @testset "Shinriki" begin
    ds = Systems.shinriki([-2, 0, 0.2])

    pvalues = range(19,stop=22,length=11)
    parameter = 1
    i = 1
    j = 2
    tf = 200.0

    de = (abstol=1e-9, reltol = 1e-9)
    output = produce_orbitdiagram(ds, (j, 0.0), i, parameter, pvalues; tfinal = tf,
    Ttr = 100.0, printparams = false, direction = +1, de...)
    @test length(output) == length(pvalues)

    v = round.(output[1], digits = 4)
    s = collect(Set(v))
    @test length(s) == 1
    @test s[1] == -0.856

    v = round.(output[4], digits = 4)
    s = Set(v)
    @test length(s) == 2
    @test s == Set([-0.376, -1.2887])

  end
end
