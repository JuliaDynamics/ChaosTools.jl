using ChaosTools, Test

@testset "Henon map" begin
    henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
    henon_jacob(x, p, n) = SMatrix{2,2}(-2*p[1]*x[1], p[2], 1.0, 0.0)
    ds = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
    x = interval(-1.5, 1.5)
    y = interval(-0.5, 0.5)
    box = x × y

    # henon fixed points analytically
    hmfp(a, b) = (x = -(1-b)/2a - sqrt((1-b)^2 + 4a)/2a; return SVector(x, b*x))
    henon_fp = hmfp(current_parameters(ds)...)

    @testset "J=$(J)" for J in (nothing, henon_jacob)
        fp, eigs, stable = fixedpoints(ds, box, J)
        @test length(fp) == 2
        @test dimension(fp) == 2
        @test stable == [false, false]
        @test (henon_fp ≈ fp[1]) || (henon_fp ≈ fp[2])
    end
end

# TODO: This will be used once higher order FP are allowed
# @testset "standard map" begin
#     ds = Systems.standardmap()
#     x = interval(0.0, (2π - 100eps()))
#     box = x × x

#     fp, eigs, stable = fixedpoints(ds, box)

#     for (i, e) in enumerate(fp)
#         if isapprox(e, SVector(0,0); atol = 1e-8)
#             @test !stable[i]
#         elseif e ≈ SVector(π, 0)
#             @test stable[i]
#         else
#             @test false
#         end
#     end
# end
@testset "Henon map (n-th order)" begin
    henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
    henon_jacob(x, p, n) = SMatrix{2,2}(-2*p[1]*x[1], p[2], 1.0, 0.0)
    ds = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
    x = interval(-3.0, 3.0)
    y = interval(-10.0, 10.0)
    box = x × y
    o = 4
    fp, eigs, stable = fixedpoints(ds, box, henon_jacob; order=o)
    tol = 1e-8
    for x0 in fp
        set_state!(ds, x0)
        step!(ds, o)
        xn = current_state(ds)
        @test isapprox(x0, xn; atol = tol)
    end
end

@testset "Lorenz system" begin
    function lorenz_rule(u, p, t)
        @inbounds begin
            σ = p[1]; ρ = p[2]; β = p[3]
            du1 = σ*(u[2]-u[1])
            du2 = u[1]*(ρ-u[3]) - u[2]
            du3 = u[1]*u[2] - β*u[3]
            return SVector{3}(du1, du2, du3)
        end
    end
    function lorenz_jacob(u, p, t)
        @inbounds begin
            σ, ρ, β = p
            return SMatrix{3,3}(-σ, ρ - u[3], u[2], σ, -1, u[1], 0, -u[1], -β)
        end
    end

    x = -20..20
    y = -20..20
    z = 0.0..40
    box = x × y × z
    σ = 10.0
    β = 8/3
    lorenzfp(ρ, β) = [
        SVector(0,0,0.0),
        SVector(sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1),
        SVector(-sqrt(β*(ρ-1)), -sqrt(β*(ρ-1)), ρ-1),
    ]

    stabil(σ, β) = σ*(σ+β+3)/(σ-β-1)

    ds = CoupledODEs(lorenz_rule, zeros(3), [10, 22, 8/3])
    afps = lorenzfp(22,8/3)

    @testset "J=$(J)" for J in (nothing, lorenz_jacob)
        fp, eigs, stable = fixedpoints(ds, box, J)
        @test length(fp) == 3
        for p in fp
            @test any(w -> isapprox(w, p; atol = 1e-8), afps)
        end
        @test count(stable) == 2
    end
end
