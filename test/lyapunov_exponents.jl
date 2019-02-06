using ChaosTools
using Test, StaticArrays
using DynamicalSystemsBase: CDS, DDS
using DynamicalSystemsBase.Systems: hoop, hoop_jac, hiip, hiip_jac
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac
using OrdinaryDiffEq

println("\nTesting lyapunov exponents...")
let
u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0h = ones(2)
ph = [1.4, 0.3]

FUNCTIONS = [liip, liip_jac, loop, loop_jac, hiip, hiip_jac, hoop, hoop_jac]
INITCOD = [u0, u0h]
PARAMS = [p, ph]
MLE = [[0.75, 0.95], [0.41, 0.43]]
SLE = [[-0.1, 0.1], [-1.63, -1.61]]

@testset "Lyapunovs" begin
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

        λ = lyapunovs(ds, 2000)
        if i < 5
            λ2 = lyapunovs(ds, 2000; dt = 2.0, Ttr = 10.0,
            abstol = 1e-9, alg = Tsit5())
        else
            λ2 = lyapunovs(ds, 1000; dt = 5, Ttr = 20)
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
                λ2 = lyapunov(ds, 2000; dt = 1.0, Ttr = 10.0,
                abstol = 1e-9, alg = Tsit5())
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
ls, t = ChaosTools.lyapunovs_convergence(tinteg, 20000, 1, 0)
l1 = [x[1] for x in ls]
@test 0.434 < l1[end] < 0.436
end

end
