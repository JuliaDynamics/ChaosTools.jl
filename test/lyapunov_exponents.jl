using ChaosTools
using Base.Test, StaticArrays
using DynamicalSystemsBase: CDS, DDS
using DynamicalSystemsBase.Systems: hoop, hoop_jac, hiip, hiip_jac
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac
using OrdinaryDiffEq

println("\nTesting lyapunov exponents...")

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
            diff_eq_kwargs = Dict(:abstol=>1e-9, :solver => Tsit5()))
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
            if i < 5
                λ2 = lyapunov(ds, T; dt = 2.0, Ttr = 10.0,
                diff_eq_kwargs = Dict(:abstol=>1e-9, :solver => Tsit5()))
            else
                λ2 = lyapunovs(ds, T; dt = 5, Ttr = 20)
            end
            @test MLE[sysindx][1] < λ < MLE[sysindx][2]
            @test MLE[sysindx][1] < λ2[1] < MLE[sysindx][2]
        end
    end
end
end

@testset "1D Lyapunovs" begin
    ds = Systems.logistic(;r = 4.0)
    λ = lyapunov(ds, 10000; Ttr = 100)
    @test 0.692 < λ < 0.694
end
