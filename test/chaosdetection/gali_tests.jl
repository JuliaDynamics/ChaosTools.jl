using ChaosTools
using Test
using OrdinaryDiffEq: Vern9

@testset "standard map" begin
    @inbounds function standardmap_rule(x, par, n)
        theta = x[1]; p = x[2]
        p += par[1]*sin(theta)
        theta += p
        while theta >= 2π; theta -= 2π; end
        while theta < 0; theta += 2π; end
        while p >= 2π; p -= 2π; end
        while p < 0; p += 2π; end
        return SVector(theta, p)
    end
    ds = DeterministicIteratedMap(standardmap_rule, [0.001, 0.008], [1.0])
    k = 2
    @testset "chaotic" begin
        g, t = gali(ds, 1000, k)
        @test g[end] ≤ 1e-12
        @test t[end] < 1000
    end
    @testset "regular" begin
        g, t = ChaosTools.gali(ds, 1000, k; u0 =  [π, rand()])
        @test t[end] == 1000
        @test g[end] > 1/1000
    end
end

@testset "Henon-Helies" begin
    @inbounds function henonheiles_rule(du, u, p, t)
        du[1] = u[3]
        du[2] = u[4]
        du[3] = -u[1] - 2u[1]*u[2]
        du[4] = -u[2] - (u[1]^2 - u[2]^2)
        return nothing
    end
    @inbounds function henonheiles_jacob(J, u, p, t)
        o = 0.0; i = 1.0
        J[1,:] .= (o,    o,     i,    o)
        J[2,:] .= (o,    o,     o,    i)
        J[3,:] .= (-i - 2*u[2],   -2*u[1],   o,   o)
        J[4,:] .= (-2*u[1],  -1 + 2*u[2],  o,   o)
        return nothing
    end
    sp = [0, .295456, .407308431, 0] #stable periodic orbit: 1D torus
    qp = [0, .483000, .278980390, 0] #quasiperiodic orbit: 2D torus
    ch = [0, -0.25, 0.42081, 0] # chaotic orbit
    tt = 1000
    diffeq = Dict(:abstol=>1e-9, :reltol=>1e-9, :solver => Vern9())
    ds = CoupledODEs(henonheiles_rule, sp, nothing; diffeq)

    @testset "k = $k" for k in [2,4]
        tands = TangentDynamicalSystem(ds; k, J = henonheiles_jacob)
        @testset "regular" begin
            reinit!(tands, sp)
            g, t = gali(tands, tt)
            @test t[end] ≥ tt
            if k == 2
                @test g[end] > 0.01
            else
                @test g[end] > 1e-9
            end
        end
        @testset "quasiperiodic" begin
            reinit!(tands, qp)
            g, t = gali(tands, tt)
            @test t[end] ≥ tt
            if k == 2
                @test g[end] > 0.01
            else
                @test g[end] > 1e-9
            end
        end
        @testset "chaotic" begin
            reinit!(tands, ch)
            g, t = gali(tands, tt)
            @test t[end] < tt
            @test g[end] ≤ 1e-12
        end
    end
end