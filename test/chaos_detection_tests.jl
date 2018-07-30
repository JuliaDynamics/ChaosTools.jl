using ChaosTools
using Test, LsqFit
using LsqFit: curve_fit
using OrdinaryDiffEq: Tsit5

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

println("\nTesting chaos detection algorithms...")

@testset "GALI discrete" begin

    @testset "Chaotic - towel map" begin
        ds = Systems.towel()
        model(x,p)= @. exp(-p[1]*x)
        ls = lyapunovs(ds, 10000)
        threshold = 1e-16
        @testset "k=$k" for k in [2,3]
            ex = sum(ls[1] - ls[j] for j in 2:k)
            g, t = gali(ds, 1000, k; threshold = threshold)
            fite = curve_fit(model, t, g, [ex]).param[1]
            @test g[end] < threshold
            @test t[end] < 1000
            @test isapprox(fite, ex, rtol=1)
        end
    end

    @testset "Standard Map" begin
        ds = Systems.standardmap()
        k = 2
        @testset "chaotic" begin
            g, t = gali(ds, 1000, k)
            @test g[end] ≤ 1e-12
            @test t[end] < 1000
        end
        @testset "regular" begin
            g, t = ChaosTools.gali(ds, 1000, k; u0 =  [π, rand()])
            @test t[end] == 1000
            @test g[end] > 1/1000^2
        end
    end

    @testset "Coupled Standard Maps" begin
        M = 3; ks = 3ones(M); Γ = 0.1;
        stable = [π, π, π, 0.01, 0, 0] .+ 0.1
        chaotic = rand(2M)

        ds = Systems.coupledstandardmaps(M, stable; ks=ks, Γ = Γ)

        @testset "regular k=$k" for k in [2,3,4, 5, 6]
            g, t = gali(ds, 1000, k; threshold=1e-12)
            @test t[end] == 1000
            @test g[end] > 1e-12
        end

        @testset "chaotic k=$k" for k in [2,3,4, 5, 6]
            g, t = gali(ds, 1000, k; threshold=1e-12, u0 = chaotic)
            @test t[end] < 1000
            @test g[end] ≤ 1e-12
        end

    end
end


@testset "GALI continuous" begin
    @testset "Henon-Helies" begin
        sp = [0, .295456, .407308431, 0] #stable periodic orbit: 1D torus
        qp = [0, .483000, .278980390, 0] #quasiperiodic orbit: 2D torus
        ch = [0, -0.25, 0.42081, 0] # chaotic orbit
        tt = 1000
        ds = Systems.henonheiles()
        diffeq = Dict(:abstol=>1e-9, :reltol=>1e-9, :solver => Tsit5())
        @testset "regular" begin
            for k in [2,3,4]
                g, t = gali(ds, tt, k; diff_eq_kwargs = diffeq, u0 = sp)
                @test t[end] ≥ tt
            end
            for k in [2,3,4]
                g, t = gali(ds, tt, k; diff_eq_kwargs = diffeq, u0 = qp)
                @test t[end] ≥ tt
            end
        end
        @testset "chaotic" begin
            for k in [2,3,4]
                g, t = gali(ds, tt, k; diff_eq_kwargs = diffeq, u0 = ch)
                @test t[end] < tt
                @test g[end] ≤ 1e-12
            end
        end
    end

    # Test henon helies out of place & autodiffed
    function hhoop(u, p, t)
        du1 = u[3]
        du2 = u[4]
        du3 = -u[1] - 2u[1]*u[2]
        du4 = -u[2] - (u[1]^2 - u[2]^2)
        return SVector{4}(du1, du2, du3, du4)
    end
    @testset "Henon-Helies OOP" begin
        sp = [0, .295456, .407308431, 0] #stable periodic orbit: 1D torus
        qp = [0, .483000, .278980390, 0] #quasiperiodic orbit: 2D torus
        ch = [0, -0.25, 0.42081, 0] # chaotic orbit
        tt = 1000
        ds = Systems.ContinuousDynamicalSystem(hhoop, sp, nothing)
        diffeq = Dict(:abstol=>1e-9, :reltol=>1e-9, :solver => Tsit5())
        @testset "regular" begin
            for k in [2,3,4]
                g, t = gali(ds, tt, k; diff_eq_kwargs = diffeq, u0 = sp)
                @test t[end] ≥ tt
            end
            for k in [2,3,4]
                g, t = gali(ds, tt, k; diff_eq_kwargs = diffeq, u0 = qp)
                @test t[end] ≥ tt
            end
        end
        @testset "chaotic" begin
            for k in [2,3,4]
                g, t = gali(ds, tt, k; diff_eq_kwargs = diffeq, u0 = ch)
                @test t[end] < tt
                @test g[end] ≤ 1e-12
            end
        end
    end

end
