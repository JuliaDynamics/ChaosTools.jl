using ChaosTools, DynamicalSystemsBase
using Test
using ChaosTools.Distances: Cityblock, Euclidean

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

println("\nTesting nonlinear timeseries analysis...")
@testset "Henon embed" begin
    ds = Systems.henon()
    data = trajectory(ds, 100000)
    x = data[:, 1] # some "recorded" timeseries

    @testset "Dimension" begin
        τ = 1; D = 2
        R = embed(x, D, τ)
        D2 = generalized_dim(R)
        test_value(D2, 1.1, 1.3)
    end

    @testset "Numerical Lyapunov" begin
        ks = 1:20
        @testset "meth = $meth" for meth in
            [NeighborNumber(1), NeighborNumber(4), WithinRange(0.01)]
            @testset "distance = $di" for di in [Euclidean(), Cityblock()]
                for D in [2, 4]
                    R = embed(x, D, 1)
                    E = numericallyapunov(R, ks,
                    refstates = 1:1000, distance=di, ntype=meth)
                    λ = linear_region(ks, E)[2]
                    test_value(λ, 0.3, 0.5)
                end
            end
        end
    end
end

@testset "Broomhead-King" begin
    using Random
    Random.seed!(1234);
    ds = Systems.gissinger(ones(3)) # standard initial condition

    data = trajectory(ds, 1000.0, dt = 0.05)
    x = data[1:end-1, 1]
    s = x .+ 0.01rand(length(x))

    Ux, Σx = broomhead_king(x, 40)
    Us, Σs = broomhead_king(s, 40)
    R = embed(x, 2, 30)

    for j in 10:40
        @test Σx[j] < Σs[j]
    end

    newcoords = Dataset(Us[:, 1], Us[:, 2])
    newcoordsclean = Dataset(Ux[:, 1], Ux[:, 2])

    Dnew = generalized_dim(newcoords)
    DR = generalized_dim(R)
    DC = generalized_dim(newcoordsclean)
    @test abs(Dnew - 0.1 - DR) < 0.2 # subtract 0.1 for "added dimensionality"
    @test abs(DC - DR) < 0.2
end
