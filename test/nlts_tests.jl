using ChaosTools, DynamicalSystemsBase
using Test
using ChaosTools.Distances: Cityblock, Euclidean

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

println("\nTesting nonlinear timeseries analysis...")

@testset "Broomhead-King" begin
    using Random
    Random.seed!(1234);
    ds = Systems.gissinger(ones(3)) # standard initial condition

    data = trajectory(ds, 1000.0; Δt = 0.05)
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
