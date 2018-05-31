using ChaosTools
using Base.Test
using Distances: Cityblock, Euclidean
using Distributions: Normal

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

println("\nTesting nonlinear timeseries analysis...")
@testset "Henon Reconstruction" begin
    ds = Systems.henon()
    data = trajectory(ds, 100000)
    x = data[:, 1] # some "recorded" timeseries
    @testset "Sizes" begin
        for τ in [1, 2, 7]
            for D in [2, 3, 6]
                R = Reconstruction(x, D, τ)
                @test length(size(R)) == 2
                @test length(R) == length(x) - (D-1)*τ
                @test length(R[1]) == D
            end
        end
    end
    @testset "Dimension" begin
        τ = 1; D = 2
        R = Reconstruction(x, D, τ)
        D2 = information_dim(R)
        test_value(D2, 1.1, 1.3)
    end
    ks = 1:20
    @testset "Numerical Lyapunov" begin
        @testset "meth = $meth" for meth in
            [FixedMassNeighborhood(1), FixedMassNeighborhood(4),
            FixedSizeNeighborhood(0.01)]
            @testset "distance = $di" for di in [Euclidean(), Cityblock()]
                for D in [2, 4]
                    R = Reconstruction(x, D, 1)
                    E = numericallyapunov(R, ks,
                    refstates = 1:1000, distance=di, ntype=meth)
                    λ = linear_region(ks, E)[2]
                    test_value(λ, 0.3, 0.5)
                end
            end
        end
    end
end
@testset "Multidim Multitime R" begin
    ds = Systems.towel()
    data = trajectory(ds, 10000)

    taus = [0 0; 2 3; 4 6; 6 8]
    data2 = data[:, 1:2]
    R = Reconstruction(data2, 4, taus)

    ks = 1:20
    E = numericallyapunov(R, ks,
    refstates = 1:1000, distance=Cityblock(), ntype=FixedMassNeighborhood(1))
    λ = linear_region(ks, E)[2]
    test_value(λ, 0.3, 0.6)
end

@testset "Broomhead-King" begin
    ds = Systems.gissinger()
    data = trajectory(ds, 1000.0, dt = 0.05)
    x = data[1:end-1, 1] # "exactly" 20000 points
    distrib = Normal(0, 0.1)
    s = x .+ rand(distrib, length(x))

    Ux, Σx = broomhead_king(x, 40)
    Us, Σs = broomhead_king(s, 40)
    R = Reconstruction(x, 2, 30)
    newcoords = Dataset(Us[:, 1], Us[:, 2])
    newcoordsclean = Dataset(Ux[:, 1], Ux[:, 2])

    for j in 10:40
        @test Σx[j] < Σs[j]
    end

    Dnew = information_dim(newcoords)
    DR = information_dim(R)
    DC = information_dim(newcoordsclean)
    @test abs(Dnew - 0.1 - DR) < 0.15 # subtract 0.1 for "added dimensionality"
    @test abs(DC - DR) < 0.2
end
