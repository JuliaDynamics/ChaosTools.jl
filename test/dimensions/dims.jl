using ChaosTools
using Test
using StatsBase
using Statistics

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

println("\nTesting generalized entropy (genentropy) & linear scaling...")
@testset "Generalized Dimensions" begin
    @testset "Henon Map" begin
        ds = Systems.henon()
        ts = trajectory(ds, 200000)
        mat = Matrix(ts)
        # Test call with dataset
        genentropy(1, 0.001, ts)
        es = 10 .^ range(-0, stop = -3, length = 7)
        dd = zero(es)
        for q in [0,2,1, 2.56]
            for (i, ee) in enumerate(es)
                dd[i] = genentropy(mat, ee; α = q)
            end
            linr, dim = linear_region(-log.(es), dd)
            test_value(dim, 1.1, 1.3)
        end
    end
    @testset "Lorenz System" begin
        ds = Systems.lorenz()
        ts = trajectory(ds, 5000)
        es = 10 .^ range(1, stop = -3, length = 11)
        dd = zero(es)
        for q in [0,1,2]
            for (i, ee) in enumerate(es)
                dd[i] = genentropy(ts, ee; α=q)
            end
            linr, dim = linear_region(-log.(es), dd)
            if q == 0
                test_value(dim, 1.7, 1.9)
            else
                test_value(dim, 1.85, 2.0)
            end
        end
    end
end

println("\nTesting generalized entropy using Molteno's boxing method...")
@testset "Molteno's boxing method" begin
    @testset "Henon Map" begin
        ds = Systems.henon()
        ts = trajectory(ds, 200000)
        boxes, ϵs = molteno_boxing(ts)
        for q in [0,2,1, 2.56]
            dd = genentropy.(q, boxes)
            linr, dim = linear_region(-log.(ϵs), dd)
            test_value(dim, 1.1, 1.3)
        end
    end
    @testset "Lorenz System" begin
        ds = Systems.lorenz()
        ts = trajectory(ds, 5000)
        boxes, ϵs = molteno_boxing(ts)
        for q in [0,2,1, 2.56]
            dd = genentropy.(q, boxes)
            linr, dim = linear_region(-log.(ϵs), dd)
            test_value(dim, 1.85, 2.2)
        end
    end
end

println("\nTesting dimension calls (all names)...")
@testset "Dimension calls" begin
    ds = Systems.henon()
    ts = trajectory(ds, 50000)
    test_value(generalized_dim(ts; α = 1.32), 1.1, 1.3)
    test_value(generalized_dim(ts, base = 2, α = 1.32), 1.1, 1.3)
end
