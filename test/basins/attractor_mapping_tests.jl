using ChaosTools
using DynamicalSystemsBase
using DynamicalSystemsBase.DelayEmbeddings
using Test
using LinearAlgebra
using OrdinaryDiffEq
using Random
using Statistics

@testset "AttractorMappers" begin

@testset "Henon map" begin
    u1 = [0.0, 0.0] # converges to attractor
    u2 = [0, 2.0]   # diverges to inf
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.0, 2.0; length=100)
    grid = (xg, yg)
    sampler, = statespace_sampler(Random.MersenneTwister(1234); 
    min_bounds = minimum.(grid), max_bounds = maximum.(grid))
    ics = Dataset([sampler() for i in 1:1000])

    function henon_fractions_test(mapper; k = [1, -1])
        fs = basin_fractions(mapper, sampler)
        @test 0.1 < fs[k[1]] < 0.9
        @test 0.1 < fs[k[2]] < 0.9
        @test sum(values(fs)) == 1

        # Deterministic test, should be tested with exact accuracy
        fs, labels = basin_fractions(mapper, ics)
        @test eltype(labels) == Int
        @test fs[k[1]] == 0.451
        @test fs[k[2]] == 0.549
        @test sum(values(fs)) == 1
    end


    @testset "Recurrences" begin
        mapper = AttractorsViaRecurrences(ds, grid)
        @test 1  == mapper(u1)
        @test -1 == mapper(u2)
        henon_fractions_test(mapper)
    end

    @testset "Proximity" begin
        A = trajectory(ds, 1000, u1; Ttr = 100)
        attractors = Dict(1 => A)
        mapper = AttractorsViaProximity(ds, attractors; Ttr = 100)
        @test 1  == mapper(u1)
        @test -1 == mapper(u2)
        henon_fractions_test(mapper)
    end

    @testset "Featurizing" begin
        # Featurizing is hard in the Henon map
        function henon_featurizer(A, t)
            x = [mean(A[:, 1]), mean(A[:, 2])]
            return any(isinf, x) ? [200.0, 200.0] : x
        end
        @testset "unsupervised" begin
        mapper = AttractorsViaFeaturizing(ds, henon_featurizer; Ttr = 100)
        henon_fractions_test(mapper; k = [2, 1])
        end
        @testset "supervised" begin
        mapper = AttractorsViaFeaturizing(ds, henon_featurizer;
        Ttr = 100, attractors_ic = Dataset([u1]), clustering_threshold = 20.0)
        henon_fractions_test(mapper; k = [1, -1])
        end
    end

end
end
