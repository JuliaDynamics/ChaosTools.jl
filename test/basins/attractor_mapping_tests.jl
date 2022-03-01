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
        mapper = AttractorsViaProximity(ds, attractors, 1e-3; Ttr = 100)
        @test 1  == mapper(u1)
        @test -1 == mapper(u2)
        henon_fractions_test(mapper)
    end

    @testset "Featurizing" begin
        function henon_featurizer(A, t)
            x = [mean(A[:, 1]), mean(A[:, 2])]
            return any(isinf, x) ? [200.0, 200.0] : x
        end
        @testset "unsupervised" begin
        mapper = AttractorsViaFeaturizing(ds, henon_featurizer; Ttr = 100)
        # Notice that unsupervised clustering cannot support "divergence to infinity",
        # which it identifies as another attractor (in fact, the first one).
        henon_fractions_test(mapper; k = [2, 1])
        end
        @testset "supervised" begin
        mapper = AttractorsViaFeaturizing(ds, henon_featurizer;
        Ttr = 100, attractors_ic = Dataset([u1]), clustering_threshold = 20.0)
        henon_fractions_test(mapper)
        end
    end

@testset "Lorenz-84 system" begin
    F = 6.886
    G = 1.347
    a = 0.255
    b = 4.0
    ds = Systems.lorenz84(; F, G, a, b)
    u1 = [2.0, 1, 0] # periodic
    u2 = [-2.0, 1, 0] # chaotic
    u3 = [0, 1.5, 1.0] # fixed point
    diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9)

    M = 200
    xg = range(-3, 3; length = M)
    yg = range(-3, 3; length = M)
    zg = range(-3, 3; length = M)
    grid = (xg, yg, zg)
    sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = minimum.(grid), max_bounds = maximum.(grid))
    ics = Dataset([sampler() for i in 1:1000])

    function lorenz84_fractions_test(mapper)
        fs = basin_fractions(mapper, sampler; show_progress = false)
        @test length(fs) == 3
        for i in 1:3; @test 0 < fs[i] < 1; end
        @test sum(values(fs)) == 1

        # Deterministic test, should be tested with exact accuracy
        fs, labels = basin_fractions(mapper, ics; show_progress = false)
        @test sort!(unique(labels)) == [1,2,3]
        @test sort!(round.([fs[i] for i in 1:3]; digits=2)) == round.([0.165, 0.193, 0.642];  digits=2)
    end

    # TODO: At the moment, mapping via recurrences or proximity gives DIFFERENT
    # fractions. This needs to be resolved. I am using fairly high accuracy in all versions.
    # Which one gives the "correct" results?

    @testset "Proximity" begin
        udict = (1 => u1, 2 => u2, 3 => u3)
        attractors = Dict(
            k => trajectory(ds, 100, v; Ttr=100, Δt = 0.01, diffeq) for (k,v) in udict
        )

        # Compute minimum distance between attractors
        mapper = AttractorsViaProximity(ds, attractors; Ttr=1000, Δt=0.1, diffeq)
        @test 1 == mapper(u1)
        @test 2 == mapper(u2)
        @test 3 == mapper(u3)
        lorenz84_fractions_test(mapper)
    end

    @testset "Recurrences" begin
        mapper = AttractorsViaRecurrences(ds, grid;
            Δt = 0.2, mx_chk_fnd_att = 400, mx_chk_loc_att = 400,
            mx_chk_att = 20, diffeq
        )
        @test 1 == mapper(u1)
        @test 2 == mapper(u2)
        @test 3 == mapper(u3)
        lorenz84_fractions_test(mapper)
    end

end

end
end
