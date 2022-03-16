using ChaosTools
using DynamicalSystemsBase
using DynamicalSystemsBase.DelayEmbeddings
using Test
using LinearAlgebra
using OrdinaryDiffEq
using Random
using Statistics

@testset "AttractorMappers" begin

# Define generic testing framework
function test_basins(ds, u0s, grid, expected_fs_raw, featurizer;
        rerr = 1e-3, ferr = 1e-3, ε = nothing, clustering_threshold = 20.0,
        diffeq = NamedTuple(),
    )
    # u0s is Vector{Pair}
    known_attractors = Dict(
        k => trajectory(ds, 10000, v; Δt = 1, Ttr=100) for (k,v) in u0s if k ≠ -1
    )
    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = minimum.(grid), max_bounds = maximum.(grid)
    )
    ics = Dataset([sampler() for i in 1:1000])
    expected_fs = sort!(collect(values(expected_fs_raw)))
    known_ids = sort!(collect(u[1] for u in u0s))

    # reusable testing function
    function test_basin_fractions(mapper;
            err = 1e-3, known=false, single_u_mapping = true,
            known_ids = known_ids
        )
        if single_u_mapping
            for (k, u0) in u0s
                @test k  == mapper(u0)
            end
        end
        # Generic test
        fs = basin_fractions(mapper, sampler; show_progress = false)
        @test sort!(collect(keys(fs))) == known_ids
        for k in keys(fs)
            @test 0 < fs[k] < 1
        end
        @test sum(values(fs)) == 1

        # Precise test with known initial conditions
        fs, labels = basin_fractions(mapper, ics; show_progress = false)
        @test sort!(unique(labels)) == known_ids
        found_fs = sort(collect(values(fs)))
        errors = abs.(expected_fs .- found_fs)
        for er in errors
            @test er .≤ err
        end
        if known # also test whether the attractor index is correct
            for k in known_ids
                @test abs(fs[k] - expected_fs_raw[k]) ≤ err
            end
        end
    end

    @testset "Proximity" begin
        mapper = AttractorsViaProximity(ds, known_attractors, ε; diffeq, Ttr = 100)
        test_basin_fractions(mapper; known = true, err = 1e-15)
    end
    @testset "Recurrences" begin
        mapper = AttractorsViaRecurrences(ds, grid; diffeq, show_progress = false)
        test_basin_fractions(mapper; err = rerr)
    end
    @testset "Featurizing, unsupervised" begin
        mapper = AttractorsViaFeaturizing(ds, featurizer; diffeq, Ttr = 100)
        test_basin_fractions(mapper; err = ferr, single_u_mapping=false, known_ids=[1,2])
    end
    @testset "Featurizing, supervised" begin
        attractors_ic = Dataset([v for (k,v) in u0s if k ≠ -1])
        mapper = AttractorsViaFeaturizing(ds, featurizer;
            Ttr = 100, attractors_ic, clustering_threshold, diffeq,
        )
        test_basin_fractions(mapper; err = ferr, single_u_mapping = false)
    end
end


@testset "Henon map: discrete & divergence" begin
    u0s = [1 => [0.0, 0.0], -1 => [0.0, 2.0]]
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.0, 2.0; length=100)
    grid = (xg, yg)
    expected_fs_raw = Dict(1 => 0.451, -1 => 0.549)
    function henon_featurizer(A, t)
        # Notice that unsupervised clustering cannot support "divergence to infinity",
        # which it identifies as another attractor (in fact, the first one).
        x = [mean(A[:, 1]), mean(A[:, 2])]
        return any(isinf, x) ? [200.0, 200.0] : x
    end
    test_basins(ds, u0s, grid, expected_fs_raw, henon_featurizer; ε = 1e-3)
end


@testset "Lorenz-84 system: interlaced close-by" begin
    F = 6.886; G = 1.347; a = 0.255; b = 4.0
    ds = Systems.lorenz84(; F, G, a, b)
    u0s = [
        1 => [2.0, 1, 0], # periodic
        2 => [-2.0, 1, 0], # chaotic
        3 => [0, 1.5, 1.0], # fixed point
    ]
    diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9)
    M = 200; z = 4
    xg = yg = zg = range(-z, z; length = M)
    grid = (xg, yg, zg)
    expected_fs_raw = Dict(2 => 0.165, 3 => 0.642, 1 => 0.193)

    function lorenz84_featurizer(A, t)
        x1 = ChaosTools.Entropies.genentropy(Dataset(A[:, 2:3]), 0.1)
        mini, maxi = minmaxima(A)
        x2 = maxi[1] - mini[1]
        x3 = maxi[3] - mini[3]
        return [x1, x2, x3]
    end

    test_basins(ds, u0s, grid, expected_fs_raw, lorenz84_featurizer; diffeq, ferr=1e-2)
end


@testset "Duffing oscillator: stroboscopic map" begin

    ds = Systems.duffing([0.1, 0.25]; ω = 1.0, f = 0.2, d = 0.15, β = -1)
    xg = yg = range(-2.2, 2.2; length=100)
    grid = (xg, yg)
    diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9)
    T = 2π/1.0
    smap = stroboscopicmap(ds, T; diffeq)
    u0s = [
        1 => [-0.8, 0],
        2 => [1.8, 0],
    ]
    expected_fs_raw = Dict(2 => 0.506, 1 => 0.494)
    function duffing_featurizer(A, t)
        return [A[end][1]]
    end

    test_basins(smap, u0s, grid, expected_fs_raw, duffing_featurizer; diffeq, ferr=1e-2)

end



end

# TODO: Add a call to `basins_of_attraction` within the `test_basin_fractions` function.
# TODO: Tests for  projected system (magnetic) and poincare map (thomas cyclical)
# TODO: Tests of Lorenz84 fail, significant differences between recurrence and proximity
# TODO: Tests of Duffing fail, significant differences between recurrence and proximity