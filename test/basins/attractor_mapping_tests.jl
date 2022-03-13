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
            @test 0.1 < fs[k] < 0.9
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








end








@testset "Lorenz-84 system: interlaced close-by" begin
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

    function lorenz84_fractions_test(mapper; e = 1e-2, known = false)
        fs = basin_fractions(mapper, sampler; show_progress = false)
        @test length(fs) == 3
        for i in 1:3; @test 0 < fs[i] < 1; end
        @test sum(values(fs)) == 1

        # Deterministic test with known initial conditions. Must be strict.
        fs, labels = basin_fractions(mapper, ics; show_progress = false)
        @test sort!(unique(labels)) == [1,2,3]
        found_fs = sort(collect(values(fs)))

        # Expected fractions come from the Proximity version, which should
        # give the most correct numbers
        expected_fs_raw = Dict(2 => 0.165, 3 => 0.642, 1 => 0.193)
        expected_fs = sort!(collect(values(expected_fs_raw)))

        errors = abs.(expected_fs .- found_fs)
        for er in errors
            @test er .≤ e
        end

        if known # also test whether the attractor index is correct
            for i in 1:3
                @test abs(fs[i] - expected_fs_raw[i]) ≤ e
            end
        end
    end

    @testset "Proximity" begin
        udict = (1 => u1, 2 => u2, 3 => u3)
        attractors = Dict(
            k => trajectory(ds, 400, v; Ttr=100, Δt = 0.01, diffeq) for (k,v) in udict
        )

        # Compute minimum distance between attractors
        mapper = AttractorsViaProximity(ds, attractors; Ttr=1000, Δt=0.1, diffeq)
        @test 1 == mapper(u1)
        @test 2 == mapper(u2)
        @test 3 == mapper(u3)
        lorenz84_fractions_test(mapper; e = 1e-15, known = true)
    end

    @testset "Recurrences" begin
        mapper = AttractorsViaRecurrences(ds, grid;
            Δt = 0.2, mx_chk_fnd_att = 400, mx_chk_loc_att = 400,
            mx_chk_att = 20, diffeq, show_progress = false
        )
        @test 1 == mapper(u1)
        @test 2 == mapper(u2)
        @test 3 == mapper(u3)
        lorenz84_fractions_test(mapper; e = 1e-3)
    end

    @testset "Featurizing" begin
        function lorenz84_featurizer(A, t)
            x1 = ChaosTools.Entropies.genentropy(Dataset(A[:, 2:3]), 0.1)
            mini, maxi = minmaxima(A)
            x2 = maxi[1] - mini[1]
            x3 = maxi[3] - mini[3]
            return [x1, x2, x3]
        end

        @testset "unsupervised" begin
            # Unfortunately, the unsupervised version is straight-out incapable of
            # actually finding the attractors. Half the time it coalesces the periodic
            # and chaotic attractors, and often it attribes -1 to some initial conditions.
            mapper = AttractorsViaFeaturizing(ds, lorenz84_featurizer;
            T = 100, Δt = 0.1, min_neighbors = 1)
            # lorenz84_fractions_test(mapper; e = 1e-2)
        end
        @testset "supervised" begin
            mapper = AttractorsViaFeaturizing(ds, lorenz84_featurizer;
            T = 100, Δt = 0.1, attractors_ic = Dataset([u1, u2, u3]))
            lorenz84_fractions_test(mapper; known = true, e = 1e-2)
        end
    end

end

@testset "Duffing oscillator: stroboscopic map" begin
    ds = Systems.duffing([0.1, 0.25]; ω = 1.0, f = 0.2, d = 0.15, β = -1)
    xg = yg = range(-2.2,2.2,length=100)
    grid = (xg, yg)
    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = minimum.(grid), max_bounds = maximum.(grid))
    diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9)
    ics = Dataset([sampler() for i in 1:1000])
    T = 2π/1.0
    smap = stroboscopicmap(ds, T; diffeq)
    u1 = [-0.8, 0]
    u2 = [1.8, 0]

    function duffing_fractions_test(mapper; e = 1e-2)
        fs = basin_fractions(mapper, sampler; show_progress = false)
        @test length(fs) == 2
        for i in 1:2; @test 0.4 < fs[i] < 0.6; end
        @test sum(values(fs)) == 1

        # Deterministic test with known initial conditions. Must be strict.
        fs, labels = basin_fractions(mapper, ics; show_progress = false)
        @test sort!(unique(labels)) == [1,2]
        found_fs = sort(collect(values(fs)))

        # Expected fractions come from the Proximity version, which should
        # give the most correct numbers
        expected_fs_raw = Dict(2 => 0.504, 1 => 0.496)
        expected_fs = sort!(collect(values(expected_fs_raw)))

        errors = abs.(expected_fs .- found_fs)
        for er in errors
            @test er .≤ e
        end
    end

    @testset "Proximity" begin
        udict = (1 => u1, 2 => u2)
        attractors = Dict(
            k => trajectory(smap, 5, v; Ttr=100, ) for (k,v) in udict
        )

        # Compute minimum distance between attractors
        mapper = AttractorsViaProximity(smap, attractors; Ttr=1000)
        @test 1 == mapper(u1)
        @test 2 == mapper(u2)
        duffing_fractions_test(mapper; e = 1e-15)
    end

    @testset "Recurrences" begin
        mapper = AttractorsViaRecurrences(smap, grid; show_progress = false)
        @test 1 == mapper(u1)
        @test 2 == mapper(u2)
        lorenz84_fractions_test(mapper; e = 1e-3)
    end



end


# TODO: Tests for stroboscopic map (forced duffing), projected system (magnetic)
# and poincare map (thomas cyclical)

end
