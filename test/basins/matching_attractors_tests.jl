using Test, ChaosTools
using ChaosTools.DynamicalSystemsBase, ChaosTools.DelayEmbeddings

@testset "analytic" begin
    a_befo = Dict(1 => [SVector(0.0, 0.0)], 2 => [SVector(1.0, 1.0)])
    a_befo = Dict(keys(a_befo) .=> Dataset.(values(a_befo)))
    @testset "infinite threshold" begin
        a_afte = Dict(2 => [SVector(0.0, 0.0)], 1 => [SVector(2.0, 2.0)])
        a_afte = Dict(keys(a_afte) .=> Dataset.(values(a_afte)))
        rmap = match_attractor_ids!(a_afte, a_befo)
        @test rmap == Dict(1 => 2, 2 => 1)
        @test a_afte[1] == a_befo[1] == Dataset([SVector(0.0, 0.0)])
        @test haskey(a_afte, 2)
        @test a_afte[2] == Dataset([SVector(2.0, 2.0)])
    end
    @testset "separating threshold" begin
        a_afte = Dict(2 => [SVector(0.0, 0.0)], 1 => [SVector(2.0, 2.0)])
        a_afte = Dict(keys(a_afte) .=> Dataset.(values(a_afte)))
        rmap = match_attractor_ids!(a_afte, a_befo; threshold = 0.1)
        @test rmap == Dict(1 => 3, 2 => 1)
        @test a_afte[1] == a_befo[1] == Dataset([SVector(0.0, 0.0)])
        @test !haskey(a_afte, 2)
        @test a_afte[3] == Dataset([SVector(2.0, 2.0)])
    end
    @testset "No double duplication" begin
        rmap = Dict(4 => 3, 3 => 2)
        a = Dict(4 => 1)
        swap_dict_keys!(a, rmap)
        @test haskey(a, 3)
        @test !haskey(a, 2)
    end
end

@testset "magnetic pendulum" begin
    using LinearAlgebra: norm
    d, α, ω = 0.3, 0.2, 0.5
    ds = Systems.magnetic_pendulum(; d, α, ω)
    xg = yg = range(-3, 3, length = 100)
    ds = projected_integrator(ds, 1:2, [0.0, 0.0])
    mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
    b₋, a₋ = basins_of_attraction(mapper; show_progress = false)
    # still 3 attractors at γ3 = 0.2, but only 2 at 0.1
    @testset "γ3 $γ3" for γ3 ∈ [0.2, 0.1]
        set_parameter!(ds, :γs, [1, 1, γ3])
        mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
        b₊, a₊ = basins_of_attraction(mapper; show_progress = false)
        @testset "distances match" begin
            rmap = match_attractor_ids!(a₊, a₋)
            for k in keys(a₊)
                dist = minimum(norm(x .- y) for x ∈ a₊[k] for y ∈ a₋[k])
                @test dist < 0.2
            end
        end
        @testset "overlap match" begin
            fs0 = basins_fractions(b₊)
            rmap = match_basins_ids!(b₊, b₋)
            @test !isempty(rmap)
            fs1 = basins_fractions(b₊)
            @test sort!(collect(values(fs0))) == sort!(collect(values(fs1)))
        end
    end
end

@testset "matching attractors in vector" begin
    # Make fake attractors with points that become more "separated" as "parameter"
    # is increased
    jrange = 0.1:0.1:1
    allatts = [Dict(1 => [SVector(0.0, 0.0)], 2 => [SVector(j, j)]) for j in jrange]
    allatts = [Dict(keys(d) .=> Dataset.(values(d))) for d in allatts]
    for i in eachindex(jrange)
        if isodd(i) && i ≠ 1
            # swap key of first attractor to from 1 to i
            allatts[i][i] = allatts[i][1]
            delete!(allatts[i], 1)
        end
    end
    # Test with distance not enough to increment
    match_attractor_ids!(allatts; threshold = 100.0) # all odd keys become 1
    @test all(haskey(d, 1) for d in allatts)
    @test all(haskey(d, 2) for d in allatts)
    # Test with distance enough to increment
    allatts2 = deepcopy(allatts)
    match_attractor_ids!(allatts2; threshold = 0.1) # all keys there were `2` get incremented
    @test all(haskey(d, 1) for d in allatts2)
    for i in 2:length(jrange)
        @test haskey(allatts2[i], i+1)
        @test !haskey(allatts2[i], 2)
    end
    @test haskey(allatts2[1], 2)
end
