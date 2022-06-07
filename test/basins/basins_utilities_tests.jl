using Test, ChaosTools, ChaosTools.DynamicalSystemsBase

d, α, ω = 0.3, 0.2, 0.5
ds = Systems.magnetic_pendulum(; d, α, ω)
xg = yg = range(-3, 3, length = 100)
ds = projected_integrator(ds, 1:2, [0.0, 0.0])
mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
b₋, a₋ = basins_of_attraction(mapper; show_progress = false)

@testset "basins_fractions(::Array)" begin
    fs = basins_fractions(b₋)
    @test keytype(fs) <: Integer
    # Default magnetic pendulum has approximately 33% each fraction
    @test all(v -> 0.3 < v < 0.35, values(fs))
end

@testset "matching attractors" begin
    @testset "method $method" for method ∈ (:overlap, :distance)
        @testset "γ3 $γ3" for γ3 ∈ [0.2, 0.1] # still 3 at 0.2, but only 2 at 0.1
            ds = Systems.magnetic_pendulum(; d, α, ω,  γs = [1, 1, γ3])
            ds = projected_integrator(ds, 1:2, [0.0, 0.0])
            mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
            b₊, a₊ = basins_of_attraction(mapper; show_progress = false)
            match_attractor_ids!(b₋, a₋, b₊, a₊, method)
            for k in keys(a₊)
                dist = minimum(norm(x .- y) for x ∈ a₊[k] for y ∈ a₋[k])
                @test dist < 0.2
            end
        end
    end
end

@testset "unique attractor ids" begin
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
    unique_attractor_ids!(allatts, 100.0) # all odd keys become 1
    @test all(haskey(d, 1) for d in allatts)
    @test all(haskey(d, 2) for d in allatts)
    # Test with distance enough to increment
    allatts2 = deepcopy(allatts)
    unique_attractor_ids!(allatts2, 0.1) # all keys there were `2` get incremented
    @test all(haskey(d, 1) for d in allatts2)
    for i in 2:length(jrange)
        @test haskey(allatts2[i], i+1)
        @test !haskey(allatts2[i], 2)
    end
    @test haskey(allatts2[1], 2)
end
