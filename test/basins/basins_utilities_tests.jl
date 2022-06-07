
@testset "matching attractors" begin
    # TODO: Need to test what happens if I have one attractor with key 4 and one with 1.
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
                match_attractors!(b₋, a₋, b₊, a₊, method)
                for k in keys(a₊)
                    dist = minimum(norm(x .- y) for x ∈ a₊[k] for y ∈ a₋[k])
                    @test dist < 0.2
                end
            end
        end
    end
    @testset "unique attractor ids" begin
        # TODO:
    end
end
