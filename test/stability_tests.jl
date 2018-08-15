using ChaosTools
using Test

println("\nTesting type stability...")
@testset "Type stability" begin
for ds ∈ [Systems.towel(), Systems.lorenz()]

    pinteg = parallel_integrator(ds, [get_state(ds), get_state(ds)+1e-9])
    @test_nowarn @inferred lyapunov(pinteg, 1000, 100, 1, 1e-9, 1e-6, 1e-12)
    @test_nowarn @inferred lyapunov(ds, 1000)

    tinteg = tangent_integrator(ds, 3)
    @test_nowarn @inferred lyapunovs(tinteg, 1000, 1, 100, Val{3}())
    @test_nowarn @inferred lyapunovs(ds, 1000)

    tinteg = tangent_integrator(ds, 2)
    @test_nowarn @inferred ChaosTools.gali(tinteg, 1000, 1, 1e-12)
    @test_nowarn @inferred gali(ds, 2, 1000)

end
end
