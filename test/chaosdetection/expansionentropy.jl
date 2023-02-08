using ChaosTools, Test

@testset "maximalexpansion" begin
    @test ChaosTools.maximalexpansion(
        [1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]
        ) ≈ 3.0 * 2.23606797749979 * 2.0
    @test ChaosTools.maximalexpansion([1 0; 0 1]) == 1
end


@testset "discrete 2D" begin
    function arnoldcat_rule(u, p, n)
        x,y = u
        return SVector{2}((2x + y) % 1.0, (x + y) % 1.0)
    end
    cat = DeterministicIteratedMap(arnoldcat_rule, [0.001245, 0.00875])
    cat_gen() = [rand(), rand()]
    cat_inside(x) = true
    arnoldcat_jacob(u, p, n) = SMatrix{2,2}(2.0, 1, 1, 1)

    exact_ee = log( 1/2 * (3 + sqrt(5)))

    times, Es, ee = expansionentropy(cat, cat_gen, cat_inside;
        J = arnoldcat_jacob,  batches=100, N=100, steps=30
    )
    @test ee ≈ exact_ee
end
