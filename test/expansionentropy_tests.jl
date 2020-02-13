using ChaosTools, DynamicalSystemsBase, Test

@testset "maximalexpansion" begin
@test ChaosTools.maximalexpansion([1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]) ≈ 3.0 * 2.23606797749979 * 2.0
@test ChaosTools.maximalexpansion([1 0; 0 1]) == 1
end


@testset "discrete 1D" begin
    # Test expansionentropy_batch on discrete dynamical systems.
    tent_eom(x, p, n) = (x < -0.2 ? -0.6 : (x < 0.4 ? 3x : 2(1-x)))
    tent_jacob(x, p, n) = (x < -0.2 ? 0 : (x < 0.4 ? 3 : -2))
    tent = DiscreteDynamicalSystem(tent_eom, 0.2, nothing, tent_jacob)
    _, tent_meanlist, tent_stdlist = expansionentropy_batch(tent, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, steps=60)

    exact_ee = log(2)
    for (i, mean) in enumerate(tent_meanlist)
        @test abs(mean/i - exact_ee) < 0.05
    end

    ee = expansionentropy(tent, rand, x->0<x<1; batchcount=100, samplecount=100000, steps=60)
    @test abs(ee/exact_ee - 1) < 0.05
end

@testset "discrete 1D regular" begin
    expand2_eom(x, p, n) = 2*x
    expand2_jacob(x, p, n) = 2
    expand2 = DiscreteDynamicalSystem(expand2_eom, 0.2, nothing, expand2_jacob)
    _, expand_meanlist, _ = expansionentropy_batch(expand2, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, steps=10)

    for (i, mean) in enumerate(expand_meanlist)
        @test -0.1< mean/i < 0.1
    end
end

@testset "discrete 2D" begin
    cat = DynamicalSystemsBase.Systems.arnoldcat()
    cat_gen() = [rand(), rand()]
    cat_inside(x) = true

    _, cat_meanlist, _ = expansionentropy_batch(cat, cat_gen, cat_inside; batchcount=100, samplecount=100, steps=30, dt=1)

    exact_ee = log( 1/2 * (3 + sqrt(5)))
    for i in 1:length(cat_meanlist)
        @test abs(cat_meanlist[i]/i - exact_ee) < 1e-6
    end

    ee = expansionentropy(cat, cat_gen, cat_inside; batchcount=100, samplecount=100, steps=30, dt=1)
    @test ee ≈ exact_ee
end

@testset "continuous 3D" begin
    lor = DynamicalSystemsBase.Systems.lorenz()
    lor_gen() = [rand()*40-20, rand()*60-30, rand()*50]
    lor_isinside(x) = -20 < x[1] < 20 && -30 < x[2] < 30 && 0 < x[3] < 50
    ee = expansionentropy(lor, lor_gen, lor_isinside; batchcount=100, samplecount=100, steps=40, dt=1.0)

    @test abs(ee - 0.9) < 0.05
end

@testset "continusou 3D regular" begin
    lor2 = DynamicalSystemsBase.Systems.lorenz(ρ=160)
    tr = trajectory(lor2, 200.0, dt = 0.005, Ttr=20)
    x, y, z = columns(tr)

    lor2_gen, lor2_isinside = boxregion(map(minimum, [x,y,z]), map(maximum, [x,y,z]))

    ee = expansionentropy(lor2, lor2_gen, lor2_isinside; batchcount=100, samplecount=100, steps=20, dt=1.0, Ttr=40)
    @test abs(ee) < 0.05

    _, meanlist, _ = expansionentropy_batch(lor2, lor2_gen, lor2_isinside; batchcount=100, samplecount=100, steps=20, dt=1.0, Ttr=40)
    @test all(meanlist[10:20] .< 0.01)
end
