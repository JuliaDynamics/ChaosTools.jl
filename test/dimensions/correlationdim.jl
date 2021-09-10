using ChaosTools, DynamicalSystemsBase
using Test
using Statistics
using ChaosTools.DynamicalSystemsBase
using ChaosTools.DelayEmbeddings

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

println("\nTesting correlation sum...")
@testset "Correlation dim" begin
    @testset "Henon Map" begin
        ds = Systems.henon()
        ts = trajectory(ds, 5000)
        R = 1.5maximum(maxima(ts) - minima(ts))
        # check for right normalisation with various combinations of `q` and `w`
        @test correlationsum(ts, R; q = 2, w = 0) ≈ 1
        @test correlationsum(ts, R; q = 2, w = 1) ≈ 1
        @test correlationsum(ts, R; q = 2.2, w = 0) ≈ 1
        @test correlationsum(ts, R; q = 2.2, w = 1) ≈ 1
        es = 10 .^ range(-3, stop = 0, length = 7)
        cs1 = correlationsum(ts, es)
        dim1 = linear_region(log.(es), log.(cs1))[2]
        test_value(dim1, 1.1, 1.3)
        cs2 = correlationsum(ts, es; q = 2.0001)
        dim2 = linear_region(log.(es), log.(cs2))[2]
        test_value(dim2, 1.1, 1.3)
        test_value(grassberger_dim(ts), 1.1, 1.3)
    end
    @testset "Lorenz System" begin
        ds = Systems.lorenz()
        ts = trajectory(ds, 1000; Δt = 0.1)
        es = 10 .^ range(-3, stop = 1, length = 8)
        cs1 = correlationsum(ts, es)
        dim1 = linear_region(log.(es), log.(cs1))[2]
        test_value(dim1, 1.85, 2.2)
        cs2 = correlationsum(ts, es; q = 2.001, w = 5)
        dim2 = linear_region(log.(es), log.(cs2))[2]
        test_value(dim2, 1.85, 2.2)
        test_value(grassberger_dim(ts), 1.85, 2.2)
    end
end

println("\nTesting box-assisted correlation sum...")
@testset "Box-assisted correlation sum" begin
    @testset "Henon Map" begin
        ds = Systems.henon()
        ts = standardize(trajectory(ds, 10000))
        r0 = estimate_r0_buenoorovio(ts)[1]
        es = r0 .* 10 .^ range(-2, stop = 0, length = 10)
        C = correlationsum(ts, es)
        @test boxed_correlationsum(ts, es, r0) ≈ C
        @test boxed_correlationsum(ts, es) ≈ C
        @test boxed_correlationsum(ts, es, r0; q = 2.3) ≈ correlationsum(ts, es, q = 2.3)
        @test boxed_correlationsum(ts, es, r0; w = 10) ≈ correlationsum(ts, es; w = 10)
        @test boxed_correlationsum(ts, es, r0; q = 2.3, w = 10) ≈ correlationsum(ts, es; q = 2.3, w = 10)
    end
    @testset "Lorenz System" begin
        ds = Systems.lorenz()
        ts = standardize(trajectory(ds, 1000; Δt = 0.1))
        r0 = estimate_r0_buenoorovio(ts)[1]
        es = r0 .* 10 .^ range(-2, stop = 0, length = 10)
        C = correlationsum(ts, es)
        @test boxed_correlationsum(ts, es, r0) ≈ C
        @test boxed_correlationsum(ts, es) ≈ C
        @test boxed_correlationsum(ts, es, r0; q = 2.3) ≈ correlationsum(ts, es; q = 2.3)
        @test boxed_correlationsum(ts, es, r0; w = 10) ≈ correlationsum(ts, es; w = 10)
        @test boxed_correlationsum(ts, es, r0; q = 2.3, w = 10) ≈ correlationsum(ts, es; q = 2.3, w = 10)
    end
end


println("\nTesting the fixed mass correlation sum...")
@testset "Fixed mass correlation sum" begin
    @testset "Henon Map" begin
        ds = Systems.henon()
        ts = trajectory(ds, 10000)
        rs, ys = correlationsum_fixedmass(ts, 30)
        test_value(linear_region(rs, ys)[2], 1.0, 1.3)
        rs, ys = correlationsum_fixedmass(ts, 30; M = 2000)
        test_value(linear_region(rs, ys)[2], 1.0, 1.3)
        rs, ys = correlationsum_fixedmass(ts, 30; w = 5)
        test_value(linear_region(rs, ys)[2], 1.0, 1.3)
    end
    @testset "Lorenz System" begin
        ds = Systems.lorenz()
        ts = trajectory(ds, 1000; Δt = 0.1)
        rs, ys = correlationsum_fixedmass(ts, 30)
        test_value(linear_region(rs, ys)[2], 1.7, 2.2)
        rs, ys = correlationsum_fixedmass(ts, 30; M = 2000)
        test_value(linear_region(rs, ys)[2], 1.7, 2.2)
        rs, ys = correlationsum_fixedmass(ts, 30; w = 5)
        test_value(linear_region(rs, ys)[2], 1.7, 2.2)
    end
end


println("\nTesting Takens' best estimate")
@testset "Takens best" begin
    @testset "Henon Map" begin
        ds = Systems.henon()
        ts = trajectory(ds, 5000)
        x = ts[:, 1]
        X = embed(x, 2, 1)
        D_C, D_C_95u, D_C_95l = takens_best_estimate(X, std(x)/4)
        @test 1.15 < D_C < 1.25
        @test D_C_95u < 1.05*D_C
        @test D_C_95l > 0.95*D_C
    end
    @testset "Lorenz System" begin
        ds = Systems.lorenz()
        tr = trajectory(ds, 2000; Δt = 0.1)
        x = tr[:, 1]
        τ = estimate_delay(x, "mi_min", 1:20)
        X = embed(x, 4, τ)
        D_C, D_C_95u, D_C_95l = takens_best_estimate(X, std(x)/4)
        @test 1.85 < D_C < 2.1
        @test D_C_95u < 1.05*D_C
        @test D_C_95l > 0.95*D_C
    end
end
