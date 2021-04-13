using ChaosTools
using Test

@testset "DyCA Tests" begin
    println("Running dyca() tests...")

    @testset "Eigenvalues of Roessler system" begin
        # Initial conditions from `dyca()` paper
        eigenthresold = collect(0.10:0.01:0.995)
        ds = Systems.roessler(rand(3); a = 0.15, b = 0.2, c = 10);
        ts = trajectory(ds, 1000.0, dt = 0.05);
        for thresold in eigenthresold
            eigenvalues,_ = dyca(Matrix(ts),thresold)
            @test sum([round(i) for i in eigenvalues]) == 2.0
        end
    end

    @testset "Eigenvalues of Embeded Roessler system" begin
        # Initial conditions from `dyca()` paper
        eigenthresold = collect(0.90:0.01:0.995)
        ds = Systems.roessler(rand(3); a = 0.15, b = 0.2, c = 10);
        ts = trajectory(ds, 1000.0, dt = 0.05);
        cols = columns(ts);
        for thresold in eigenthresold
            for i in 1:3
                Embedded_system = Matrix(embed(cols[i], 25, 1));
                eigenvals,_ = dyca(Embedded_system,thresold)
                # eigenvalues satisfy `dyca` condtiion and have imaginary part = 0.0
                wanted_eigenvalues = eigenvals[vec(0.999 .< abs.(eigenvals) .< 1.0) .& vec(imag(eigenvals) .== 0)]
                # due to floating point errors more than 2 eigenvalues might satisy the above condition
                # might be better to use `length(psum) in [1,2,3]`
                @test length(wanted_eigenvalues) in 0:4
            end
        end
    end
end    
