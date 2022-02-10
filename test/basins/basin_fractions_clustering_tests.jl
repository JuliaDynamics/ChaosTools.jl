using Test
using Statistics, Random
using ChaosTools, DelayEmbeddings, DynamicalSystemsBase

@testset "Basin Fractions Clustering Pendulum" begin

@testset "Damped Driven Pendulum" begin
    @inline @inbounds function damped_driven_pendulum(u, p, t)
        α, T, K = p
        du1 = u[2]
        du2 = -α*u[2] +T -K*sin(u[1]) #Equation in bSTAB paper has a typo, with the first term being -α*u[1]. It actually is -α*u[2], as written here.
        return SVector{2}(du1, du2)
    end

    function featurizer(y, t)
        Δ = abs(maximum(y[:, 2]) - mean(y[:,2])) #"spread" of values, how far max and mean are
        X = zeros(2)
        if(Δ < 0.01) #mean ~ max -> FP
            X[1] = 1
            X[2] = 0
        else #max > mean -> LC
            X[1] = 0
            X[2] = 1
        end
        return X #[0 1] for LC; [1 0] for FP
    end

    α = 0.1;
    T = 0.5;
    K = 1.0;
    p = [α, T, K]

    Texec = 1000
    Ttr = 950
    Δt= 1.0/25
    ds = ContinuousDynamicalSystem(damped_driven_pendulum, [0.,0.], p)

    #region of interest (roi)
    N = 100; # number of samples (N)
    min_limits = [-π + asin(T/K), -10]; # must be of length dof
    max_limits = [π  + asin(T/K), 10]; # must be of length dof
    sampling_method = "uniform"; # sampling strategy / probability density

    s, _ = statespace_sampler(min_bounds=min_limits, max_bounds=max_limits, method=sampling_method)
    ics_foo = s
    ics = [s() for i=1:N]
    ics = Dataset(ics)


    #---Running for supervised
    #prepare templates
    attractors_ic = Dataset([0.5 0; 2.7 0]) #each IC along a row

    # println("Test No. 1 Supervised, generated ics.")
    fs, class_labels = basin_fractions_clustering(ds, featurizer, ics, attractors_ic; show_progress = false, T = Texec, Ttr, Δt)
    # fs = (Dict(2 => 0.8452,1 => 0.1548)
    @test 0.10 < fs[1] < 0.22
    @test 0.75 < fs[2] < 0.90


    # println("Test No. 1 Supervised, generator ics.")
    fs = basin_fractions_clustering(ds, featurizer, ics_foo, attractors_ic; show_progress = false, N = N, T = Texec, Ttr, Δt)
    # fs = (Dict(2 => 0.8452,1 => 0.1548)
    @test 0.10 < fs[1] < 0.25
    @test 0.76 < fs[2] < 0.88


    # println("Test No2. Unsupervised, generated ics.")
    fs, class_labels = basin_fractions_clustering(ds, featurizer, ics; show_progress = false, T = Texec, Ttr, Δt)
    # fs = (Dict(2 => 0.1548,1 => 0.8452)
    @test 0.75 < fs[1] < 0.90
    @test 0.10 < fs[2] < 0.25
    @test fs[1] + fs[2] ≈ 1.0

    # plot basins
    # using PyPlot
    #make N = 10000 to replicate the paper
    # cmap = matplotlib.colors.ListedColormap(["red", "white"])
    # fig = figure()
    # scatter(ics[:,1], ics[:,2], c=class_labels, cmap=cmap)
    # colorbar()
    # savefig("pendulum-basins-colored-unsupervised.png")
end


@testset "Duffing Oscillator" begin
    Texec = 100
    Ttr = 900
    fs = 50
    Δt = 1/fs
    ds = Systems.duffing([0., 0.], f=0.2, ω=1, d=0.08, β=0.0)
    function featurizer(y, t)
        x1 = maximum(p[1] for p in y)
        X = [x1, std(y[:,1])]
        return X
    end

    #region of interest (roi)
    N = 1000
    min_limits = [-1, -0.5]; # must be of length <props.model.dof>
    max_limits = [1.0, 1.0]; # must be of length <props.model.dof>
    sampling_method = "uniform"; # sampling strategy / probability density

    s, _ = statespace_sampler(Random.MersenneTwister(1234); min_bounds=min_limits, max_bounds=max_limits, method=sampling_method)
    ics_foo = s
    ics = [s() for i=1:N]
    ics = Dataset(ics)


    #---Running for supervised
    #templates
    attractors_ic = Dataset([0.21 0.02; 1.05 0.77; -0.67 0.02; -0.46 0.3; -0.43 0.12])
    fs, class_labels = basin_fractions_clustering(ds, featurizer, ics, attractors_ic; show_progress = false, T=Texec, Ttr, Δt)
    #original result: fs = (Dict(4 => 0.0248,2 => 0.5086,3 => 0.028,5 => 0.2424,1 => 0.1962), [1, 2, 5, 2, 2, 1, 2, 1, 5, 5  …  2, 5, 1, 2, 2, 2, 2, 2, 1, 2])
    @test fs[5] == 0.252
    @test fs[4] == 0.023
    @test fs[2] == 0.492
    @test fs[3] == 0.024
    @test fs[1] == 0.209

    # The remaining tests all don't test the actual values anymore (not dealing with random)
    #The results depend on the (random) sampling of the ics, so results may very on subsequent tests. I am not sure how much, though.
    @test maximum(values(fs)) > 0.3
    @test length(fs) ≤ 5
    @test sum(values(fs)) ≈ 1.0

    fs = basin_fractions_clustering(ds, featurizer, ics_foo, attractors_ic; show_progress = false, N=N, T=Texec, Ttr, Δt)
    @test maximum(values(fs)) > 0.3
    @test length(fs) ≤ 5
    @test sum(values(fs)) ≈ 1.0


    fs, class_labels = basin_fractions_clustering(ds, featurizer, ics; show_progress = false, T=Texec, Ttr, Δt)
    #original result: fs = (Dict(4 => 0.028,2 => 0.2424,3 => 0.1962,5 => 0.0248,1 => 0.5086), [3, 1, 2, 1, 1, 3, 1, 3, 2, 2  …  1, 2, 3, 1, 1, 1, 1, 1, 3, 1])
    @test maximum(values(fs)) > 0.3
    @test length(fs) ≤ 5
    @test sum(values(fs)) ≈ 1.0

    #plot basins
    # cmap = matplotlib.colors.ListedColormap(["red", "gray", "orange", "cyan", "blue"])
    # fig = figure()
    # scatter(ics[:,1], ics[:,2], c=class_labels, cmap=cmap)
    # gca().set_aspect("equal")
    # colorbar()
    # savefig("duffing-basins-colored-unsupervised.png")
end

end
