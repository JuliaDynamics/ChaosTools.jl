using DynamicalSystems
using Test
include("../ChaosTools.jl/src/basins/basins_fractions.jl")

@testset "Basin Stability Pendulum" begin
    @inline @inbounds function damped_driven_pendulum(u, p, t)
        α, T, K = p
        du1 = u[2]
        du2 = -α*u[2] +T -K*sin(u[1]) #Equation in bSTAB paper has a typo, with the first term being -α*u[1]. It actually is -α*u[2], as written here.
        return SVector{2}(du1, du2)
    end

    function feature_extraction(t, y, extract_params=NamedTuple())
        # Ttr = extract_params.Ttr;
        # idx_steady = findall(x->x>Ttr, t) #I don't think we need to do this since the integration already takes the transients out
        Δ = abs(maximum(y[:, 2]) - mean(y[:,2])) #"spread" of vaqlues, how far max and mean are
        X = zeros(2)
        if(Δ < 0.01) #mean ~ max -> FP
            X[1,1] = 1
            X[2,1] = 0
        else #max > mean -> LC
            X[1,1] = 0
            X[2,1] = 1
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
    minLimits = [-π + asin(T/K), -10]; # must be of length dof
    maxLimits = [π  + asin(T/K), 10]; # must be of length dof
    samplingVarDims = [true; true]; # boolean: dims to vary # select dimensions for which to create sampling points (some dimensions may be excluded from the sampling through this selection)
    samplingPDF = "uniform"; # sampling strategy / probability density
    ic_grid = generate_ic_grid(N, minLimits, maxLimits, samplingVarDims, samplingPDF); # draw initial conditions from the state space subspace specified by the % limits per state dimension and the number of samples

    #---Running for supervised
    #prepare templates
    y0 = [0.5 0; 2.7 0]' #each IC along a row
    templates_labels = ["IC", "LC"]

    S, class_labels = basin_fractions(ds, feature_extraction, ic_grid, "supervised"; T=Texec, Ttr=Ttr, Δt=Δt, clust_mod="supervised", y0=y0, templates_labels=templates_labels )
    # S = (Dict(2 => 0.8452,1 => 0.1548)
    @test S[1] == 0.17
    @test S[2] == 0.83

    #-- plot basins
    # cmap = matplotlib.colors.ListedColormap(["red", "white"])
    # fig = figure()
    # scatter(ic_grid[1,:], ic_grid[2,:], c=class_labels, cmap=cmap)
    # colorbar()
    # savefig("pendulum-basins-colored-supervised.png")

    S, class_labels = basin_fractions(ds, feature_extraction, ic_grid, "unsupervised"; T=Texec, Ttr=Ttr, Δt=Δt, clust_mod="supervised", y0=y0, templates_labels=templates_labels )
    # S = (Dict(2 => 0.1548,1 => 0.8452)
    @test S[1] == 0.83
    @test S[2] == 0.17

end


@testset "Basin Stability Duffing" begin
    Texec = 100
    Ttr = 900
    fs = 50
    Δt = 1/fs
    ds = Systems.duffing([0., 0.], f=0.2, ω=1, d=0.08, β=0.0)


    #region of interest (roi)
    N = 5000
    minLimits = [-1, -0.5]; # must be of length <props.model.dof>
    maxLimits = [1.0, 1.0]; # must be of length <props.model.dof>
    samplingVarDims = [true; true]; # boolean: dims to vary # select dimensions for which to create sampling points (some dimensions may be excluded from the sampling through this selection)
    samplingPDF = "uniform"; # sampling strategy / probability density
    ic_grid = generate_ic_grid(N, minLimits, maxLimits, samplingVarDims, samplingPDF; seed=2); # draw initial conditions from the state space subspace specified by the % limits per state dimension and the number of samples

    function feature_extraction(t, y, extract_params=NamedTuple())
        # % we will simply go for the mean and the std of the first state
        X = [maximum(y[:,1]); std(y[:,1])]
        return X
    end

    #---Running for supervised
    #templates
    y0 = [0.21 0.02; 1.05 0.77; -0.67 0.02; -0.46 0.3; -0.43 0.12]'
    S, class_labels = basin_fractions(ds, feature_extraction, ic_grid, "supervised"; T=Texec, Ttr=Ttr, Δt=Δt, clust_mod="supervised", y0=y0)
    # S = (Dict(4 => 0.0248,2 => 0.5086,3 => 0.028,5 => 0.2424,1 => 0.1962), [1, 2, 5, 2, 2, 1, 2, 1, 5, 5  …  2, 5, 1, 2, 2, 2, 2, 2, 1, 2])

    @test S[1] == 0.1962
    @test S[2] == 0.5086
    @test S[3] == 0.028
    @test S[4] == 0.0248
    @test S[5] == 0.2424

    # println("Test No2. Unsupervised")
    S, class_labels = basin_fractions(ds, feature_extraction, ic_grid, "unsupervised"; T=Texec, Ttr=Ttr, Δt=Δt, clust_mod="supervised", y0=y0)
    # S = (Dict(4 => 0.028,2 => 0.2424,3 => 0.1962,5 => 0.0248,1 => 0.5086), [3, 1, 2, 1, 1, 3, 1, 3, 2, 2  …  1, 2, 3, 1, 1, 1, 1, 1, 3, 1])

    @test S[1] == 0.5086
    @test S[2] == 0.2424
    @test S[3] == 0.1962
    @test S[4] == 0.028 
    @test S[5] == 0.0248

    #plot basins
    # cmap = matplotlib.colors.ListedColormap(["red", "gray", "orange", "cyan", "blue"])
    # fig = figure()
    # scatter(ic_grid[1,:], ic_grid[2,:], c=class_labels, cmap=cmap)
    # gca().set_aspect("equal")
    # colorbar()
    # savefig("duffing-basins-colored-unsupervised.png")
end
