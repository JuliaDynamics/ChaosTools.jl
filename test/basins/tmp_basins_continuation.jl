using Test, ChaosTools
using ChaosTools.DynamicalSystemsBase, ChaosTools.DelayEmbeddings
using Random




@testset "Newton map" begin
    # Newton algorithm for finding roots of a polynomial in the complex plane.
    # We change the locus of the roots continuously. 
    function newton_map(dz,z, p, n)
        f(x) = (x-1)*(x^2 + p[1])
        df(x)= (x-1)*2*x + (x^2 + p[1]) 
        z1 = z[1] + im*z[2]
        dz1 = f(z1)/df(z1)
        z1 = z1 - dz1
        dz[1]=real(z1)
        dz[2]=imag(z1)
        return
    end

    # dummy function to keep the initializator happy
    function newton_map_J(J,z0, p, n)
       return
    end

    ds = DiscreteDynamicalSystem(newton_map,[0.1, 0.2], [1. .+ im] , newton_map_J)

    ps = map(x -> exp(im*x), range(0., 2*pi/3; length = 100))

    xg = yg = range(-2.5, 2.5, length = 500)
    mapper = AttractorsViaRecurrences(ds, (xg, yg),
            mx_chk_fnd_att = 300,
            mx_chk_loc_att = 300
    )
    pidx = 1
    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = [-2,-2], max_bounds = [2,2]
    )
    continuation = RecurrencesSeedingContinuation(mapper;
        threshold = 0.29
    )
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, ps, pidx, sampler;
        show_progress = true, samples_per_parameter = 10000
    )

    for (i, p) in enumerate(ps)
        fs = fractions_curves[i]
        attractors = attractors_info[i]
        @test sum(values(fs)) ≈ 1
        attk = sort!(collect(keys(attractors)))
        @test length(attk) == 3
        for f in values(fs)
            @test f ≤ 0.5
        end
    end

end



@testset "Second Order Kuramoto network" begin
    using OrdinaryDiffEq
    using LinearAlgebra:norm
    using Statistics:mean
    using Graphs

    function second_order_kuramoto!(du, u, p, t)
        N = p[1]; drive = p[5]; damping = p[2]; coupling = p[3]; incidence = p[4];    
        du[1:N] .= u[1 + N:2*N]
        du[N+1:end] .= drive .- damping .* u[1 + N:2*N] .- coupling .* (incidence * sin.(incidence' * u[1:N]))
    end

    seed = 5386748129040267798
    Random.seed!(seed)
    # Set up the parameters for the network
    N = 30 # in this case this is the number of oscillators, the system dimension is twice this value
    g = random_regular_graph(N, 3)
    E = incidence_matrix(g, oriented=true)
    drive = [isodd(i) ? +1. : -1. for i = 1:N]
    K = 2.
    ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2*N), [N, 0.1, K, E, drive], (J,z0, p, n) -> nothing)
    diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e7)
    yg = range(-12.5, 12.5; length = 10)
    _get_rand_ic(y) = [pi*(rand(N) .- 0.5); y]
    psys = projected_integrator(ds, N+1:2*N, _get_rand_ic; diffeq)
    pgrid = ntuple(x -> yg, N)
    mapper = AttractorsViaRecurrences(psys, pgrid; Δt = .1, diffeq, 
        sparse = true,
        mx_chk_fnd_att = 100,
        mx_chk_loc_att = 100,
        mx_chk_att = 2,
        mx_chk_hit_bas = 10,
        unsafe = true)

    pidx = 3 
    ps = range(7., 10., length = 4) 
    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = minimum.(pgrid),  max_bounds = maximum.(pgrid)
    )
    distance_function = function (A, B)
        # Compute Euclidean distance between the average frequencies of the attractors. 
        # @show dataset_distance(A, B, Hausdorff())
        # @show dataset_distance(A, B)
        if norm(mean(A) - mean(B)) < 20
            @show  norm(mean(A) - mean(B))
        end
        return norm(mean(A) - mean(B))
        # return dataset_distance(A, B, Hausdorff())
    end
    continuation = RecurrencesSeedingContinuation(mapper;
        threshold = 4.99, metric = distance_function
    )
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, ps, pidx, sampler;
        show_progress = true, samples_per_parameter = 100
    )

end
