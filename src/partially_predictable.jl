using LinearAlgebra, Distributions
using Statistics: mean
#export #TODO

# TODO - write docstring
#
# [1] H. Wernecke, B. Sándor, and C. Gros, ‘How to test for partially
#     predictable chaos’, Scientific Reports, vol. 7, no. 1, Dec. 2017.

function predictability(ds::DynamicalSystem)
    # ========= Definitions of things which should be arguments =========== #
    # TODO - implement as function parameters
    alg = Vern9()
    maxiters = 1e9
    Ttr = 200 # Transient time
    T = 1e5 # Time for generating samples
    Tly =  2500 # Time for use by `lyapunov` TODO - principled way to choose?
    Tλmax = 200 # Maximum lyapunov prediction time (see below)
    n_samples = 1000
    scale_ly = 50 # Scale factor for Lyapunov prediction time
    δ_range = 10.0 .^ (-9:-6)
    # ===================================================================== #

    # ======================== Internal Constants ========================= #
    const ν_thresh = 0.5
    const C_thresh = 0.5
    # ===================================================================== #


    # Simulate initial transient
    integ = integrator(ds, alg=alg, maxiters=maxiters)
    while integ.t < Ttr
        step!(integ)
    end

    # Sample points
    # This will sample *approximately* `n_samples` points. It does so by
    # sampling the time to the next sample from an Exponential distribution with
    # mean λ set to the total time divided by the number of samples desired.
    samples = typeof(integ.u)[]
    λ = T/n_samples
    D = Exponential(λ)
    while integ.t < Ttr + T
        step!(integ, rand(D), true)
        push!(samples, integ.u)
    end

    # Calculate the mean position and variance of the trajectory as described on
    # pg. 5 of [1], using the samples generated rather than attempting
    # integration again
    μ = mean(samples)
    s² = mean(map(x->(x-μ)⋅(x-μ), samples))
    
    # Calculate maximal Lyapunov exponent using library function `lyapunov`
    # (i.e. Benettin's algorithm), and from this calculate Lyapunov prediction
    # time
    λmax = lyapunov(ds, Tly) # TODO options?
    Tλ = abs(scale_ly/λmax)
    # The Lyapunov prediction time is bounded by Tλmax, defined above (as
    # option?) The reason for this is that laminar flows have very small maximal
    # lyapunov exponents, which results in infeasibly long trajectories. In [1]
    # they just 'choose' 200 which works nicely, but their suggested Tλ =
    # 10/λmax doesn't seem to work in practice.
    Tλ = min(Tλ, Tλmax)

    # Calculate cross-distance scaling and correlation scaling
    ds = Float64[] # Mean distances at time T for different δ
    Cs = Float64[] # Cross-correlation at time T for different δ
    p_integ = parallel_integrator(lz, samples[1:2], alg=alg, maxiters=maxiters) #TODO options
    for δ in δ_range
        Σd = 0
        Σd² = 0
        for u in samples
            # Sample perturbation from surface of sphere
            n = rand(Normal(), size(u))
            n /= norm(n)
            û = u + δ*n
            # Update integrator with new initial conditions
            reinit!(p_integ, [u, û])
            # Simulate trajectory until Tλ
            while p_integ.t < Tλ
                step!(p_integ)
            end
            # Accumulate distance and square-distance
            d = norm(p_integ.u[1]-p_integ.u[2], 2)
            Σd  += d
            Σd² += d^2
        end
        # Calculate mean distance and square-distance
        d = Σd/length(samples)
        D = Σd²/length(samples)
        # Convert mean square-distance into cross-correlation
        C = 1 - D/2s²
        # Store mean distance and cross-correlation
        push!(ds, d)
        push!(Cs, C)
    end

    # Perform regression to check cross-distance scaling
    ν = slope(log.(δ_range), log.(ds))
    C = mean(Cs)
    
    # Determine chaotic nature of the system
    if ν > ν_thresh && C > C_thresh
        chaos_type = :LAM
    elseif ν <= ν_thresh && C > C_thresh
        chaos_type = :PPC
    elseif ν <= ν_thresh && C <= C_thresh
        chaos_type = :SC
    else
        # Covers the case when ν > ν_thresh but C <= C_thresh
        chaos_type = :INDETERMINATE
    end

    return chaos_type, ν, C
end
