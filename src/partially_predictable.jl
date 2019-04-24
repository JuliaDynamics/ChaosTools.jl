using LinearAlgebra, Distributions
using Statistics: mean
export predictability

# TODO - write docstring
#
# [1] H. Wernecke, B. Sándor, and C. Gros, ‘How to test for partially
#     predictable chaos’, Scientific Reports, vol. 7, no. 1, Dec. 2017.

function predictability(ds::DynamicalSystem;
                        T_transient::Real = 200,
                        T_sample::Real = 1e5,
                        n_samples::Integer = 1000,
                        λ_max::Real = abs(lyapunov(ds, 5000)),
                        d_tol::Real = 1e-3,
                        T_multiplier::Real = 10,
                        T_max::Real = Inf,
                        δ_range::AbstractArray{T,1} = 10.0 .^ (-9:-6),
                        diffeq...
                       ) where T <: Real

    # ======================== Internal Constants ========================= #
    ν_threshold = 0.5
    C_threshold = 0.5
    # ===================================================================== #


    # Simulate initial transient
    integ = integrator(ds; diffeq...)
    while integ.t < T_transient
        step!(integ)
    end

    # Samples *approximately* `n_samples` points.
    # Time to the next sample is sampled from Exponential distribution with
    # mean λ set to the total time divided by the number of samples desired.
    samples = typeof(integ.u)[]
    λ = T_sample/n_samples
    D = Exponential(λ)
    while integ.t < T_transient + T_sample
        step!(integ, rand(D), true)
        push!(samples, integ.u)
    end

    # Calculate the mean position and variance of the trajectory. ([1] pg. 5)
    # Using samples 'Monte Carlo' approach instead of direct integration
    μ = mean(samples)
    s² = mean(map(x->(x-μ)⋅(x-μ), samples))

    # Calculate cross-distance scaling and correlation scaling
    ds = Float64[] # Mean distances at time T for different δ
    Cs = Float64[] # Cross-correlation at time T for different δ
    p_integ = parallel_integrator(lz, samples[1:2]; diffeq...)
    for δ in δ_range
        Tλ = log(d_tol/δ)/λ_max
        T = min(T_multiplier * Tλ, T_max)
        Σd = 0
        Σd² = 0
        for u in samples
            # Sample perturbation from surface of sphere
            n = rand(Normal(), size(u))
            n /= norm(n)
            û = u + δ*n
            # Update integrator with new initial conditions
            reinit!(p_integ, [u, û])
            # Simulate trajectory until T
            while p_integ.t < T
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
    if ν > ν_threshold && C > C_threshold
        chaos_type = :LAM
    elseif ν <= ν_threshold && C > C_threshold
        chaos_type = :PPC
    elseif ν <= ν_threshold && C <= C_threshold
        chaos_type = :SC
    else
        # Covers the case when ν > ν_threshold but C <= C_threshold
        chaos_type = :INDETERMINATE
    end

    return chaos_type, ν, C
end
