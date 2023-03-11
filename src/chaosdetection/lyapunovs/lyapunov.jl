export lyapunov

"""
    lyapunov(ds::DynamicalSystem, Τ; kwargs...) -> λ

Calculate the maximum Lyapunov exponent `λ` using a method due to Benettin [^Benettin1976],
which simply evolves two neighboring trajectories (one called "given" and one called "test")
while constantly rescaling the test one.

`T`  denotes the total time of evolution (should be `Int` for discrete time systems).

See also [`lyapunovspectrum`](@ref), [`local_growth_rates`](@ref).

## Keyword arguments

* `show_progress = false`: Display a progress bar of the process.
* `u0 = initial_state(ds)`: Initial condition.
* `Ttr = 0`: Extra "transient" time to evolve the trajectories before
  starting to measure the exponent. Should be `Int` for discrete systems.
* `d0 = 1e-9`: Initial & rescaling distance between the two neighboring trajectories.
* `d0_lower = 1e-3*d0`: Lower distance threshold for rescaling.
* `d0_upper = 1e+3*d0`: Upper distance threshold for rescaling.
* `Δt = 1`: Time of evolution between each check rescaling of distance.
  For continuous time systems this is approximate.
* `inittest = (u1, d0) -> u1 .+ d0/sqrt(length(u1))`: A function that given `(u1, d0)`
  initializes the test state with distance `d0` from the given state `u1`
   (`D` is the dimension of the system). This function can be used when you want to avoid
  the test state appearing in a region of the phase-space where it would have
  e.g. different energy or escape to infinity.

## Description

Two neighboring trajectories with initial distance `d0` are evolved in time.
At time ``t_i`` if their distance ``d(t_i)`` either exceeds the `d0_upper`,
or is lower than `d0_lower`, the test trajectory is rescaled back to having distance
`d0` from the reference one, while the rescaling keeps the difference vector along the maximal
expansion/contraction direction: `` u_2 \\to u_1+(u_2−u_1)/(d(t_i)/d_0)``.

The maximum Lyapunov exponent is the average of the time-local Lyapunov exponents
```math
\\lambda = \\frac{1}{t_{n} - t_0}\\sum_{i=1}^{n}
\\ln\\left( a_i \\right),\\quad a_i = \\frac{d(t_{i})}{d_0}.
```

## Performance notes

This function simply initializes a [`ParallelDynamicalSystem`](@ref) and calls
the method below.

[^Benettin1976]: G. Benettin *et al.*, Phys. Rev. A **14**, pp 2338 (1976)
"""
function lyapunov(ds::DynamicalSystem, T;
        u0 = initial_state(ds),
        d0 = 1e-9,
        inittest = inittest_default(dimension(ds)),
        kwargs...
    )
    # initialize parallel
    states = [u0, inittest(u0, d0)]
    pds = ParallelDynamicalSystem(ds, states)
    return lyapunov(pds, T; d0, kwargs...)
end

inittest_default(D) = (state1, d0) -> state1 .+ d0/sqrt(D)

"""
    lyapunov(pds::ParallelDynamicalSystem, T; Ttr, Δt, d0, d0_upper, d0_lower)

The low-level method that is called by `lyapunov(ds::DynamicalSystem, ...)`.
Use this method for looping over different initial conditions or parameters by
calling [`reinit!`](@ref) to `pds`.
"""
function lyapunov(pds::ParallelDynamicalSystem, T;
        Ttr = 0, Δt = 1, d0 = λdist(pds), d0_upper = d0*1e+3, d0_lower = d0*1e-3,
        show_progress = false,
    )
    progress = ProgressMeter.Progress(round(Int, T);
        desc = "Lyapunov exponent: ", dt = 1.0, enabled = show_progress
    )
    # transient
    while current_time(pds) - initial_time(pds) < Ttr
        step!(pds, Δt)
        d = λdist(pds)
        # We do the rescaling to orient the difference vector
        d0_lower ≤ d ≤ d0_upper || λrescale!(pds, d/d0)
    end
    # Set up algorithm
    t0 = current_time(pds)
    d = λdist(pds)
    d == 0 && error("Initial distance between states is zero!!!")
    d != d0 && λrescale!(pds, d/d0)
    λ = zero(d)
    # Perform algorithm
    while current_time(pds) < t0 + T
        d = λdist(pds)
        if !(d0_lower ≤ d ≤ d0_upper)
            error(
                "After rescaling, the distance of reference and test states "*
                "was not `d0_lower ≤ d ≤ d0_upper` as expected. "*
                "Perhaps you are using a dynamical system where the algorithm doesn't work."
            )
        end
        # evolve until rescaling
        while d0_lower ≤ d ≤ d0_upper
            step!(pds, Δt)
            d = λdist(pds)
            current_time(pds) ≥ t0 + T && break
        end
        # local lyapunov exponent is the relative distance of the trajectories
        a = d/d0
        λ += log(a)
        λrescale!(pds, a)
        ProgressMeter.update!(progress, round(Int, current_time(pds)))
    end
    # Do final rescale, in case no other happened
    d = λdist(pds)
    a = d/d0
    λ += log(a)
    return λ/(current_time(pds) - t0)
end

function λdist(ds::ParallelDynamicalSystem)
    u1 = current_state(ds, 1)
    u2 = current_state(ds, 2)
    # Compute euclidean dinstace in a loop (don't care about static or not)
    d = zero(eltype(u1))
    @inbounds for i in eachindex(u1)
        d += (u1[i] - u2[i])^2
    end
    return sqrt(d)
end

# TODO: Would be nice to generalize this so that it can accept a user-defined function
function λrescale!(pds::ParallelDynamicalSystem, a)
    u1 = current_state(pds, 1)
    u2 = current_state(pds, 2)
    if ismutable(u2) # if mutable we assume `Array`
        @. u2 = u1 + (u2 - u1)/a
    else # if not mutable we assume `SVector`
        u2 = @. u1 + (u2 - u1)/a
    end
    set_state!(pds, u2, 2)
end
