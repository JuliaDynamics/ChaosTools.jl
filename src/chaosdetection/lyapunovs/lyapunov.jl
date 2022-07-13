#####################################################################################
# Maximum Lyapunov Exponent
#####################################################################################
"""
    lyapunov(ds::DynamicalSystem, Τ; kwargs...) -> λ

Calculate the maximum Lyapunov exponent `λ` using a method due to Benettin [^Benettin1976],
which simply
evolves two neighboring trajectories (one called "given" and one called "test")
while constantly rescaling the test one.
`T`  denotes the total time of evolution (should be `Int` for discrete systems).

See also [`lyapunovspectrum`](@ref), [`local_growth_rates`](@ref).

## Keyword Arguments
* `u0 = get_state(ds)` : Initial condition.
* `Ttr = 0` : Extra "transient" time to evolve the trajectories before
  starting to measure the expontent. Should be `Int` for discrete systems.
* `d0 = 1e-9` : Initial & rescaling distance between the two neighboring trajectories.
* `upper_threshold = 1e-6` : Upper distance threshold for rescaling.
* `lower_threshold = 1e-12` : Lower distance threshold for rescaling (in order to
   be able to detect negative exponents).
* `Δt = 1` : Time of evolution between each check of
  distance exceeding the thresholds. For continuous
  systems this is approximate.
* `inittest = (u1, d0) -> u1 .+ d0/sqrt(D)` :
  A function that given `(u1, d0)`
  initializes the test state with distance
  `d0` from the given state `u1` (`D` is the dimension
  of the system). This function can be used when you want to avoid
  the test state appearing in a region of the phase-space where it would have
  e.g. different energy or escape to infinity.
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.


## Description
Two neighboring trajectories with initial distance `d0` are evolved in time.
At time ``t_i`` their distance ``d(t_i)`` either exceeds the `upper_threshold`,
or is lower than `lower_threshold`, which initializes
a rescaling of the test trajectory back to having distance `d0` from
the given one, while the rescaling keeps the difference vector along the maximal
expansion/contraction direction: `` u_2 \\to u_1+(u_2−u_1)/(d(t_i)/d_0)``.

The maximum
Lyapunov exponent is the average of the time-local Lyapunov exponents
```math
\\lambda = \\frac{1}{t_{n} - t_0}\\sum_{i=1}^{n}
\\ln\\left( a_i \\right),\\quad a_i = \\frac{d(t_{i})}{d_0}.
```

## Performance Notes
This function uses a [`parallel_integrator`](@ref).
For loops over initial conditions and/or
parameter values one should use the low level method that accepts
an integrator, and `reinit!` it to new initial conditions.
See the "advanced documentation" for info on the integrator object.
The low level method is
```
lyapunov(pinteg, T, Ttr, Δt, d0, ut, lt)
```

[^Benettin1976]: G. Benettin *et al.*, Phys. Rev. A **14**, pp 2338 (1976)
"""
function lyapunov(ds::DS, T;
        u0 = get_state(ds),
        Ttr = 0,
        d0=1e-9,
        upper_threshold = 1e-6,
        lower_threshold = 1e-12,
        inittest = inittest_default(dimension(ds)),
        Δt = 1,
        diffeq = NamedTuple(), kwargs...
    )

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

    ST = stateeltype(ds)
    lower_threshold ≤ d0 ≤ upper_threshold || throw(ArgumentError(
    "d0 must be between thresholds!"))
    D = dimension(ds)
    if typeof(ds) <: DDS
        pinteg = parallel_integrator(ds, [deepcopy(u0), inittest(u0, d0)])
    else
        pinteg = parallel_integrator(ds, [deepcopy(u0), inittest(u0, d0)]; diffeq)
    end
    λ::ST = lyapunov(pinteg, T, Ttr, Δt, d0, upper_threshold, lower_threshold)
    return λ
end

inittest_default(D) = (state1, d0) -> state1 .+ d0/sqrt(D)

function lyapunov(pinteg, T, Ttr, Δt, d0, ut, lt)
    # transient
    t0 = pinteg.t
    while pinteg.t < t0 + Ttr
        step!(pinteg, Δt)
		successful_step(pinteg) || return NaN
        d = λdist(pinteg)
        lt ≤ d ≤ ut || rescale!(pinteg, d/d0)
    end

    t0 = pinteg.t
    d = λdist(pinteg)
    d == 0 && error("Initial distance between states is zero!!!")
    rescale!(pinteg, d/d0)
    λ = zero(d)
    while pinteg.t < t0 + T
        d = λdist(pinteg)
        # evolve until rescaling
        while lt ≤ d ≤ ut
            step!(pinteg, Δt)
			successful_step(pinteg) || return NaN
            d = λdist(pinteg)
            pinteg.t ≥ t0 + T && break
        end
        # local lyapunov exponent is the relative distance of the trajectories
        a = d/d0
        λ += log(a)
        rescale!(pinteg, a)
    end
    # Do final rescale, in case no other happened
    d = λdist(pinteg)
    a = d/d0
    λ += log(a)
    return λ/(pinteg.t - t0)
end

lyapunov(ds::DDS{false, T, 1}, N; Ttr = 0) where {T} = lyapunovspectrum(ds, N; Ttr = Ttr)

#####################################################################################
# Helper functions that allow a single definition
#####################################################################################
function λdist(integ::AbstractODEIntegrator{Alg, IIP, M}) where {Alg, IIP, M<:Matrix}
    d = 0.0
    for i in 1:size(integ.u)[1]
        d += (integ.u[i, 1] - integ.u[i, 2])^2
    end
    return sqrt(d)
    # return norm(view(integ.u, :, 1) .- view(integ.u, :, 2))
end
# No-annotation case is with Vectors
function λdist(integ)
    @inbounds s = zero(eltype(integ.u[1]))
    @inbounds for k in 1:length(integ.u[1])
        x = (integ.u[1][k] - integ.u[2][k])
        s += x*x
    end
    return sqrt(s)
end
function λdist(integ::MinimalDiscreteIntegrator{true, Vector{S}}) where {S<:SVector}
    return norm(integ.u[1] - integ.u[2])
end
function λdist(integ::AbstractODEIntegrator{Alg, IIP, Vector{S}}) where {Alg, IIP, S<:SVector}
    return norm(integ.u[1] - integ.u[2])
end

# Rescales:
function rescale!(integ::MinimalDiscreteIntegrator{true, Vector{S}}, a) where {S<:SVector}
    integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
    u_modified!(integ, true)
end
function rescale!(integ::MinimalDiscreteIntegrator{true, Vector{S}}, a) where {S<:Vector}
    @. integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
    u_modified!(integ, true)
end
function rescale!(integ::AbstractODEIntegrator{Alg, IIP, M}, a) where {Alg, IIP, M<:Matrix}
    for i in 1:size(integ.u)[1]
        integ.u[i, 2] = integ.u[i,1] + (integ.u[i,2] - integ.u[i,1])/a
    end
    u_modified!(integ, true)
end
function rescale!(integ::AbstractODEIntegrator{Alg, IIP, Vector{S}}, a) where {Alg, IIP, S<:Vector}
    @. integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
    u_modified!(integ, true)
end
function rescale!(integ::AbstractODEIntegrator{Alg, IIP, Vector{S}}, a) where {Alg, IIP, S<:SVector}
    integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
    u_modified!(integ, true)
end
