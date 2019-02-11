using LinearAlgebra, StaticArrays
using DynamicalSystemsBase: MinimalDiscreteIntegrator

export lyapunovs, lyapunov

#####################################################################################
#                               Lyapunov Spectum                                    #
#####################################################################################
"""
    lyapunovs(ds::DynamicalSystem, N [, k::Int | Q0]; kwargs...) -> λs

Calculate the spectrum of Lyapunov exponents [1] of `ds` by applying
a QR-decomposition on the parallelepiped matrix `N` times. Return the
spectrum sorted from maximum to minimum.

The third argument `k` is optional, and dictates how many lyapunov exponents
to calculate (defaults to `dimension(ds)`).
Instead of passing an integer `k` you can pass
a pre-initialized matrix `Q0` whose columns are initial deviation vectors (then
`k = size(Q0)[2]`).

## Keyword Arguments
* `u0 = get_state(ds)` : State to start from.
* `Ttr = 0` : Extra "transient" time to evolve the system before application of the
  algorithm. Should be `Int` for discrete systems. Both the system and the
  deviation vectors are evolved for this time.
* `dt = 1` : Time of individual evolutions
  between successive orthonormalization steps. For continuous systems this is approximate.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.

## Description
The method we employ is "H2" of [2], originally stated in [3]. The deviation vectors
defining a `D`-dimensional parallepiped in tangent space
are evolved using the tangent dynamics of the system.
A QR-decomposition at each step yields the local growth rate for each dimension
of the parallepiped. The growth rates are
then averaged over `N` successive steps, yielding the lyapunov exponent spectrum
(at each step the parallepiped is re-normalized).

## Performance Notes
This function uses a [`tangent_integrator`](@ref).
For loops over initial conditions and/or
parameter values one should use the low level method that accepts
an integrator, and `reinit!` it to new initial conditions.
See the "advanced documentation" for info on the integrator object.
The low level method is
```julia
lyapunovs(tinteg, N, dt::Real, Ttr::Real)
```

If you want to obtain the convergence timeseries of the Lyapunov spectrum,
use the method
```julia
ChaosTools.lyapunovs_convergence(tinteg, N, dt, Ttr)
```
(not exported).

## References

[1] : A. M. Lyapunov, *The General Problem of the Stability of Motion*,
Taylor & Francis (1992)

[2] : K. Geist *et al.*, Progr. Theor. Phys. **83**, pp 875 (1990)

[3] : G. Benettin *et al.*, Meccanica **15**, pp 9-20 & 21-30 (1980)
"""
lyapunovs(ds::DS, N, k::Int = dimension(ds); kwargs...) =
lyapunovs(ds, N, orthonormal(dimension(ds), k); kwargs...)

function lyapunovs(ds::DS{IIP, S, D}, N, Q0::AbstractMatrix; Ttr::Real = 0,
    dt::Real = 1, u0 = get_state(ds), diffeq...) where {IIP, S, D}

    T = stateeltype(ds)
    # Create tangent integrator:
    if typeof(ds) <: DDS
        @assert typeof(Ttr) == Int
        integ = tangent_integrator(ds, Q0; u0 = u0)
    else
        integ = tangent_integrator(ds, Q0; u0 = u0, diffeq...)
    end

    λ::Vector{T} = lyapunovs(integ, N, dt, Ttr)
    return λ
end

function lyapunovs(integ, N, dt::Real, Ttr::Real = 0.0)

    T = stateeltype(integ)
    t0 = integ.t
    if Ttr > 0
        while integ.t < t0 + Ttr
            step!(integ, dt)
            qrdec = LinearAlgebra.qr(get_deviations(integ))
            set_deviations!(integ, _get_Q(qrdec))
        end
    end
    k = size(get_deviations(integ))[2]
    λ::Vector{T} = zeros(T, k)
    t0 = integ.t

    for i in 2:N
        step!(integ, dt)
        qrdec = LinearAlgebra.qr(get_deviations(integ))
        for j in 1:k
            @inbounds λ[j] += log(abs(qrdec.R[j,j]))
        end
        set_deviations!(integ, _get_Q(qrdec))
    end
    λ ./= (integ.t - t0)
    return λ
end

_get_Q(qrdec::StaticArrays.QR) = qrdec.Q
_get_Q(qrdec::LinearAlgebra.QRCompactWY) = Matrix(qrdec.Q)

lyapunovs(ds::DynamicalSystem{IIP, T, 1}, a...; kw...) where {IIP, T} = error(
"For 1D systems, only discrete & out-of-place method is implemented.")

function lyapunovs(ds::DDS{false, T, 1}, N; Ttr = 0) where {T}

    x = get_state(ds); f = ds.f
    p = ds.p; t0 = ds.t0
    t = 0
    if Ttr > 0
        for i in t0:(Ttr+t0)
            x = f(x, p, i)
        end
    end
    λ = zero(T)
    for i in (t0+Ttr):(t0+Ttr+N)
        x = f(x, p, i)
        λ += log(abs(ds.jacobian(x, p, i)))
    end
    return λ/N
end

lyapunov(ds::DDS{false, T, 1}, N; Ttr = 0) where {T} = lyapunovs(ds, N; Ttr = Ttr)

#####################################################################################
#                           Maximum Lyapunov Exponent                               #
#####################################################################################
inittest_default(D) = (state1, d0) -> state1 .+ d0/sqrt(D)

"""
    lyapunov(ds::DynamicalSystem, Τ; kwargs...) -> λ

Calculate the maximum Lyapunov exponent `λ` using a method due to Benettin [1],
which simply
evolves two neighboring trajectories (one called "given" and one called "test")
while constantly rescaling the test one.
`T`  denotes the total time of evolution (should be `Int` for discrete systems).

## Keyword Arguments

* `Ttr = 0` : Extra "transient" time to evolve the trajectories before
  starting to measure the expontent. Should be `Int` for discrete systems.
* `d0 = 1e-9` : Initial & rescaling distance between the two neighboring trajectories.
* `upper_threshold = 1e-6` : Upper distance threshold for rescaling.
* `lower_threshold = 1e-12` : Lower distance threshold for rescaling (in order to
   be able to detect negative exponents).
* `dt = 1` : Time of evolution between each check of
  distance exceeding the thresholds. For continuous
  systems this is approximate.
* `inittest = (u1, d0) -> u1 .+ d0/sqrt(D)` :
  A function that given `(u1, d0)`
  initializes the test state with distance
  `d0` from the given state `u1` (`D` is the dimension
  of the system). This function can be used when you want to avoid
  the test state appearing in a region of the phase-space where it would have
  e.g. different energy or escape to infinity.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
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
lyapunov(pinteg, T, Ttr, dt, d0, ut, lt)
```

## References
[1] : G. Benettin *et al.*, Phys. Rev. A **14**, pp 2338 (1976)
"""
function lyapunov(ds::DS, T;
                  Ttr = 0,
                  d0=1e-9,
                  upper_threshold = 1e-6,
                  lower_threshold = 1e-12,
                  inittest = inittest_default(dimension(ds)),
                  dt = 1,
                  diffeq...
                  )

    ST = stateeltype(ds)
    lower_threshold ≤ d0 ≤ upper_threshold || throw(ArgumentError(
    "d0 must be between thresholds!"))
    D = dimension(ds)
    if typeof(ds) <: DDS
        pinteg = parallel_integrator(ds,
            [deepcopy(get_state(ds)), inittest(get_state(ds), d0)])
    else
        pinteg = parallel_integrator(ds,
            [deepcopy(get_state(ds)), inittest(get_state(ds), d0)];
            diffeq...)
    end
    λ::ST = lyapunov(pinteg, T, Ttr, dt, d0, upper_threshold, lower_threshold)
    return λ
end

function lyapunov(pinteg, T, Ttr, dt, d0, ut, lt)
    # transient
    t0 = pinteg.t
    while pinteg.t < t0 + Ttr
        step!(pinteg, dt)
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
        #evolve until rescaling:
        while lt ≤ d ≤ ut
            step!(pinteg, dt)
            d = λdist(pinteg)
            pinteg.t ≥ t0 + T && break
        end
        # local lyapunov exponent is simply the relative distance of the trajectories
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

################ Helper functions that allow a single definition ######################
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
