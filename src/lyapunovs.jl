using StaticArrays, OrdinaryDiffEq
using OrdinaryDiffEq: ODEIntegrator
using DynamicalSystemsBase: MinimalDiscreteIntegrator

export lyapunovs, lyapunov

#####################################################################################
#                               Lyapunov Spectum                                    #
#####################################################################################
"""
    lyapunovs(ds::DynamicalSystem, N, k::Int | Q0; kwargs...) -> λs

Calculate the spectrum of Lyapunov exponents [1] of `ds` by applying
a QR-decomposition on the parallelepiped matrix `N` times. Return the
spectrum sorted from maximum to minimum.

The third argument `k` is optional, and dictates how many lyapunov exponents
to calculate (defaults to `dimension(ds)`).
Instead of passing an integer `k` you can pass
a pre-initialized matrix `Q0` whose columns are initial deviation vectors (then
`k = size(Q0)[2]`).

## Keyword Arguments
* `Ttr = 0` : Extra "transient" time to evolve the system before application of the
  algorithm. Should be `Int` for discrete systems. Both the system and the
  deviation vectors are evolved for this time.
* `dt` : Time of individual evolutions
  between successive orthonormalization steps. Defaults to `1`. For continuous
  systems this is approximate.
* `diff_eq_kwargs = Dict()` : (only for continuous)
  Keyword arguments passed into the solvers of the
  `DifferentialEquations` package (see [`trajectory`](@ref) for more info).

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
parameter values one should use the lower level methods that accept
an integrator, and `reinit!` it to new initial conditions.

See the "advanced documentation" for info on the integrator object
and use `@which ...` to go to the source code for the low-level
call signature.

## References

[1] : A. M. Lyapunov, *The General Problem of the Stability of Motion*,
Taylor & Francis (1992)

[2] : K. Geist *et al.*, Progr. Theor. Phys. **83**, pp 875 (1990)

[3] : G. Benettin *et al.*, Meccanica **15**, pp 9-20 & 21-30 (1980)
"""
lyapunovs(ds::DS, N, k::Int = dimension(ds); kwargs...) =
lyapunovs(ds, N, orthonormal(dimension(ds), k); kwargs...)

function lyapunovs(ds::DS{IIP, S, D}, N, Q0::AbstractMatrix; Ttr::Real = 0,
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS, dt::Real = 1) where {IIP, S, D}

    T = stateeltype(ds)
    # Create tangent integrator:
    if typeof(ds) <: DDS
        # Time assertions
        @assert typeof(Ttr) == Int
        tode = tangent_integrator(ds, Q0; t0 = inittime(ds)+Ttr)
    else
        tode = tangent_integrator(ds, Q0; diff_eq_kwargs = diff_eq_kwargs,
        t0 = inittime(ds)+Ttr)
    end
    k = size(Q0)[2]
    @assert k > 1

    # Choose algorithm
    λ::Vector{T} = if IIP
        # if k == D && D < 20
        #     _lyapunovs_iip(tode, N, dt, Ttr, k, DynamicalSystemsBase.qr_sq)
        # else
            _lyapunovs_iip(tode, N, dt, Ttr, k, Base.qr)::Vector{T}
        # end
    else
        _lyapunovs_oop(tode, N, dt, Ttr, Val{k}())
    end
    return λ
end

function _lyapunovs_iip(integ, N, dt::Real, Ttr::Real, k::Int, qrf::Function)

    T = stateeltype(integ)
    t0 = integ.t
    if Ttr > 0
        while integ.t < t0 + Ttr
            step!(integ, dt)
            Q, R = qrf(view(integ.u, :, 2:k+1))
            view(integ.u, :, 2:k+1) .= Q
            u_modified!(integ, true)
        end
    end

    λ::Vector{T} = zeros(T, k)
    t0 = integ.t

    for i in 2:N
        step!(integ, dt)
        Q, R = qrf(view(integ.u, :, 2:k+1))
        for i in 1:k
            λ[i] += log(abs(R[i,i]))
        end
        view(integ.u, :, 2:k+1) .= Q
        u_modified!(integ, true)
    end
    λ ./= (integ.t - t0)
    return λ
end

function _lyapunovs_oop(integ, N, dt::Real, Ttr::Real, ::Val{k}) where {k}

    T = stateeltype(integ)
    t0 = integ.t
    ws_idx = SVector{k, Int}(collect(2:k+1))
    D = size(state(integ))[1]
    O = D*k; T = eltype(state(integ))

    if Ttr > 0
        step!(integ, dt)
        Q, R = qr(integ.u[:, ws_idx])
        integ.u = hcat(integ.u[:,1], Q)
        u_modified!(integ, true)
    end

    λ::Vector{T} = zeros(T, k)
    t0 = integ.t

    for i in 2:N
        step!(integ, dt)
        Q, R = qr(integ.u[:, ws_idx])
        for i in 1:k
            λ[i] += log(abs(R[i,i]))
        end
        integ.u = hcat(integ.u[:,1], Q)
        u_modified!(integ, true)
    end
    λ ./= (integ.t - t0)
    return λ
end

#####################################################################################
#                           Maximum Lyapunov Exponent                               #
#####################################################################################
inittest_default(D) = (state1, d0) -> state1 + d0/sqrt(D)

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
* `diff_eq_kwargs = Dict(:abstol=>d0, :reltol=>d0)` : (only for continuous)
  Keyword arguments passed into the solvers of the
  `DifferentialEquations` package (see [`trajectory`](@ref) for more info).
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
\\lambda = \\frac{1}{t_{n}}\\sum_{i=1}^{n}
\\ln\\left( a_i \\right),\\quad a_i = \\frac{d(t_{i})}{d_0}.
```

## Performance Notes
This function uses a `parallel_integrator`. For loops over initial conditions and/or
parameter values one should use the lower level methods that accept
an integrator, and `reinit!` it to new initial conditions.

See the "advanced documentation" for info on the integrator object
and use `@which ...` to go to the source code for the low-level
call signature.

## References
[1] : G. Benettin *et al.*, Phys. Rev. A **14**, pp 2338 (1976)
"""
function lyapunov(ds::DS, T;
                  Ttr = 0,
                  d0=1e-9,
                  upper_threshold = 1e-6,
                  lower_threshold = 1e-12,
                  inittest = inittest_default(dimension(ds)),
                  diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
                  dt = 1
                  )

    ST = stateeltype(ds)
    lower_threshold ≤ d0 ≤ upper_threshold || throw(ArgumentError(
    "d0 must be between thresholds!"))
    D = dimension(ds)
    if typeof(ds) <: DDS
        pinteg = parallel_integrator(ds, [deepcopy(state(ds)), inittest(state(ds), d0)])
    else
        pinteg = parallel_integrator(ds, [deepcopy(state(ds)), inittest(state(ds), d0)];
        diff_eq_kwargs = diff_eq_kwargs)
    end
    λ::ST = _lyapunov(pinteg, T, Ttr, dt, d0, upper_threshold, lower_threshold)
    return λ
end

function _lyapunov(pinteg, T, Ttr, dt, d0, ut, lt)
    # transient
    t0 = pinteg.t
    while pinteg.t < t0 + Ttr
        step!(pinteg, dt)
        d = λdist(pinteg)
        lt ≤ d ≤ ut || ( rescale!(pinteg, d/d0); u_modified!(pinteg, true) )
    end

    t0 = pinteg.t
    d = λdist(pinteg)
    d == 0 && error("Initial distance between states is zero!!!")
    rescale!(pinteg, d/d0); u_modified!(pinteg, true)
    λ = zero(d)
    while pinteg.t < t0 + T
        d = λdist(pinteg)
        #evolve until rescaling:
        while lt ≤ d ≤ ut
            step!(pinteg, dt)
            d = λdist(pinteg)
            pinteg.t ≥ Ttr + T && break
        end
        # local lyapunov exponent is simply the relative distance of the trajectories
        a = d/d0
        λ += log(a)
        rescale!(pinteg, a); u_modified!(pinteg, true)
    end
    # Do final rescale, in case no other happened
    d = λdist(pinteg)
    a = d/d0
    λ += log(a)
    return λ/(pinteg.t - t0)
end

################ Helper functions that allow a single definition ######################
function λdist(integ::ODEIntegrator{Alg, M}) where {Alg, M<:Matrix}
    d = 0.0
    for i in 1:size(integ.u)[1]
        d += (integ.u[i, 1] - integ.u[i, 2])^2
    end
    return sqrt(d)
    # return norm(view(integ.u, :, 1) .- view(integ.u, :, 2))
end
# No-annotation case is with Vectors
function λdist(integ)
    return norm(integ.u[1] .- integ.u[2])
end
function λdist(integ::MinimalDiscreteIntegrator{true, Vector{S}}) where {S<:SVector}
    return norm(integ.u[1] - integ.u[2])
end
function λdist(integ::ODEIntegrator{Alg, Vector{S}}) where {Alg, S<:SVector}
    return norm(integ.u[1] - integ.u[2])
end

# Rescales:
function rescale!(integ::MinimalDiscreteIntegrator{true, Vector{S}}, a) where {S<:SVector}
    integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
end
function rescale!(integ::MinimalDiscreteIntegrator{true, Vector{S}}, a) where {S<:Vector}
    @. integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
end
function rescale!(integ::ODEIntegrator{Alg, M}, a) where {Alg, M<:Matrix}
    for i in 1:size(integ.u)[1]
        integ.u[i, 2] = integ.u[i,1] + (integ.u[i,2] - integ.u[i,1])/a
    end
end
function rescale!(integ::ODEIntegrator{Alg, Vector{S}}, a) where {Alg, S<:Vector}
    @. integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
end
function rescale!(integ::ODEIntegrator{Alg, Vector{S}}, a) where {Alg, S<:SVector}
    integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
end
