using LinearAlgebra, StaticArrays
import ProgressMeter
using DynamicalSystemsBase: MinimalDiscreteIntegrator

export lyapunovspectrum, lyapunov, local_growth_rates

#####################################################################################
#                               Lyapunov Spectum                                    #
#####################################################################################
"""
    lyapunovspectrum(ds::DynamicalSystem, N [, k::Int | Q0]; kwargs...) -> λs

Calculate the spectrum of Lyapunov exponents [^Lyapunov1992] of `ds` by applying
a QR-decomposition on the parallelepiped matrix `N` times. Return the
spectrum sorted from maximum to minimum.

The third argument `k` is optional, and dictates how many lyapunov exponents
to calculate (defaults to `dimension(ds)`).
Instead of passing an integer `k` you can pass
a pre-initialized matrix `Q0` whose columns are initial deviation vectors (then
`k = size(Q0)[2]`).

See also [`lyapunov`](@ref), [`local_growth_rates`](@ref).

## Keyword Arguments
* `u0 = get_state(ds)` : State to start from.
* `Ttr = 0` : Extra "transient" time to evolve the system before application of the
  algorithm. Should be `Int` for discrete systems. Both the system and the
  deviation vectors are evolved for this time.
* `Δt = 1` : Time of individual evolutions
  between successive orthonormalization steps. For continuous systems this is approximate.
* `show_progress = false` : Display a progress bar of the process.
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.

## Description
The method we employ is "H2" of [^Geist1990], originally stated in [^Benettin1980]. The deviation vectors
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
lyapunovspectrum(tinteg, N, Δt::Real, Ttr::Real)
```

If you want to obtain the convergence timeseries of the Lyapunov spectrum,
use the method
```julia
ChaosTools.lyapunovspectrum_convergence(tinteg, N, Δt, Ttr)
```
(not exported).

[^Lyapunov1992]: A. M. Lyapunov, *The General Problem of the Stability of Motion*, Taylor & Francis (1992)

[^Geist1990]: K. Geist *et al.*, Progr. Theor. Phys. **83**, pp 875 (1990)

[^Benettin1980]: G. Benettin *et al.*, Meccanica **15**, pp 9-20 & 21-30 (1980)
"""
lyapunovspectrum(ds::DS, N, k::Int = dimension(ds); kwargs...) =
lyapunovspectrum(ds, N, orthonormal(dimension(ds), k); kwargs...)

function lyapunovspectrum(ds::DS{IIP, S, D}, N, Q0::AbstractMatrix; 
        Ttr::Real = 0, Δt::Real = 1, u0 = get_state(ds), show_progress = false, 
        diffeq = NamedTuple(), kwargs...
    ) where {IIP, S, D}

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

    if typeof(ds) <: DDS
        @assert typeof(Ttr) == Int
        integ = tangent_integrator(ds, Q0; u0)
    else
        integ = tangent_integrator(ds, Q0; u0, diffeq)
    end
    λ = lyapunovspectrum(integ, N, Δt, Ttr, show_progress)
    return λ
end

function lyapunovspectrum(integ, N, Δt::Real, Ttr::Real = 0.0, show_progress = false)
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Lyapunov Spectrum: ", dt = 1.0)
    end
    B = copy(get_deviations(integ)) # for use in buffer
    if Ttr > 0
        t0 = integ.t
        while integ.t < t0 + Ttr
            step!(integ, Δt)
            Q, R = _buffered_qr(B, get_deviations(integ))
            set_deviations!(integ, Q)
        end
    end

    k = size(get_deviations(integ))[2]
    λ = zeros(stateeltype(integ), k)
    t0 = integ.t
    for i in 1:N
        step!(integ, Δt)
        Q, R = _buffered_qr(B, get_deviations(integ))
        for j in 1:k
            @inbounds λ[j] += log(abs(R[j,j]))
        end
        set_deviations!(integ, Q)
        show_progress && ProgressMeter.update!(progress, i)
    end
    λ ./= (integ.t - t0)
    return λ
end

# For out-of-place systems, this is just standard QR decomposition.
# For in-place systems, this is a more performant buffered version.
function _buffered_qr(B::SMatrix, Y) # Y are the deviations
    Q, R = LinearAlgebra.qr(Y)
    return Q, R
end
function _buffered_qr(B::Matrix, Y) # Y are the deviations
    B .= Y
    Q, R = LinearAlgebra.qr!(B)
    return Q, R
end


lyapunovspectrum(ds::DiscreteDynamicalSystem{IIP, T, 1}, a...; kw...) where {IIP, T} =
error("For discrete 1D systems, only method with state type = number is implemented.")

function lyapunovspectrum(ds::DDS{false, T, 1}, N; Ttr = 0) where {T}
    x = get_state(ds); f = ds.f
    p = ds.p; t0 = ds.t0
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

#####################################################################################
#                           Lyapunov Spectrum Convergence                           #
#####################################################################################
"""
    lyapunovspectrum_convergence(ds::DynamicalSystem, N [, k::Int | Q0]; kwargs...) -> λs, t

Computes the convergence of the Lyapunov exponents, calculated exactly as in `lyapunovspectrum`.
(See [`lyapunovspectrum`](@ref) for information on the parameters and implementation).
Returns `λs`, a vector of `N` vectors, wherein vector of index `i` contains the Lyapunov
spectrum evaluated up to the time `t[i]`. The final vector is equal to the returned vector
of `lyapunovspectrum`.

## Performance Notes
This function uses a [`tangent_integrator`](@ref).
For loops over initial conditions and/or
parameter values one should use the low level method that accepts
an integrator, and `reinit!` it to new initial conditions.
See the "advanced documentation" for info on the integrator object.
The low level method is
```julia
lyapunovspectrum_convergence(tinteg, N, Δt::Real, Ttr::Real)
```
"""
lyapunovspectrum_convergence(ds::DS, N, k::Int = dimension(ds); kwargs...) =
lyapunovspectrum_convergence(ds, N, orthonormal(dimension(ds), k); kwargs...)

function lyapunovspectrum_convergence(ds::DS{IIP, S, D}, N, Q0::AbstractMatrix;
        Ttr::Real = 0, Δt::Real = 1, u0 = get_state(ds), show_progress = false, diffeq...
    ) where {IIP, S, D}

    if typeof(ds) <: DDS
        @assert typeof(Ttr) == Int
        integ = tangent_integrator(ds, Q0; u0)
    else
        integ = tangent_integrator(ds, Q0; u0, diffeq...)
    end
    λ = lyapunovspectrum_convergence(integ, N, Δt, Ttr, show_progress)
    return λ
end

function lyapunovspectrum_convergence(integ, N, Δt::Real, Ttr::Real = 0.0, show_progress=false)
    if show_progress
        progress = ProgressMeter.Progress(N; desc = "Lyapunov Spectrum: ", dt = 1.0)
    end
    B = copy(get_deviations(integ)) # for use in buffer
    if Ttr > 0
        t0 = integ.t
        while integ.t < t0 + Ttr
            step!(integ, Δt)
            Q, R = _buffered_qr(B, get_deviations(integ))
            set_deviations!(integ, Q)
        end
    end

    k = size(get_deviations(integ))[2]
    T = stateeltype(integ)
    t0 = integ.t; t = zeros(T, N); t[1] = t0
    λs = [zeros(T, k) for i in 1:N];

    for i in 2:N
        step!(integ, Δt)
        Q, R = _buffered_qr(B, get_deviations(integ))
        for j in 1:k
            @inbounds λs[i][j] = λs[i-1][j] + log(abs(R[j,j]))
        end
        t[i] = integ.t
        set_deviations!(integ, Q)
        show_progress && ProgressMeter.update!(progress, i)
    end
    popfirst!(λs); popfirst!(t)
    for j in eachindex(t)
        λs[j] ./= (t[j] - t0)
    end
    return λs, t
end

lyapunovspectrum_convergence(ds::DiscreteDynamicalSystem{IIP, T, 1}, a...; kw...) where {IIP, T} =
error("For discrete 1D systems, only method with state type = number is implemented.")

function lyapunovspectrum_convergence(ds::DDS{false, T, 1}, N; Ttr = 0) where {T}
    x = get_state(ds); f = ds.f
    p = ds.p; t0 = ds.t0
    if Ttr > 0
        for i in t0:(Ttr+t0)
            x = f(x, p, i)
        end
    end
    t = (t0+Ttr):(t0+Ttr+N)
    λs = zeros(T, length(t));
    for idx = 2:length(t)
        i = t[idx]
        x = f(x, p, i)
        @inbounds λs[idx] = λs[idx-1] + log(abs(ds.jacobian(x, p, i)))
    end
    t = collect(t)
    popfirst!(λs); popfirst!(t)
    λs ./= t .- t0
    return λs, t
end

#####################################################################################
#                           Maximum Lyapunov Exponent                               #
#####################################################################################
inittest_default(D) = (state1, d0) -> state1 .+ d0/sqrt(D)

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

function lyapunov(pinteg, T, Ttr, Δt, d0, ut, lt)
    # transient
    t0 = pinteg.t
    while pinteg.t < t0 + Ttr
        step!(pinteg, Δt)
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
            step!(pinteg, Δt)
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

lyapunov(ds::DDS{false, T, 1}, N; Ttr = 0) where {T} = lyapunovspectrum(ds, N; Ttr = Ttr)

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



#####################################################################################
#                              Local Growth Rates                                   #
#####################################################################################
"""
    local_growth_rates(ds, points::Dataset; S=100, Δt=5, kwargs...) → λlocal
Compute the exponential local growth rate(s) of perturbations of the dynamical system
`ds` for initial conditions given in `points`. For each initial condition `u ∈ points`,
`S` total perturbations are created and evolved for time `Δt`. The exponential local growth
rate is defined simply by `log(g/g0)/Δt` with `g0` the initial pertrubation size
and `g` the size after `Δt`. Thus, `λlocal` is a matrix of size `(length(points), S)`.

This function is a modification of [`lyapunov`](@ref). It uses the full nonlinear dynamics
to evolve the perturbations, but does not do any re-scaling, thus allowing
probing state and time dependence of perturbation growth. The actual growth
is given by `exp(λlocal * Δt)`.

The output of this function is sometimes referred as "Nonlinear Local Lyapunov Exponent".

## Keyword Arguments
* `perturbation`: If given, it should be a function `perturbation(ds, u, j)` that
  outputs a pertrubation vector (preferrably `SVector`) given the system, current initial
  condition `u` and the counter `j ∈ 1:S`. If not given, a random perturbation is
  generated with norm given by the keyword `e = 1e-6`.
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.
"""
function local_growth_rates(ds::DynamicalSystem, points;
        S = 100, Δt = 5, e = 1e-6,
        perturbation = (ds, u, j) -> _random_Q0(ds, u, j, e),
        diffeq = NamedTuple(), kwargs...
    )

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

    λlocal = zeros(length(points), S)
    Q0 = perturbation(ds, points[1], 1)
    states = [points[1], points[1] .+ Q0]
    pinteg = parallel_integrator(ds, states; diffeq)

    for (i, u) in enumerate(points)
        for j in 1:S
            Q0 = perturbation(ds, u, j)
            states[1] = u
            states[2] = states[1] .+ Q0
            g0 = norm(Q0)
            reinit!(pinteg, states)
            step!(pinteg, Δt, true)
            g = norm(get_state(pinteg, 2) .- get_state(pinteg, 1))
            λlocal[i, j] = log(g/g0) / (pinteg.t - pinteg.t0)
        end
    end
    return λlocal
end

function _random_Q0(ds, u, j, e)
    D, T = dimension(ds), eltype(ds)
    Q0 = randn(Random.GLOBAL_RNG, SVector{D, T})
    Q0 = e * Q0 / norm(Q0)
end

#####################################################################################
#                    Numerical Lyapunov (from reconstruction)                       #
#####################################################################################
using Neighborhood, StaticArrays
using Distances: Metric, Cityblock, Euclidean
export NeighborNumber, WithinRange
export lyapunov_from_data
export Cityblock, Euclidean
# Everything in this section is based on Ulrich Parlitz [1]

"""
    lyapunov_from_data(R::Dataset, ks;  refstates, w, distance, ntype)
Return `E = [E(k) for k ∈ ks]`, where `E(k)` is the average logarithmic distance
between states of a neighborhood that are evolved in time for `k` steps
(`k` must be integer). The slope of `E` vs `k` approximate the maximum Lyapunov exponent,
see below.
Typically `R` is the result of delay coordinates of a single timeseries.

## Keyword Arguments

* `refstates = 1:(length(R) - ks[end])` : Vector of indices
  that notes which
  states of the reconstruction should be used as "reference states", which means
  that the algorithm is applied for all state indices contained in `refstates`.
* `w::Int = 1` : The [Theiler window](@ref).
* `ntype = NeighborNumber(1)` : The neighborhood type. Either [`NeighborNumber`](@ref)
  or [`WithinRange`](@ref). See [Neighborhoods](@ref) for more info.
* `distance::Metric = Cityblock()` : The distance function used in the
  logarithmic distance of nearby states. The allowed distances are `Cityblock()`
  and `Euclidean()`. See below for more info. The metric for finding neighbors is
  always the Euclidean one.


## Description
If the dataset exhibits exponential divergence of nearby states, then it should hold
```math
E(k) \\approx \\lambda\\cdot k \\cdot \\Delta t + E(0)
```
for a *well defined region* in the `k` axis, where ``\\lambda`` is
the approximated maximum Lyapunov exponent. ``\\Delta t`` is the time between samples in the
original timeseries. You can use [`linear_region`](@ref) with arguments `(ks .* Δt, E)` to
identify the slope (= ``\\lambda``) immediatelly, assuming you
have choosen sufficiently good `ks` such that the linear scaling region is bigger
than the saturated region.

The algorithm used in this function is due to Parlitz[^Skokos2016], which itself
expands upon Kantz [^Kantz1994]. In sort, for
each reference state a neighborhood is evaluated. Then, for each point in this
neighborhood, the logarithmic distance between reference state and neighborhood
state(s) is calculated as the "time" index `k` increases. The average of the above over
all neighborhood states over all reference states is the returned result.

If the `Metric` is `Euclidean()` then use the Euclidean distance of the
full `D`-dimensional points (distance ``d_E`` in ref.[^Skokos2016]).
If however the `Metric` is `Cityblock()`, calculate
the absolute distance of *only the first elements* of the `m+k` and `n+k` points
of `R` (distance ``d_F`` in ref.[^Skokos2016], useful when `R` comes from delay embedding).

[^Skokos2016]: Skokos, C. H. *et al.*, *Chaos Detection and Predictability* - Chapter 1 (section 1.3.2), Lecture Notes in Physics **915**, Springer (2016)

[^Kantz1994]: Kantz, H., Phys. Lett. A **185**, pp 77–87 (1994)
"""
function lyapunov_from_data(
        R::AbstractDataset{D, T}, ks;
        refstates = 1:(length(R) - ks[end]),
        w = 1,
        distance = Cityblock(),
        ntype = NeighborNumber(1),
    ) where {D, T}
    Ek = lyapunov_from_data(R, ks, refstates, Theiler(w), distance, ntype)
end

function lyapunov_from_data(
        R::AbstractDataset{D, T},
        ks::AbstractVector{Int},
        ℜ::AbstractVector{Int},
        theiler,
        distance::Metric,
        ntype::SearchType
    ) where {D, T}

    # ℜ = \Re<tab> = set of indices that have the points that one finds neighbors.
    # n belongs in ℜ and R[n] is the "reference state".
    # Thus, ℜ contains all the reference states the algorithm will iterate over.
    # ℜ is not estimated. it is given by the user. Most common choice:
    # ℜ = 1:(length(R) - ks[end])

    # ⩅(n) = \Cup<tab> = neighborhood of reference state n
    # which is evaluated for each n and for the given neighborhood type

    timethres = length(R) - ks[end]
    if maximum(ℜ) > timethres
        erstr = "Maximum index of reference states is > length(R) - ks[end] "
        erstr*= "and the algorithm cannot be performed on it. You have to choose "
        erstr*= "reference state indices of at most up to length(R) - ks[end]."
        throw(ArgumentError(erstr))
    end
    E = zeros(T, length(ks))
    E_n, E_m = copy(E), copy(E)
    tree = KDTree(R, Euclidean())
    skippedm = 0; skippedn = 0

    for n in ℜ
        # The ⋓(n) can be evaluated on the spot instead of being pre-calculated
        # for all reference states. Precalculating is faster, but allocates more memory.
        # Since ⋓[n] doesn't depend on `k` one can then interchange the loops:
        # Instead of k being the outermost loop, it becomes the innermost loop!
        point = R[n]
        ⋓ = isearch(tree, point, ntype, theiler(n))
        for m in ⋓
            # If `m` is nearer to the end of the timeseries than k allows
            # is it completely skipped (and length(⋓) reduced).
            if m > timethres
                skippedm += 1
                continue
            end
            for (j, k) in enumerate(ks) #ks should be small (of order 10 to 100 MAX)
                E_m[j] = log(delay_distance(distance, R, m, n, k))
            end
            E_n .+= E_m # no need to reset E_m
        end
        if skippedm >= length(⋓)
            skippedn += 1
            skippedm = 0
            continue # be sure to continue if no valid point!
        end
        E .+= E_n ./ (length(⋓) - skippedm)
        skippedm = 0
        fill!(E_n, zero(T)) #reset distances for n-th reference state
    end

    if skippedn >= length(ℜ)
        ers = "Skipped number of points ≥ length(R)...\n"
        ers*= "Could happen because all the neighbors fall within the Theiler "
        ers*= "window. Fix: increase neighborhood size."
        error(ers)
    end
    E ./= (length(ℜ) - skippedn)
end

@inline function delay_distance(::Cityblock, R, m, n, k)
    @inbounds abs(R[m+k][1] - R[n+k][1])
end

@inline function delay_distance(::Euclidean, R, m, n, k)
    @inbounds norm(R[m+k] - R[n+k])
end
