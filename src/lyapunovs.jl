using StaticArrays, OrdinaryDiffEq
using OrdinaryDiffEq: ODEIntegrator
using DynamicalSystemsBase: MinimalDiscreteIntegrator

export lyapunovs, lyapunov

#####################################################################################
#                               Lyapunov Spectum                                    #
#####################################################################################
"""
    lyapunovs(ds::DynamicalSystem, N, k::Int | Q0; kwargs...) -> λs, t

Calculate the spectrum of Lyapunov exponents [1] of `ds` by applying
a QR-decomposition on the parallelepiped matrix `N` times. Return the
spectrum convergence timeseries and the time vector.

The third argument `k` is optional, and dictates how many lyapunov exponents
to calculate (defaults to `dimension(ds)`).
Instead of passing an integer `k` you can also pass
a pre-initialized matrix `Q0` whose columns are initial deviation vectors.

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
The method we employ is "H2" of [2], originally stated in [3]. The vectors
defining a `D`-dimensional parallepiped are evolved using the tangent dynamics
of the system.
A QR-decomposition at each step yields the local growth rate for each dimension
of the parallepiped. The growth rates are
then averaged over `N` successive steps, yielding the lyapunov exponent spectrum.

## Performance Notes
This function uses a `tangent_integrator`. For loops over initial conditions and/or
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
    if IIP
        if k == D && D < 20
            return _lyapunovs_iip(tode, N, dt, Ttr, k, DynamicalSystemsBase.qr_sq)
        else
            return _lyapunovs_iip(tode, N, dt, Ttr, k, Base.qr)
        end
    else
        return _lyapunovs_oop(tode, N, dt, Ttr, Val{k}())
    end
end

function _lyapunovs_iip(integ, N, dt::Real, Ttr::Real, k::Int, qrf::Function)

    t0 = integ.t
    if Ttr > 0
        while integ.t < t0 + Ttr
            step!(integ, dt)
            Q, R = qrf(view(integ.u, :, 2:k+1))
            view(integ.u, :, 2:k+1) .= Q
            u_modified!(integ, true)
        end
    end

    λ = zeros(k)
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

    λ = zeros(k)
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
    lyapunov(ds::DynamicalSystem, Τ; kwargs...)

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
  distance exceeding the thresholds.
* `inittest = (st1, d0) -> st1 .+ d0/sqrt(D)` :
  A function that given `(st1, d0)`
  initializes the test state with distance
  `d0` from the given state `st1` (`D` is the dimension
  of the system). This function can be used when you want to avoid
  the test state appearing in a region of the phase-space where it would have
  e.g. different energy or escape to infinity.

## Description
Two neighboring trajectories with initial distance `d0` are evolved in time.
At time ``d(t_i)`` their distance either exceeds the `upper_threshold`,
or is lower than `lower_threshold`, which initializes
a rescaling of the test trajectory back to having distance `d0` from
the given one, while the rescaling keeps the difference vector along the maximal
expansion/contraction direction.

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

    lower_threshold ≤ d0 ≤ upper_threshold || throw(ArgumentError(
    "d0 must be between thresholds!"))
    D = dimension(ds)
    if typeof(ds) <: DDS
        # Time assertions
        @assert typeof(Ttr) == Int
        pinteg = parallel_integrator(ds, [deepcopy(state(ds)), inittest(state(ds), d0)])
    else
        pinteg = parallel_integrator(ds, [deepcopy(state(ds)), inittest(state(ds), d0)];
        diff_eq_kwargs = diff_eq_kwargs)
    end

    return _lyapunov(pinteg, T, Ttr, dt, d0, upper_threshold, lower_threshold)
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
    for i in size(integ.u)[1]
        d += (integ.u[i, 1] - integ.u[i, 2])^2
    end
    return sqrt(d)
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
    for i in size(integ.u)[1]
        integ.u[i, 2] = integ.u[i,1] + (integ.u[i,2] - integ.u[i,1])/a
    end
end
function rescale!(integ::ODEIntegrator{Alg, Vector{S}}, a) where {Alg, S<:Vector}
    @. integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
end
function rescale!(integ::ODEIntegrator{Alg, Vector{S}}, a) where {Alg, S<:SVector}
    integ.u[2] = integ.u[1] + (integ.u[2] - integ.u[1])/a
end


# tvector = dt:dt:T
# finalτ = dt
#
# # start evolution and rescaling:
# for τ in tvector
#     # evolve until rescaling:
#     push!(integ1.opts.tstops, τ)
#     step!(integ1)
#     push!(integ2.opts.tstops, τ)
#     step!(integ2)
#     dist = norm(integ1.u .- integ2.u)
#     # Rescale:
#     if dist ≥ threshold
#         # add computed scale to accumulator (scale = local lyaponov exponent):
#         a = dist/d0
#         λ += log(a)
#         finalτ = τ
#         # Rescale and reset everything:
#         # Must rescale towards difference direction:
#         @. integ2.u = integ1.u + (integ2.u - integ1.u)/a
#         u_modified!(integ2, true)
#         set_proposed_dt!(integ2, integ1)
#         dist = d0
#     end
# end
# ds.prob.u0 .= initu0
# return λ/finalτ
# end
#
#
#
#
# function lyapunov(ds::DiscreteDS,
#                   N::Int;
#                   Ttr::Int = 0,
#                   d0=1e-9,
#                   threshold=1e-5,
#                   inittest = inittest_default(dimension(ds))
#                   )
#
#     threshold <= d0 && throw(ArgumentError("Threshold must be bigger than d0!"))
#
#     st1 = evolve(ds, Ttr)
#     st2 = inittest(st1, d0)
#     eom = (x) -> ds.eom(x, ds.p)
#     dist = d0
#     λ = zero(eltype(st1))
#     i = 0
#
#     while i < N
#         #evolve until rescaling:
#         while dist < threshold
#             st1 = eom(st1)
#             st2 = eom(st2)
#             dist = norm(st1 - st2)
#             i+=1
#             i>=N && break
#         end
#         # local lyapunov exponent is simply the relative distance of the trajectories
#         a = dist/d0
#         λ += log(a)
#         i>=N && break
#         #rescale:
#         st2 = st1 + (st2 - st1)/a #must rescale in direction of difference
#         dist = d0
#     end
#     return λ/i
# end
#
#
#
# function lyapunov(ds::BigDiscreteDS,
#                   N::Int;
#                   Ttr::Int = 0,
#                   d0=1e-9,
#                   threshold=1e-5,
#                   inittest = inittest_default(dimension(ds))
#                   )
#
#     threshold <= d0 && throw(ArgumentError("Threshold must be bigger than d0!"))
#
#     st1 = evolve(ds, Ttr)
#     st2 = inittest(st1, d0)
#
#     dist::eltype(st1) = d0
#     λ = zero(eltype(st1))
#     i = 0
#     while i < N
#         #evolve until rescaling:
#         while dist < threshold
#             ds.dummystate .= st1
#             ds.eom!(st1, ds.dummystate, ds.p);
#             ds.dummystate .= st2
#             ds.eom!(st2, ds.dummystate, ds.p);
#             ds.dummystate .= st1 .- st2
#             dist = norm(ds.dummystate)
#             i+=1
#             i>=N && break
#         end
#         # local lyapunov exponent is simply the relative distance of the trajectories
#         a = dist/d0
#         λ += log(a)
#         i>=N && break
#         #rescale:
#         @. st2 = st1 + (st2 - st1)/a #must rescale in direction of difference
#         dist = d0
#     end
#     return λ/i
# end
#
#
#
# function lyapunovs(ds::DiscreteDS1D, N::Real = 10000; Ttr::Int = 0)
#
#     #transient system evolution
#     x = Ttr > 0 ? evolve(ds, Ttr) : state(ds)
#
#     # The case for 1D systems is trivial: you add log(abs(der(x))) at each step
#     λ = log(abs(ds.deriv(x, ds.p)))
#     for i in 1:N
#         x = ds.eom(x, ds.p)
#         λ += log(abs(ds.deriv(x, ds.p)))
#     end
#     λ/N
# end
#
# lyapunov(ds::DiscreteDS1D, N::Int=10000; Ttr::Int = 100) = lyapunovs(ds, N, Ttr=Ttr)
#
#
#
#
#
# #####################################################################################
# #                            Continuous Lyapunovs                                   #
# #####################################################################################
# function lyapunovs(ds::ContinuousDynamicalSystem, N::Real=1000;
#     Ttr::Real = 0.0, diff_eq_kwargs::Dict = Dict(), dt::Real = 0.1)
#     # Initialize
#     tstops = dt:dt:N*dt
#     D = dimension(ds)
#     λ = zeros(eltype(ds), D)
#     Q = eye(eltype(ds), D)
#     # Transient evolution:
#     st = Ttr != 0 ? evolve(ds, Ttr; diff_eq_kwargs = diff_eq_kwargs) : ds.prob.u0
#     # Create integrator for dynamics and tangent space:
#     S = [st eye(eltype(ds), D)]
#     integ = variational_integrator(
#     ds, S, tstops[end]; diff_eq_kwargs = diff_eq_kwargs)
#
#     # Main algorithm
#     for τ in tstops
#         integ.u[:, 2:end] .= Q # update tangent dynamics state (super important!)
#         u_modified!(integ, true)
#         # Integrate
#         while integ.t < τ
#             step!(integ)
#         end
#         # Perform QR (on the tangent flow):
#         Q, R = DynamicalSystemsBase.qr_sq(view(integ.u, :, 2:D+1))
#         # Add correct (positive) numbers to Lyapunov spectrum
#         for j in 1:D
#             λ[j] += log(abs(R[j,j]))
#         end
#     end
#     λ./(integ.t) #return spectrum
# end
#
#
#
# function lyapunov(ds::ContinuousDynamicalSystem,
#                   T::Real;
#                   Ttr = 0.0,
#                   d0=1e-9,
#                   threshold=1e-5,
#                   dt = 1.0,
#                   diff_eq_kwargs = Dict(:abstol=>d0, :reltol=>d0),
#                   inittest = inittest_default(dimension(ds)),
#                   )
#
#     DynamicalSystemsBase.check_tolerances(d0, diff_eq_kwargs)
#     S = eltype(ds)
#
#     T = convert(eltype(ds), T)
#     threshold <= d0 && throw(ArgumentError("Threshold must be bigger than d0!"))
#
#     # Transient evolution:
#     initu0 = deepcopy(ds.prob.u0)
#     if Ttr != 0
#         ds.prob.u0 .= evolve(ds, Ttr; diff_eq_kwargs = diff_eq_kwargs)
#     end
#
#     # Initialize:
#     st1 = copy(ds.prob.u0)
#     integ1 = ODEIntegrator(ds, T; diff_eq_kwargs=diff_eq_kwargs)
#     integ1.opts.advance_to_tstop=true
#     ds.prob.u0 .= inittest(st1, d0)
#     integ2 = ODEIntegrator(ds, T; diff_eq_kwargs=diff_eq_kwargs)
#     integ2.opts.advance_to_tstop=true
#     ds.prob.u0 .= st1
#     dist = d0
#     λ::eltype(integ1.u) = 0.0
#     i = 0;
#     tvector = dt:dt:T
#     finalτ = dt
#
#     # start evolution and rescaling:
#     for τ in tvector
#         # evolve until rescaling:
#         push!(integ1.opts.tstops, τ)
#         step!(integ1)
#         push!(integ2.opts.tstops, τ)
#         step!(integ2)
#         dist = norm(integ1.u .- integ2.u)
#         # Rescale:
#         if dist ≥ threshold
#             # add computed scale to accumulator (scale = local lyaponov exponent):
#             a = dist/d0
#             λ += log(a)
#             finalτ = τ
#             # Rescale and reset everything:
#             # Must rescale towards difference direction:
#             @. integ2.u = integ1.u + (integ2.u - integ1.u)/a
#             u_modified!(integ2, true)
#             set_proposed_dt!(integ2, integ1)
#             dist = d0
#         end
#     end
#     ds.prob.u0 .= initu0
#     return λ/finalτ
# end
# =#
# #=
# function lyapunov_distance(S, D)
#     x = zero(eltype(S))
#     for i in 1:D
#         @inbounds x += (S[i] - S[i+D])^2
#     end
#     sqrt(x)
# end
#
# function lyapunov2(ds::ContinuousDynamicalSystem,
#                   T::Real;
#                   Ttr = 0.0,
#                   d0=1e-9,
#                   threshold=1e-5,
#                   dt = 1.0,
#                   diff_eq_kwargs = Dict(:abstol=>d0, :reltol=>d0),
#                   inittest = inittest_default(dimension(ds)),
#                   )
#
#     DynamicalSystemsBase.check_tolerances(d0, diff_eq_kwargs)
#     S = eltype(ds); D = dimension(ds)
#
#     threshold <= d0 && throw(ArgumentError("Threshold must be bigger than d0!"))
#
#     # Transient evolution:
#     u0 = Ttr != 0 ? evolve(ds, Ttr; diff_eq_kwargs = diff_eq_kwargs) : state(ds)
#
#     # Initialize:
#     S = cat(2, u0, inittest(u0, d0))
#     pinteg = parallel_integrator(ds, S, T; diff_eq_kwargs = diff_eq_kwargs)
#
#     λ::eltype(pinteg.u) = 0.0
#     i = 0;
#     tvector = dt:dt:T
#     finalτ = dt
#
#     # start evolution and rescaling:
#     for τ in tvector
#         # evolve until rescaling:
#         # Integrate
#         while pinteg.t < τ
#             step!(pinteg)
#         end
#         dist = lyapunov_distance(pinteg.u, D)
#         # Rescale:
#         if dist ≥ threshold
#             # add computed scale to accumulator (scale = local lyaponov exponent):
#             a = dist/d0
#             λ += log(a)
#             finalτ = pinteg.t
#             # Rescale and reset everything
#             # Must rescale towards difference direction:
#             @. @inbounds pinteg.u[:, 2] =
#             @views pinteg.u[:, 1] + (pinteg.u[:, 2] - pinteg.u[:, 1])/a
#             u_modified!(pinteg, true)
#             dist = d0
#         end
#     end
#     return λ/finalτ
# end
# =#
