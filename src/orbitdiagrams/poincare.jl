using DynamicalSystemsBase: DEFAULT_DIFFEQ_KWARGS, _get_solver
using Roots: find_zero, A42
export poincaresos, produce_orbitdiagram, PlaneCrossing, poincaremap, PoincareMap

const ROOTS_ALG = A42()

#####################################################################################
#                               Hyperplane                                          #
#####################################################################################
"""
    PlaneCrossing(plane, dir) → z
Create a struct that can be called as a function `z(u)` that returns the signed distance
of state `u` from the hyperplane `plane` (positive means in front of the hyperplane).
See [`poincaresos`](@ref) for what `plane` can be (tuple or vector).
"""
struct PlaneCrossing{P, D, T}
    plane::P
    dir::Bool
    n::SVector{D, T}  # normal vector
    p₀::SVector{D, T} # arbitrary point on plane
end
PlaneCrossing(plane::Tuple, dir) = PlaneCrossing(plane, dir, SVector(true), SVector(true))
function PlaneCrossing(plane::AbstractVector, dir)
    n = plane[1:end-1] # normal vector to hyperplane
    i = findfirst(!iszero, plane)
    D = length(plane)-1; T = eltype(plane)
    p₀ = zeros(D)
    p₀[i] = plane[end]/plane[i] # p₀ is an arbitrary point on the plane.
    PlaneCrossing(plane, dir, SVector{D, T}(n), SVector{D, T}(p₀))
end

# Definition of functional behavior
function (hp::PlaneCrossing{P})(u::AbstractVector) where {P<:Tuple}
    @inbounds x = u[hp.plane[1]] - hp.plane[2]
    hp.dir ? x : -x
end
function (hp::PlaneCrossing{P})(u::AbstractVector) where {P<:AbstractVector}
    x = zero(eltype(u))
    D = length(u)
    @inbounds for i in 1:D
        x += u[i]*hp.plane[i]
    end
    @inbounds x -= hp.plane[D+1]
    hp.dir ? x : -x
end

#####################################################################################
#                               Poincare Section                                    #
#####################################################################################
"""
    poincaresos(ds::ContinuousDynamicalSystem, plane, tfinal = 1000.0; kwargs...)
Calculate the Poincaré surface of section[^Tabor1989]
of the given system with the given `plane`.
The system is evolved for total time of `tfinal`.
Return a [`Dataset`](@ref) of the points that are on the surface of section.
See also [`poincaremap`](@ref) for the map version.

If the state of the system is ``\\mathbf{u} = (u_1, \\ldots, u_D)`` then the
equation defining a hyperplane is
```math
a_1u_1 + \\dots + a_Du_D = \\mathbf{a}\\cdot\\mathbf{u}=b
```
where ``\\mathbf{a}, b`` are the parameters of the hyperplane.

In code, `plane` can be either:

* A `Tuple{Int, <: Real}`, like `(j, r)` : the plane is defined
  as when the `j`th variable of the system equals the value `r`.
* A vector of length `D+1`. The first `D` elements of the
  vector correspond to ``\\mathbf{a}`` while the last element is ``b``.

This function uses `ds`, higher order interpolation from DifferentialEquations.jl,
and root finding from Roots.jl, to create a high accuracy estimate of the section.
See also [`produce_orbitdiagram`](@ref).

Notice that `poincaresos` is just a fancy wrapper of initializing a [`poincaremap`](@ref)
and then calling `trajectory` on it.

## Keyword Arguments
* `direction = -1` : Only crossings with `sign(direction)` are considered to belong to
  the surface of section. Positive direction means going from less than ``b``
  to greater than ``b``.
* `idxs = 1:dimension(ds)` : Optionally you can choose which variables to save.
  Defaults to the entire state.
* `Ttr = 0.0` : Transient time to evolve the system before starting
  to compute the PSOS.
* `u0 = get_state(ds)` : Specify an initial state.
* `warning = true` : Throw a warning if the Poincaré section was empty.
* `rootkw = (xrtol = 1e-6, atol = 1e-6)` : A `NamedTuple` of keyword arguments
  passed to `find_zero` from [Roots.jl](https://github.com/JuliaMath/Roots.jl).
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.

[^Tabor1989]:
    M. Tabor, *Chaos and Integrability in Nonlinear Dynamics: An Introduction*,
    §4.1, in pp. 118-126, New York: Wiley (1989)
"""
function poincaresos(
		ds::CDS{IIP, S, D}, plane, tfinal = 1000.0;
	    direction = -1, Ttr::Real = 0.0, warning = true, idxs = 1:D, u0 = get_state(ds),
	    rootkw = (xrtol = 1e-6, atol = 1e-6), diffeq = NamedTuple(), kwargs...
	) where {IIP, S, D}

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

    _check_plane(plane, D)
    integ = integrator(ds, u0; diffeq)
    i = typeof(idxs) <: Int ? idxs : SVector{length(idxs), Int}(idxs...)
    planecrossing = PlaneCrossing(plane, direction > 0)
	Ttr ≠ 0 && step!(integ, Ttr)
	plane_distance = (t) -> planecrossing(integ(t))

	data = _poincaresos(integ, plane_distance, planecrossing, tfinal+Ttr, i, rootkw)
    warning && length(data) == 0 && @warn PSOS_ERROR
    return Dataset(data)
end

# The separation into two functions here exists only to introduce a function barrier
# for the low level method, to ensure optimization on argments of `poincaremap!`.
function _poincaresos(integ, plane_distance, planecrossing, tfinal, i, rootkw)
	data = _initialize_output(integ.u, i)
	while integ.t < tfinal
		out, t = poincaremap!(integ, plane_distance, planecrossing, tfinal, rootkw)
		if !isnothing(out)
            push!(data, out[i])
        else
            break # if we evolved for more than tfinal, we should break anyways.
        end
	end
	return data
end

_initialize_output(::S, ::Int) where {S} = eltype(S)[]
_initialize_output(u::S, i::SVector{N, Int}) where {N, S} = typeof(u[i])[]
function _initialize_output(u, i)
    error("The variable index when producing the PSOS must be Int or SVector{Int}")
end

const PSOS_ERROR = "The Poincaré surface of section did not have any points!"

"""
	poincaremap!(integ, plane_distance, planecrossing, Tmax, rootkw)
Low level function that actual performs the algorithm of finding the next crossing
of the Poincaré surface of section. Return the state at the section or `nothing` if
evolved for more than `Tmax` without any crossing.
"""
function poincaremap!(integ, plane_distance, planecrossing, Tmax, rootkw)
    t0 = integ.t
    # Check if initial condition is already on the plane
    side = planecrossing(integ.u)
    if side == 0
		dat = integ.u
        step!(integ)
		return dat, t0
    end
    # Otherwise evolve until juuuuuust crossing the plane
    while side < 0
        (integ.t - t0) > Tmax && break
        step!(integ)
        side = planecrossing(integ.u)
    end
    while side ≥ 0
        (integ.t - t0) > Tmax && break
        step!(integ)
        side = planecrossing(integ.u)
    end
    # we evolved too long and no crossing, return nothing
    (integ.t - t0) > Tmax && return (nothing, nothing)
    # Else, we're guaranteed to have `t` after plane and `tprev` before plane
    tcross = Roots.find_zero(plane_distance, (integ.tprev, integ.t), Roots.A42(); rootkw...)
    ucross = integ(tcross)
    return ucross, tcross
end


function _check_plane(plane, D)
    P = typeof(plane)
    L = length(plane)
    if P <: AbstractVector
        if L != D + 1
            throw(ArgumentError(
            "The plane for the `poincaresos` must be either a 2-Tuple or a vector of "*
            "length D+1 with D the dimension of the system."
            ))
        end
    elseif P <: Tuple
        if !(P <: Tuple{Int, Number})
            throw(ArgumentError(
            "If the plane for the `poincaresos` is a 2-Tuple then "*
            "it must be subtype of `Tuple{Int, Number}`."
            ))
        end
    else
        throw(ArgumentError(
        "Unrecognized type for the `plane` argument."
        ))
    end
end

#####################################################################################
#                               Poincare Map                                        #
#####################################################################################
"""
	poincaremap(ds::ContinuousDynamicalSystem, plane, Tmax=1e3; kwargs...) → pmap

Return a map (integrator) that produces iterations over the Poincaré map of `ds`.
This map is defined as the sequence of points on the Poincaré surface of section.
See [`poincaresos`](@ref) for details on `plane` and all other `kwargs`.
Keyword `idxs` does not apply to `poincaremap`, as it doesn't save any states.

Notice that while in theory the Poincaré map has one less dimension than `ds`,
in code the map operates on the full `D`-dimensional state of `ds`
because that is the only way to accommodate planes with generic orientation.

The output `pmap` follows the [Integrator API](@ref), i.e., `step!` and `reinit!`.
`current_time(pmap)` returns the time of the last crossing.
For the special case of `plane` being a `Tuple{Int, <:Real}`, a special `reinit!` method
is allowed with input state with length `D-1` instead of `D`, i.e., a reduced state already
on the hyperplane that is then converted into the `D` dimensional state.

**Notice**: The argument `Tmax` exists so that the integrator can terminate instead
of being evolved for infinite time, to avoid cases where iteration would continue
forever for ill-defined hyperplanes or for convergence to fixed points.
If during one `step!` the system has been evolved for more than `Tmax`,
then `step!(pmap)` will terminate and return `nothing`.

## Example
```julia
ds = Systems.rikitake([0.,0.,0.], μ = 0.47, α = 1.0)
pmap = poincaremap(ds, (3, 0.0))
next_state_on_psos = step!(pmap)
# Change initial condition
reinit!(pmap, [1.0, 0]) # 3rd variable gets value 0 from the plane
next_state_on_psos = step!(pmap)
```
"""
function poincaremap(
		ds::CDS{IIP, S, D}, plane, Tmax = 1e3;
	    direction = -1, u0 = get_state(ds),
	    rootkw = (xrtol = 1e-6, atol = 1e-6), diffeq = NamedTuple(), kwargs...
	) where {IIP, S, D}

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

	_check_plane(plane, D)
    integ = integrator(ds, u0; diffeq)
	planecrossing = PlaneCrossing(plane, direction > 0)
	plane_distance = (t) -> planecrossing(integ(t))
    v = SVector{D, eltype(u0)}(u0)
    dummy = zeros(D)
    diffidxs = _indices_on_poincare_plane(plane, D)
	return PoincareMap(
        integ, plane_distance, planecrossing, Tmax, rootkw, v, 0.0, dummy, diffidxs
    )
end

_indices_on_poincare_plane(plane::Tuple, D) = setdiff(1:D, [plane[1]])
_indices_on_poincare_plane(::Vector, D) = collect(1:D-1)

mutable struct PoincareMap{I, F, P, R, V} <: GeneralizedDynamicalSystem
	integ::I
	f::F
 	planecrossing::P
	Tmax::Float64
	rootkw::R
	state_on_plane::V
    tcross::Float64
    # These two fields are for setting the state of the pmap from the plane
    # (i.e., given a D-1 dimensional state, create the full D-dimensional state)
    dummy::Vector{Float64}
    diffidxs::Vector{Int}
end
DynamicalSystemsBase.isdiscretetime(p::PoincareMap) = true
DelayEmbeddings.dimension(p::PoincareMap) = length(p.state_on_plane)
DynamicalSystemsBase.integrator(pinteg::PoincareMap, args...; kwargs...) = pinteg

function DynamicalSystemsBase.step!(pmap::PoincareMap)
	u, t = poincaremap!(pmap.integ, pmap.f, pmap.planecrossing, pmap.Tmax, pmap.rootkw)
	if isnothing(u)
		return nothing
	else
		pmap.state_on_plane = u
        pmap.tcross = t
		return pmap.state_on_plane
	end
end
DynamicalSystemsBase.step!(pmap::PoincareMap, n::Int) = for _ ∈ 1:n; step!(pmap); end

function DynamicalSystemsBase.reinit!(pmap::PoincareMap, u0)
    if length(u0) == dimension(pmap)
	    u0 = u0
    elseif length(u0) == dimension(pmap) - 1
        u0 = _recreate_state_from_poincare_plane(pmap, u0)
    else
        error("Dimension of state for poincare map reinit is inappropriate.")
    end
    reinit!(pmap.integ, u0)
end
function _recreate_state_from_poincare_plane(pmap::PoincareMap, u0)
    plane = pmap.planecrossing.plane
    if plane isa Tuple
        pmap.dummy[pmap.diffidxs] .= u0
        pmap.dummy[plane[1]] = plane[2]
    else
        error("Don't know how to convert state on generic plane into full space.")
    end
    return pmap.dummy
end

DynamicalSystemsBase.get_state(pmap::PoincareMap) = pmap.state_on_plane
DynamicalSystemsBase.current_time(pmap::PoincareMap) = pmap.tcross

function Base.show(io::IO, pmap::PoincareMap)
    println(io, "Iterator of the Poincaré map of a $(dimension(pmap))-dimensional system")
    println(io,  rpad(" rule f: ", 14),     DynamicalSystemsBase.eomstring(pmap.integ.f.f))
    println(io,  rpad(" hyperplane: ", 14),     pmap.planecrossing.plane)
end

#####################################################################################
# Poincare Section for Datasets (trajectories)                                      #
#####################################################################################
# TODO: Nice improvement would be to use cubic interpolation instead of linear,
# using points i-2, i-1, i, i+1
"""
    poincaresos(A::Dataset, plane; kwargs...)
Calculate the Poincaré surface of section of the given dataset with the given `plane`
by performing linear interpolation betweeen points that sandwich the hyperplane.

Argument `plane` and keywords `direction, warning, idxs` are the same as above.
"""
function poincaresos(A::Dataset, plane; direction = -1, warning = true, idxs = 1:size(A, 2))
    _check_plane(plane, size(A, 2))
    i = typeof(idxs) <: Int ? idxs : SVector{length(idxs), Int}(idxs...)
    planecrossing = PlaneCrossing(plane, direction > 0)
    data = poincaresos(A, planecrossing, i)
    warning && length(data) == 0 && @warn PSOS_ERROR
    return Dataset(data)
end
function poincaresos(A::Dataset, planecrossing::PlaneCrossing, j)
    i, L = 1, length(A)
    data = _initialize_output(A[1], j)
    # Check if initial condition is already on the plane
    planecrossing(A[i]) == 0 && push!(data, A[i][j])
    i += 1
    side = planecrossing(A[i])

    while i ≤ L # We always check point i vs point i-1
        while side < 0 # bring trajectory infront of hyperplane
            i == L && break
            i += 1
            side = planecrossing(A[i])
        end
        while side ≥ 0 # iterate until behind the hyperplane
            i == L && break
            i += 1
            side = planecrossing(A[i])
        end
        i == L && break
        # It is now guaranteed that A crosses hyperplane between i-1 and i
        ucross = interpolate_crossing(A[i-1], A[i], planecrossing)
        push!(data, ucross[j])
    end
    return data
end

function interpolate_crossing(A, B, pc::PlaneCrossing{<:AbstractVector})
    # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    t = LinearAlgebra.dot(pc.n, (pc.p₀ .- A))/LinearAlgebra.dot((B .- A), pc.n)
    return A .+ (B .- A) .* t
end

function interpolate_crossing(A, B, pc::PlaneCrossing{<:Tuple})
    # https://en.wikipedia.org/wiki/Linear_interpolation
    y₀ = A[pc.plane[1]]; y₁ = B[pc.plane[1]]; y = pc.plane[2]
    t = (y - y₀) / (y₁ - y₀) # linear interpolation with t₀ = 0, t₁ = 1
    return A .+ (B .- A) .* t
end
