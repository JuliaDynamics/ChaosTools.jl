using DynamicalSystemsBase: DEFAULT_DIFFEQ_KWARGS, _get_solver
using Roots: find_zero, A42, AlefeldPotraShi, Brent
export poincaresos, produce_orbitdiagram, PlaneCrossing

const ROOTS_ALG = A42()

#####################################################################################
#                               Poincare Section                                    #
#####################################################################################
"""
    poincaresos(ds::ContinuousDynamicalSystem, plane, tfinal = 1000.0; kwargs...)
Calculate the Poincaré surface of section (also called Poincaré map) [1, 2]
of the given system with the given `plane`.
The system is evolved for total time of `tfinal`.

If the state of the system is ``\\mathbf{u} = (u_1, \\ldots, u_D)`` then the
equation defining a hyperplane is
```math
a_1u_1 + \\dots + a_Du_D = \\mathbf{a}\\cdot\\mathbf{u}=b
```
where ``\\mathbf{a}, b`` are the parameters of the hyperplane.

In code, `plane` can be either:

* A `Tuple{Int, <: Number}`, like `(j, r)` : the plane is defined
  as when the `j` variable of the system equals the value `r`.
* A vector of length `D+1`. The first `D` elements of the
  vector correspond to ``\\mathbf{a}`` while the last element is ``b``.

Returns a [`Dataset`](@ref) of the points that are on the surface of section.

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
* `diffeq...` : All other extra keyword arguments are propagated into `init`
  of DifferentialEquations.jl. See [`trajectory`](@ref) for examples.

## Performance Notes
This function uses a standard [`integrator`](@ref). For loops over initial conditions
and/or parameters you should use the low level method that accepts an integrator and
`reinit!` to new initial conditions. See the "advanced documentation" for more.

The low level call signature is:
```julia
poincaresos(integ, planecrossing, tfinal, Ttr, idxs, rootkw)
```
where
```julia
planecrossing = PlaneCrossing(plane, direction > 0)
```
and `idxs` must be `Int` or `SVector{Int}`.

## References
[1] : H. Poincaré, *Les Methods Nouvelles de la Mécanique Celeste*,
Paris: Gauthier-Villars (1892)

[2] : M. Tabor, *Chaos and Integrability in Nonlinear Dynamics: An Introduction*,
§4.1, in pp. 118-126, New York: Wiley (1989)

See also [`orbitdiagram`](@ref), [`produce_orbitdiagram`](@ref).
"""
function poincaresos(ds::CDS{IIP, S, D}, plane, tfinal = 1000.0;
    direction = -1, Ttr::Real = 0.0, warning = true, idxs = 1:D, u0 = get_state(ds),
    rootkw = (xrtol = 1e-6, atol = 1e-6), diffeq...) where {IIP, S, D}

    _check_plane(plane, D)
    integ = integrator(ds, u0; diffeq...)

    i = typeof(idxs) <: Int ? i : SVector{length(idxs), Int}(idxs...)
    planecrossing = PlaneCrossing(plane, direction > 0)

    data = poincaresos(integ, planecrossing, tfinal, Ttr, i, rootkw)
    warning && length(data) == 0 && @warn PSOS_ERROR

    return Dataset(data)
end

_initialize_output(u::S, i::Int) where {S} = eltype(S)[]
_initialize_output(u::S, i::SVector{N, Int}) where {N, S} = typeof(u[i])[]
function _initialize_output(u, i)
    error("The variable index when producing the PSOS must be Int or SVector{Int}")
end

const PSOS_ERROR =
"the Poincaré surface of section did not have any points!"

struct PlaneCrossing{P}
    plane::P
    dir::Bool
end
PlaneCrossing(p::P, b) where {P} = PlaneCrossing{P}(p, b)
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

function poincaresos(integ, planecrossing, tfinal, Ttr, j, rootkw)

    f = (t) -> planecrossing(integ(t))
    data = _initialize_output(integ.u, j)
    Ttr != 0 && step!(integ, Ttr)

    # Check if initial condition is already on the plane
    side = planecrossing(integ.u)
    if side == 0
        push!(data, integ.u[j])
        step!(integ)
        side = planecrossing(integ.u)
    end

    while integ.t < tfinal + Ttr
        while side < 0
            integ.t > tfinal + Ttr && break
            step!(integ)
            side = planecrossing(integ.u)
        end
        while side ≥ 0
            integ.t > tfinal + Ttr && break
            step!(integ)
            side = planecrossing(integ.u)
        end
        integ.t > tfinal + Ttr && break

        # I am now guaranteed to have `t` in negative and `tprev` in positive
        tcross = find_zero(f, (integ.tprev, integ.t), ROOTS_ALG; rootkw...)
        ucross = integ(tcross)

        push!(data, ucross[j])
    end
    return data
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
#                            Produce Orbit Diagram                                  #
#####################################################################################
"""
    produce_orbitdiagram(ds::ContinuousDynamicalSystem, plane, i::Int,
                         p_index, pvalues; kwargs...)
Produce an orbit diagram (also called bifurcation diagram)
for the `i` variable(s) of the given continuous
system by computing Poincaré surfaces of section using `plane`
for the given parameter values (see [`poincaresos`](@ref)).

`i` can be `Int` or `AbstractVector{Int}`.
If `i` is `Int`, returns a vector of vectors. Else
it returns a vector of vectors of vectors.
Each entry are the points at each parameter value.

## Keyword Arguments
* `printparams::Bool = false` : Whether to print the parameter used
  during computation in order to keep track of running time.
* `direction, warning, Ttr, rootkw, diffeq...` :
  Propagated into [`poincaresos`](@ref).
* `u0 = get_state(ds)` : Initial condition. Besides a vector you can also give
  a vector of vectors such that `length(u0) == length(pvalues)`. Then each parameter
  has a different initial condition.

## Description
For each parameter, a PSOS reduces the system from a flow to a map. This then allows
the formal computation of an "orbit diagram" for the `i` variable
of the system, just like it is done in [`orbitdiagram`](@ref).

The parameter change is done as `p[p_index] = value` taking values from `pvalues`
and thus you must use a parameter container that supports this
(either `Array`, `LMArray`, dictionary or other).

See also [`poincaresos`](@ref), [`orbitdiagram`](@ref).
"""
function produce_orbitdiagram(ds::CDS{IIP, S, D},
                              plane,
                              idxs,
                              p_index,
                              pvalues;
                              tfinal::Real = 100.0,
                              direction = -1,
                              printparams = false,
                              warning = true,
                              Ttr = 0.0,
                              u0 = get_state(ds),
                              rootkw = (xrtol = 1e-6, atol = 1e-6),
                              diffeq...) where {IIP, S, D}

    i = typeof(idxs) <: Int ? idxs : SVector{length(idxs), Int}(idxs...)

    _check_plane(plane, D)
    typeof(u0) <: Vector{<:AbstractVector} && @assert length(u0)==length(p)

    integ = integrator(ds; diffeq...)
    planecrossing = PlaneCrossing(plane, direction > 0)

    p0 = ds.p[p_index]

    output = Vector{typeof(ds.u0[i])}[]

    for (n, p) in enumerate(pvalues)
        integ.p[p_index] = p
        printparams && println("parameter = $p")

        if typeof(u0) <: Vector{<:AbstractVector}
            st = u0[n]
        else
            st = u0
        end

        reinit!(integ, st)
        push!(output, poincaresos(integ, planecrossing, tfinal, Ttr, i, rootkw))

        warning && length(output[end]) == 0 && @warn "For parameter $p $PSOS_ERROR"

    end
    # Reset the parameter of the system:
    ds.p[p_index] = p0
    return output
end
