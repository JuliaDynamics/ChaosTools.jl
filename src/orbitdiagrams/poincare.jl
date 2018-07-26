using OrdinaryDiffEq, DiffEqBase
using DynamicalSystemsBase: DEFAULT_DIFFEQ_KWARGS, _get_solver
using Roots: find_zero, Bisection, FalsePosition
export poincaresos, produce_orbitdiagram

#####################################################################################
#                               Poincare Section                                    #
#####################################################################################
"""
    poincaresos(ds::ContinuousDynamicalSystem, plane, tfinal = 1000.0; kwargs...)
Calculate the Poincaré surface of section (also called Poincaré map) [1, 2]
of the given system with the given `plane`.
The system is evolved for total time of `tfinal`.

If the state of the system is ``\\mathbf{u} = (u_1, \\ldots, u_D)`` then the
equation for the planecrossing is
```math
a_1u_1 + \\dots + a_Du_D = \\mathbf{a}\\cdot\\mathbf{u}=b
```
where ``\\mathbf{a}, b`` are the parameters that define the planecrossing.

In code, `plane` can be either:

* A `Tuple{Int, <: Number}`, like `(j, r)` : the planecrossing is defined
  as when the `j` variable of the system crosses the value `r`.
* An `AbstractVector` of length `D+1`. The first `D` elements of the
  vector correspond to ``\\mathbf{a}`` while the last element is ``b``.

Returns a [`Dataset`](@ref) of the points that are on the surface of section.

## Keyword Arguments
* `direction = 1` : Only crossings with `sign(direction)` are considered to belong to
  the surface of section. Positive direction means going from less than ``b``
  to greater than ``b``.
* `Ttr = 0.0` : Transient time to evolve the system before starting
  to compute the PSOS.
* `warning = true` : Throw a warning if the Poincaré section was empty.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples.

## References
[1] : H. Poincaré, *Les Methods Nouvelles de la Mécanique Celeste*,
Paris: Gauthier-Villars (1892)

[2] : M. Tabor, *Chaos and Integrability in Nonlinear Dynamics: An Introduction*,
§4.1, in pp. 118-126, New York: Wiley (1989)

[3] : This function is simply manipulating [`ContinuousCallback`](http://docs.juliadiffeq.org/latest/features/callback_functions.html) from
DifferentialEquations.jl.

See also [`orbitdiagram`](@ref), [`produce_orbitdiagram`](@ref).
"""
function poincaresos(ds::CDS{IIP, S, D}, plane, tfinal = 1000.0;
    direction = +1, Ttr::Real = 0.0, warning = true,
    diffeq...) where {IIP, S, D}

    _check_plane(plane, D)
    integ = integrator(ds; diffeq...)
    planecrossing = PlaneCrossing{D}(plane, direction > 0 )
    f = (t) -> planecrossing(integ(t))

    psos = Dataset{D, eltype(S)}()
    _poincare_cross!(psos.data, integ,
                     f, planecrossing, tfinal, Ttr)
    warning && length(psos) == 0 && warn(PSOS_ERROR)
    return psos
end

const PSOS_ERROR =
"the Poincaré surface of section did not have any points!"

struct PlaneCrossing{D, P}
    plane::P
    dir::Bool
end
PlaneCrossing{D}(p::P, b) where {P, D} = PlaneCrossing{D, P}(p, b)
function (hp::PlaneCrossing{D, P})(u::AbstractVector) where {D, P<:Tuple}
    @inbounds x = u[hp.plane[1]] - hp.plane[2]
    hp.dir ? x : -x
end
function (hp::PlaneCrossing{D, P})(u::AbstractVector) where {D, P<:AbstractVector}
    x = zero(eltype(u))
    @inbounds for i in 1:D
        x += u[i]*hp.plane[i]
    end
    @inbounds x -= hp.plane[D+1]
    hp.dir ? x : -x
end

function _poincare_cross!(data, integ,
                          f, planecrossing, tfinal, Ttr, j = 1)

    Ttr != 0 && step!(integ, Ttr)

    side = planecrossing(integ.u)

    while integ.t < tfinal + Ttr
        while side < 0
            integ.t > tfinal + Ttr && break
            step!(integ)
            side = planecrossing(integ.u)
        end
        while side > 0
            integ.t > tfinal + Ttr && break
            step!(integ)
            side = planecrossing(integ.u)
        end
        integ.t > tfinal + Ttr && break

        # I am now guaranteed to have `t` in negative and `tprev` in positive
        tcross = find_zero(f, (integ.tprev, integ.t), FalsePosition(),
                           xrtol = 1e-3, atol = 1e-3)

        ucross = integ(tcross)

        _fill_crossing_data(data, ucross, j)
    end
    return data
end

_fill_crossing_data(data::Vector{<:SVector}, ucross, j) = push!(data, ucross)
_fill_crossing_data(data::Vector{<:Number}, ucross, j) = push!(data, ucross[j])

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
for the `i`-th variable of the given continuous
system by computing Poincaré surfaces of section using `plane`
for the given parameter values (see [`poincaresos`](@ref)).

## Keyword Arguments
* `printparams::Bool = false` : Whether to print the parameter used during computation
  in order to keep track of running time.
* `ics = [get_state(ds)]` : Collection of initial conditions.
  For every `state ∈ ics` a PSOS will be produced.
* `direction, warning, Ttr, diffeq...` : Propagated into [`poincaresos`](@ref).

## Description
For each parameter, a PSOS reduces the system from a flow to a map. This then allows
the formal computation of an "orbit diagram" for the `i` variable
of the system, just like it is done in [`orbitdiagram`](@ref).

The parameter change is done as `p[p_index] = value` taking values from `pvalues`
and thus you must use a parameter container that supports this
(either `Array`, `LMArray`, dictionary or other).

The returned `output` is a vector of vectors. `output[k]` are the
"orbit diagram" points of the `i`-th variable of the system,
at parameter value `pvalues[k]`.

## Performance Notes
The total amount of PSOS produced will be `length(ics)*length(pvalues)`.

See also [`poincaresos`](@ref), [`orbitdiagram`](@ref).
"""
function produce_orbitdiagram(
    ds::CDS{IIP, S, D},
    plane,
    i::Int,
    p_index,
    pvalues;
    tfinal::Real = 100.0,
    ics = [get_state(ds)],
    direction = +1,
    printparams = true,
    warning = true,
    Ttr = 0.0,
    diffeq...) where {IIP, S, D}

    _check_plane(plane, D)
    integ = integrator(ds; diffeq...)
    planecrossing = PlaneCrossing{D}(plane, direction > 0 )
    f = (t) -> planecrossing(integ(t))
    p0 = ds.p[p_index]

    output = [Vector{eltype(S)}() for j in 1:length(pvalues)]

    for (n, p) in enumerate(pvalues)
        integ.p[p_index] = p
        printparams && println("parameter = $p")

        for (m, st) in enumerate(ics)

            reinit!(integ, st)
            _poincare_cross!(output[n], integ,
                             f, planecrossing, tfinal, Ttr, i)

            warning && length(output[n]) == 0 && warn(
            "For parameter $p and initial condition index $m $PSOS_ERROR")
        end
    end
    # Reset the parameter of the system:
    ds.p[p_index] = p0
    return output
end
