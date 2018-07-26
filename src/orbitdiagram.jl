using OrdinaryDiffEq, DiffEqBase
using DynamicalSystemsBase: DEFAULT_DIFFEQ_KWARGS, _get_solver
using Roots
export poincaresos, orbitdiagram, produce_orbitdiagram

#####################################################################################
#                                 Orbit Diagram                                     #
#####################################################################################
"""
    orbitdiagram(ds::DiscreteDynamicalSystem, i, p_index, pvalues; kwargs...)
Compute the orbit diagram (also called bifurcation diagram) of the given system
for the `i`-th variable for parameter values `pvalues`. The `p_index` specifies
which parameter of the equations of motion is to be changed.

Returns a vector of vectors, where each entry are the points are each parameter value.

## Keyword Arguments
* `ics = [get_state(ds)]` : container of initial conditions that
  are used at each parameter value to evolve orbits.
* `Ttr::Int = 1000` : Transient steps;
  each orbit is evolved for `Ttr` first before saving output.
* `n::Int = 100` : Amount of points to save for each initial condition.

## Description
The method works by computing orbits at each parameter value in `pvalues` for each
initial condition in `ics`.

The parameter change is done as `p[p_index] = ...` and thus you must use
a parameter container that supports this (either `Array`, `LMArray`, dictionary
or other).

The returned `output` is a vector of vectors. `output[j]` are the orbit points of the
`i`-th variable of the system, at parameter value `pvalues[j]`.

See also [`poincaresos`](@ref) and [`produce_orbitdiagram`](@ref).
"""
function orbitdiagram(ds::DDS{IIP, S, D}, i::Int, p_index, pvalues;
    n::Int = 100, Ttr::Int = 1000, ics = [get_state(ds)]) where {IIP, S, D}

    p0 = ds.p[p_index]
    if D == 1
        i != 1 &&
        error("You have a 1D system and yet you gave `i=$i`. What's up with that!?")
    end

    output = [zeros(eltype(S), n*length(ics)) for i in 1:length(pvalues)]

    integ = integrator(ds)

    for (j, p) in enumerate(pvalues)

        integ.p[p_index] = p

        for (m, st) in enumerate(ics)
            reinit!(integ, st)
            step!(integ, Ttr)

            for k in 1:n
                step!(integ)
                output[j][(m-1)*n + k] = integ.u[i]
            end
        end
    end
    ds.p[p_index] = p0
    return output
end

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
* `direction = 1` : Only crossings of the plane that
  have direction `sign(direction)` are considered to belong to
  the surface of section.
* `u0 = get_state(ds)` : Initial state of the system.
* `Ttr = 0.0` : Transient time to evolve the system before starting
  to compute the PSOS.
* `callback_kwargs = (:abstol=>1e-6)` : Named tuple of keyword arguments passed into
  the `ContinuousCallback` type of `DifferentialEquations`, used to find
  the section. Decreasing the `abstol` makes the section more accurate.
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
function poincaresos(ds::CDS, plane, tfinal = 1000.0;
    direction = +1,
    callback_kwargs = (abstol=1e-6,), Ttr::Real = 0.0, warning = true,
    u0 = get_state(ds), diffeq...)

    _check_plane(plane, dimension(ds))

    if Ttr > 0
        integ = integrator(ds, u0; diffeq...)
        step!(integ, Ttr, true) # step exactly Ttr
        u0 = integ.u
        t0 = integ.t
    else
        t0 = ds.t0
    end

    pcb = psos_callback(plane, direction, callback_kwargs)
    psos_prob = ODEProblem(ds.f, u0, (t0, t0+tfinal), ds.p, callback = pcb)

    solver = _get_solver(diffeq)

    sol = solve(psos_prob, solver; DEFAULT_DIFFEQ_KWARGS...,
    save_start = false, save_end = false,
    save_everystep = false, diffeq...)

    warning && length(sol.u) == 0 && warn(PSOS_ERROR)

    return Dataset(sol.u)
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

function psos_callback(plane::Tuple{Int, <:Number}, direction, callback_kwargs)
    j, offset = plane
    # Prepare callback:
    s = sign(direction)
    cond = (u, t, integrator) -> s*(u - offset)
    affect! = (integrator) -> nothing

    cb = DiffEqBase.ContinuousCallback(cond, affect!, nothing; callback_kwargs...,
    save_positions = (true,false), idxs = j)
end

function psos_callback(plane::AbstractVector, direction, callback_kwargs)
    b = plane[end]
    a = plane[1:end-1]
    # Prepare callback:
    s = sign(direction)
    cond = (u, t, integrator) -> s*(dot(u, a) - b)
    affect! = (integrator) -> nothing

    cb = DiffEqBase.ContinuousCallback(cond, affect!, nothing; callback_kwargs...,
    save_positions = (true,false))
end

const PSOS_ERROR =
"the Poincaré surface of section did not have any points!"

function poincaresos2(ds::CDS{IIP, S, D}, plane, tfinal = 1000.0;
    direction = +1, Ttr::Real = 0.0, warning = true,
    u0 = get_state(ds), diffeq...) where {IIP, S, D}

    integ = integrator(ds, u0; diffeq...)

    planecrossing = PlaneCrossing{D}(plane, direction > 0 )

    psos = _poincaresos(integ, planecrossing, tfinal, Ttr, warning)
    warning && length(psos) == 0 && warn(PSOS_ERROR)
    return psos
end

struct PlaneCrossing{D, P}
    plane::P
    dir::Bool
end
PlaneCrossing{D}(p::P, b) where {P} = PlaneCrossing{D, P}(p, b)
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

function _poincaresos(
    integ, planecrossing::PlaneCrossing{D}, tfinal, Ttr, atol = 1e-6, xrtol = 1e-6, 
    ) where {D}

    Ttr != 0 && step!(integ, Ttr)

    psos = Dataset{D, eltype(integ.u)}()

    f = (t) -> planecrossing(integ(t))
    side = planecrossing(integ.u)

    while integ.t < tfinal + Ttr
        while planecrossing(integ.u) < 0
            integ.t > tfinal + Ttr && break
            step!(integ)
        end
        while planecrossing(integ.u) > 0
            integ.t > tfinal + Ttr && break
            step!(integ)
        end
        # I am now guaranteed to have `t` in negative and `tprev` in positive
        tcross = find_zero(f, (integ.tprev, integ.t), ZERO_FINDER,
                           xatol = 0, rtol = 0, xrtol = xrtol, atol = atol)

        ucross = integ(tcross)
        push!(psos.data, SVector{D}(ucross))

    end
    return psos
end

const ZERO_FINDER = FalsePosition()

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
* `direction`, `diff_eq_kwargs`, `callback_kwargs`, `Ttr` : Passed into
  [`poincaresos`](@ref).
* `printparams::Bool = false` : Whether to print the parameter used during computation
  in order to keep track of running time.
* `ics = [get_state(ds)]` : Collection of initial conditions.
  For every `state ∈ ics` a PSOS will be produced.
* `warning = true` : Throw a warning if any Poincaré section was empty.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples.

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
    ds::CDS,
    plane,
    i::Int,
    p_index,
    pvalues;
    tfinal::Real = 100.0,
    ics = [get_state(ds)],
    direction = +1,
    callback_kwargs = (abstol=1e-6,),
    printparams = false,
    warning = true,
    Ttr = 0.0,
    diffeq...)

    _check_plane(plane, dimension(ds))

    p0 = ds.p[p_index]
    output = Vector{Vector{eltype(get_state(ds))}}(undef, length(pvalues))

    # Prepare callback problem
    pcb = psos_callback(plane, direction, callback_kwargs)
    psos_prob = ODEProblem(
        ds.f, get_state(ds), (ds.t0 + Ttr, ds.t0 + Ttr + tfinal),
        ds.p, callback = pcb)

    solver = _get_solver(diffeq)
    psosinteg = init(psos_prob, solver; DEFAULT_DIFFEQ_KWARGS..., save_start = false,
    save_end = false, save_everystep=false, diffeq..., save_idxs = [i])

    integ = integrator(ds; Tfinal = Ttr, diffeq...)

    for (n, p) in enumerate(pvalues)
        psosinteg.p[p_index] = p
        integ.p[p_index] = p
        printparams && println("parameter = $p")

        for (m, st) in enumerate(ics)

            if Ttr > 0
                reinit!(integ, st)
                step!(integ, Ttr, true)
                st0 = integ.u
            else
                st0 = st
            end

            reinit!(psosinteg, st0)

            solve!(psosinteg)

            solu = psosinteg.sol.u
            warning && length(solu) == 0 && warn(
            "For parameter $p and initial condition index $m $PSOS_ERROR")

            if length(solu) != 0
                out = [a[1] for a in solu]
            else
                out = eltype(get_state(ds))[]
            end

            if m == 1
                output[n] = out
            else
                append!(output[n], out)
            end



        end
    end
    # Reset the parameter of the system:
    ds.p[p_index] = p0
    return output
end

# ds = Systems.shinriki([-2, 0, 0.2])
#
# pvalues = linspace(19,22,21)
# i = 1
# tfinal = 1000.0
# p_index = 1
# plane = (2, 2.0)# psos at variable 2 with offset = 0
# Ttr = 500.0
#
#
# output = produce_orbitdiagram(ds, plane, i, p_index, pvalues; tfinal = tfinal,
# Ttr = 200.0, direction = -1, printparams = true)
#
# figure()
# for (j, p) in enumerate(pvalues)
#     plot(p .* ones(output[j]), output[j], lw = 0,
#     marker = "o", ms = 0.5, color = "black")
# end
# xlabel("\$R_1\$"); ylabel("\$V_1\$")
#
#
# diffeq = (a = 3,)
