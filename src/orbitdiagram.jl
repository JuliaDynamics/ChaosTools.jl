using OrdinaryDiffEq, DiffEqBase
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
* `ics = [state(ds)]` : container of initial conditions that
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
    n::Int = 100, Ttr::Int = 1000, ics = [state(ds)]) where {IIP, S, D}

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
    return output
end

#####################################################################################
#                               Poincare Section                                    #
#####################################################################################
const PSOS_ERROR =
"the Poincaré surface of section did not have any points!"

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



"""
    poincaresos(ds::ContinuousDynamicalSystem, plane, tfinal = 1000.0; kwargs...)
Calculate the Poincaré surface of section (also called Poincaré map) [1, 2]
of the given system with the given `plane`.
The system is evolved for total time of `tfinal`.

If the state of the system is ``\\mathbf{u} = (u_1, \\ldots, u_D)`` then the equation for
the hyperplane is
```math
a_1u_1 + \\dots + a_Du_D = \\mathbf{a}\\cdot\\mathbf{u}=b
```
where ``\\mathbf{a}, b`` are the parameters that define the hyperplane.

In code, `plane` can be either:

* A `Tuple{Int, <: Number}`, like `(j, r)` : the hyperplane is defined
  as when the `j` variable of the system crosses the value `r`.
* An `AbstractVector` of length `D+1`. The first `D` elements of the
  vector correspond to ``\\mathbf{a}`` while the last element is ``b``.

Returns a [`Dataset`](@ref) of the points that are on the surface of section.

## Keyword Arguments
* `direction = 1` : Only crossings of the plane that
  have direction `sign(direction)` are considered to belong to the surface of section.
* `Ttr = 0.0` : Transient time to evolve the system before starting
  to compute the PSOS.
* `diff_eq_kwargs` : See [`trajectory`](@ref).
* `callback_kwargs = Dict(:abstol=>1e-6)` : Keyword arguments passed into
  the `ContinuousCallback` type of `DifferentialEquations`, used to find
  the section. The option `callback_kwargs[:idxs] = j` is enforced. Decreasing
  the `abstol` makes the section more accurate.
* `warning = true` : Throw a warning if the Poincaré section was empty.

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
    direction = +1, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    callback_kwargs = Dict(:abstol=>1e-6), Ttr::Real = 0.0, warning = true)

    _check_plane(plane, dimension(ds))

    pcb = psos_callback(plane, direction, callback_kwargs)

    if Ttr > 0
        integ = integrator(ds)
        step!(integ, Ttr, true) # step exactly Ttr
        u0 = integ.u
        t0 = integ.t
    else
        u0 = state(ds)
        t0 = inittime(ds)
    end

    psos_prob = ODEProblem(ds.prob.f, u0, (t0, t0+tfinal), ds.prob.p, callback = pcb)

    solver, newkw = DynamicalSystemsBase.extract_solver(diff_eq_kwargs)

    sol = solve(psos_prob, solver; newkw..., save_start = false, save_end = false,
    save_everystep = false)

    warning && length(sol.u) == 0 && warn(PSOS_ERROR)

    return Dataset(sol.u)
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
* `ics = [state(ds)]` : Collection of initial conditions. For every `state ∈ ics` a PSOS
  will be produced.
* `warning = true` : Throw a warning if any Poincaré section was empty.

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
    ics = [state(ds)],
    direction = +1,
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    callback_kwargs = Dict(:abstol=>1e-6),
    printparams::Bool = false,
    warning = true,
    Ttr::Real = 0.0)

    p0 = ds.prob.p[p_index]
    output = Vector{Vector{eltype(state(ds))}}(length(pvalues))
    solver, newkw = DynamicalSystemsBase.extract_solver(diff_eq_kwargs)

    # Prepare callback problem
    pcb = psos_callback(plane, direction, callback_kwargs)
    psos_prob = ODEProblem(
        ds.prob.f, state(ds), (inittime(ds), inittime(ds)+tfinal),
        ds.prob.p, callback = pcb)

    # TODO: Save only the index requested
    psosinteg = init(psos_prob, solver; newkw..., save_start = false,
    save_end = false, save_everystep=false)

    integ = integrator(ds; diff_eq_kwargs = diff_eq_kwargs)

    for (n, p) in enumerate(pvalues)
        psosinteg.p[p_index] = p
        integ.p[p_index] = p
        printparams && println("parameter = $p")

        for (m, st) in enumerate(ics)

            if Ttr > 0
                reinit!(integ, st)
                step!(integ, Ttr)
                st0 = integ.u
            else
                st0 = st
            end

            reinit!(psosinteg, st0)

            solve!(psosinteg)

            solu = psosinteg.sol.u
            warning && length(solu) == 0 && warn(
            "For parameter $p and initial condition index $m $PSOS_ERROR")

            out = [a[i] for a in solu]

            if m == 1
                output[n] = out
            else
                append!(output[n], out)
            end

        end
    end
    # Reset the parameter of the system:
    ds.prob.p[p_index] = p0
    return output
end
