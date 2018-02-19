using OrdinaryDiffEq, DiffEqBase
export poincaresos, orbitdiagram, produce_orbitdiagram

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


const PSOS_ERROR =
"The Poincaré surface of section did not have any points!"*
" Check: If the variable chosen crosses the `offset`,"*
" if the parameters used make it happen"*
" and be sure you have integrated for enough time."


"""
    poincaresos(ds::ContinuousDynamicalSystem, j, tfinal = 100.0; kwargs...)
Calculate the Poincaré surface of section (also called Poincaré map) [1, 2]
of the given system on the plane of the `j`-th variable of the system.
The system is evolved for total time of `tfinal`.

Returns a [`Dataset`](@ref) of the points that are on the surface of section.

## Keyword Arguments
* `direction = 1` and `offset = 0.0` : The surface of section is defined as the
  (hyper-) plane where `state[j] = offset`. Only crossings of the plane that
  have direction `sign(direction)` are considered to belong to the surface of section.
* `Ttr = 0.0` : Transient time to evolve the system before starting
  to compute the PSOS.
* `diff_eq_kwargs` : See [`trajectory`](@ref).
* `callback_kwargs = Dict(:abstol=>1e-6)` : Keyword arguments passed into
  the `ContinuousCallback` type of `DifferentialEquations`, used to find
  the section. The option `callback_kwargs[:idxs] = j` is enforced. Decreasing
  the `abstol` makes the section more accurate.

## References
[1] : H. Poincaré, *Les Methods Nouvelles de la Mécanique Celeste*,
Paris: Gauthier-Villars (1892)

[2] : M. Tabor, *Chaos and Integrability in Nonlinear Dynamics: An Introduction*,
§4.1, in pp. 118-126, New York: Wiley (1989)

See also [`orbitdiagram`](@ref), [`produce_orbitdiagram`](@ref).
"""
function poincaresos(ds::CDS, j::Int, tfinal = 100.0;
    direction = +1, offset::Real = 0,
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    callback_kwargs = Dict(:abstol=>1e-6),
    Ttr::Real = 0.0)

    pcb = psos_callback(j, direction, offset, callback_kwargs)

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
    length(sol.u) == 0 && error(PSOS_ERROR)

    return Dataset(sol.u)
end

function psos_callback(j, direction, offset,
    callback_kwargs)

    # Prepare callback:
    s = sign(direction)
    cond = (u, t, integrator) -> s*(u - offset)
    affect! = (integrator) -> nothing

    cb = DiffEqBase.ContinuousCallback(cond, affect!, nothing; callback_kwargs...,
    save_positions = (true,false), idxs = j)
end



"""
    produce_orbitdiagram(ds::ContinuousDynamicalSystem, j, i,
                         p_index, pvalues; kwargs...)
Produce an orbit diagram (also called bifurcation diagram)
for the `i`-th variable of the given continuous
system by computing Poincaré surfaces of section
of the `j`-th variable of the system for the given parameter values.

## Keyword Arguments
* `direction`, `offset`, `diff_eq_kwargs`, `callback_kwargs`, `Ttr` : Passed into
  [`poincaresos`](@ref).
* `printparams::Bool = false` : Whether to print the parameter used during computation
  in order to keep track of running time.
* `ics = [state(ds)]` : Collection of initial conditions. For every `state ∈ ics` a PSOS
  will be produced.

## Description
For each parameter, a PSOS reduces the system from a flow to a map. This then allows
the formal computation of an "orbit diagram" for one of the `i ≠ j` variables
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
    j::Int,
    i::Int,
    p_index,
    pvalues;
    tfinal::Real = 100.0,
    ics = [state(ds)],
    direction = +1,
    offset::Real = 0,
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    callback_kwargs = Dict(:abstol=>1e-6),
    printparams::Bool = false,
    Ttr::Real = 0.0)

    p0 = ds.prob.p[p_index]
    output = Vector{Vector{eltype(state(ds))}}(length(pvalues))
    solver, newkw = DynamicalSystemsBase.extract_solver(diff_eq_kwargs)

    # Prepare callback problem
    pcb = psos_callback(j, direction, offset, callback_kwargs)
    psos_prob = ODEProblem(
        ds.prob.f, state(ds), (inittime(ds), inittime(ds)+tfinal),
        ds.prob.p, callback = pcb)

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
            length(solu) == 0 && error(PSOS_ERROR)

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
