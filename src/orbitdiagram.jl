using OrdinaryDiffEq
export poincaresos, orbitdiagram, produce_orbitdiagram

"""
    orbitdiagram(ds::DiscreteDynamicalSystem, i::Int, parameter::Symbol, pvalues;
    n::Int = 100, Ttr::Int = 1000, ics = [state(ds)]) -> output
Compute the orbit diagram (also called bifurcation diagram) of the given system
for the `i`-th variable for parameter values `pvalues`. The `parameter` specifies
which parameter of the equations of motion is to be changed.

This function assumes that you have created the `ContinuousDS` using functors; see the
[official documentation](https://juliadynamics.github.io/DynamicalSystems.jl/latest/)
for more.

## Keyword Arguments
* `ics = [state(ds)]` : container of initial conditions that
  are used at each parameter value
  to evolve orbits.
* `Ttr::Int = 1000` : Transient steps;
  each orbit is evolved for `Ttr` first before saving output.
* `n::Int = 100` : Amount of points to save for each initial condition.

## Description
The method works by computing orbits at each parameter value in `pvalues` for each
initial condition in `ics`. The symbol
of the parameter is used to set `ds.eom.parameter` or `ds.eom!.parameter`.

The returned `output` is a vector of vectors. `output[j]` are the orbit points of the
`i`-th variable of the system, at parameter value `pvalues[j]`.

See also [`poincaresos`](@ref) and [`produce_orbitdiagram`](@ref).
"""
function orbitdiagram(ds::DiscreteDynamicalSystem, i::Int, parameter::Symbol, pvalues;
    n::Int = 100, Ttr::Int = 1000, ics = [state(ds)])

    if typeof(ds) <: DiscreteDS1D
        i != 1 &&
        error("You have a 1D system and yet you gave `i=$i`. What's up with that!?")
    end

    T = eltype(state(ds))
    output = Vector{Vector{T}}(length(pvalues))

    for (j, p) in enumerate(pvalues)

        if isa(ds, Union{DiscreteDS, DiscreteDS1D})
            setfield!(ds.eom, parameter, p)
        else
            setfield!(ds.eom!, parameter, p)
        end

        for (m, state) in enumerate(ics)
            if isa(ds, Union{DiscreteDS, DiscreteDS1D})
                ds.state = evolve(ds, Ttr, state)
            else
                ds.state .= evolve(ds, Ttr, state)
            end

            if m == 1
                output[j] = trajectory(ds, n)[:, i]
            else
                append!(output[j], trajectory(ds, n)[:, i])
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
    poincaresos(ds::ContinuousDS, j, tfinal = 100.0; kwargs...)
Calculate the Poincaré surface of section (also called Poincaré map) [1, 2]
of the given system on the plane of the `j`-th variable of the system.
The system is evolved for total time of `tfinal`.

Returns a [`Dataset`](@ref) of the points that are on the surface of section.

This function assumes that you have created the `ContinuousDS` using functors; see the
[official documentation](https://juliadynamics.github.io/DynamicalSystems.jl/latest/)
for more.

## Keyword Arguments
* `direction = 1` and `offset = 0.0` : The surface of section is defined as the
  (hyper-) plane where `state[j] = offset`. Only crossings of the plane that
  have direction `sign(direction)` are considered to belong to the surface of section.
* `Ttr = 0.0` : Transient time to evolve the system before starting
  to compute the PSOS.
* `diff_eq_kwargs = Dict()` : See [`trajectory`](@ref).
* `callback_kwargs = Dict(:abstol=>1e-9)` : Keyword arguments passed into
  the `ContinuousCallback` type of `DifferentialEquations`. The option
  `callback_kwargs[:idxs] = j` is enforced.

## References
[1] : H. Poincaré, *Les Methods Nouvelles de la Mécanique Celeste*,
Paris: Gauthier-Villars (1892)

[2] : M. Tabor, *Chaos and Integrability in Nonlinear Dynamics: An Introduction*,
§4.1, in pp. 118-126, New York: Wiley (1989)

See also [`orbitdiagram`](@ref), [`produce_orbitdiagram`](@ref).
"""
function poincaresos(ds::ContinuousDynamicalSystem, j::Int, tfinal = 100.0;
    direction = +1, offset::Real = 0,
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    callback_kwargs = Dict{Symbol, Any}(:abstol=>1e-9),
    Ttr::Real = 0.0)

    pcb = psos_callback(j, direction, offset, callback_kwargs)

    if Ttr > 0
        u0 = evolve(ds, Ttr, diff_eq_kwargs = diff_eq_kwargs)
    else
        u0 = state(ds)
    end

    extra_kw = Dict(:save_start=>false, :save_end=>false)

    psos_prob = ODEProblem(ds, tfinal, u0, pcb)

    solu, tvec = get_sol(psos_prob, diff_eq_kwargs, extra_kw)
    length(solu) == 0 && error(PSOS_ERROR)

    return Dataset(solu)
end

function psos_callback(j, direction = +1, offset::Real = 0,
    callback_kwargs = Dict{Symbol, Any}(:abstol=>1e-9))

    # Prepare callback:
    s = sign(direction)
    cond = (t,u,integrator) -> s*(u - offset)
    affect! = (integrator) -> nothing

    cb = DiffEqBase.ContinuousCallback(cond, affect!, nothing; callback_kwargs...,
    save_positions = (true,false), idxs = j)
end





"""
    produce_orbitdiagram(ds::ContinuousDynamicalSystem, j, i,
                         parameter::Symbol, pvalues; kwargs...)
Produce an orbit diagram (also called bifurcation diagram)
for the `i`-th variable of the given continuous
system by computing Poincaré surfaces of section
of the `j`-th variable of the system for the given parameter values.

This funtion assumes that you have created the `ContinuousDS` using functors; see the
[official documentation](https://juliadynamics.github.io/DynamicalSystems.jl/latest/)
for more.

## Keyword Arguments
* `direction`, `offset`, `diff_eq_kwargs`, `callback_kwargs`, `Ttr` : Passed into
  [`poincaresos`](@ref).
* `printparams::Bool = false` : Whether to print the parameter used during computation
  in order to keep track of running time.
* `ics = [state(ds)]` : Collection of initial conditions. For every `state ∈ ics` a PSOS
  will be produced.

## Description
`parameter` is a symbol that indicates which
parameter of `ds.prob.f` should be updated.
`pvalues` is a collection with all the parameter values that the orbit diagram
will be computed for.

For each parameter, a PSOS reduces the system from a flow to a map. This then allows
the formal computation of an "orbit diagram" for one of the `i ≠ j` variables
of the system, just like it is done in [`orbitdiagram`](@ref).

The returned `output` is a vector of vectors. `output[k]` are the
"orbit diagram" points of the
`i`-th variable of the system, at parameter value `pvalues[k]`.

## Performance Notes
The total amount of PSOS produced will be `length(ics)*length(pvalues)`.

See also [`poincaresos`](@ref), [`orbitdiagram`](@ref).
"""
function produce_orbitdiagram(
    ds::ContinuousDynamicalSystem,
    j::Int,
    i::Int,
    parameter::Symbol,
    pvalues;
    tfinal::Real = 100.0,
    ics = [state(ds)],
    direction = +1,
    offset::Real = 0,
    diff_eq_kwargs = Dict(),
    callback_kwargs = Dict{Symbol, Any}(:abstol=>1e-6),
    printparams::Bool = false,
    Ttr::Real = 0.0)

    p0 = getfield(ds.prob.f, parameter)
    T = eltype(ds)
    output = Vector{Vector{T}}(length(pvalues))
    extra_kw = Dict(:save_start=>false, :save_end=>false)

    # Prepare callback problem
    pcb = psos_callback(j, direction, offset, callback_kwargs)
    psos_prob = ODEProblem(ds, tfinal, deepcopy(ds.prob.u0), pcb)

    for (n, p) in enumerate(pvalues)
        # This sets the parameter on both the ds problem
        # as well as the psosprob:
        setfield!(ds.prob.f, parameter, p)
        printparams && println(parameter, " = $p")

        for (m, st) in enumerate(ics)

            Ttr > 0 && (st = evolve(ds, Ttr, st; diff_eq_kwargs = diff_eq_kwargs))

            psos_prob.u0 .= st

            solu, tvec = get_sol(psos_prob, diff_eq_kwargs, extra_kw)
            length(solu) == 0 && error(PSOS_ERROR)

            out = [a[i] for a in solu]

            if m == 1
                output[n] = out
            else
                append!(output[n], out)
            end

        end
    end
    # Reset the field parameter of the system:
    setfield!(ds.prob.f, parameter, p0)
    return output
end
