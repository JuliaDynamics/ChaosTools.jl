using OrdinaryDiffEq
export poincaresos, orbitdiagram, produce_orbitdiagram

"""
    orbitdiagram(ds::DiscreteDynamicalSystem, i::Int, parameter::Symbol, pvalues;
    n::Int = 100, Ttr::Int = 1000, ics = [ds.state]) -> output
Compute the orbit diagram (also called bifurcation diagram) of the given system
for the `i`-th variable for parameter values `pvalues`. The `parameter` specifies
which parameter of the equations of motion is to be changed.

## Keyword Arguments
* `ics = [ds.state]` : container of initial conditions that
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

## Example
```julia
using ChaosTools
ds = Systems.standardmap()
i = 2
parameter = :k
pvalues = 0:0.005:2
ics = [0.001rand(2) for m in 1:10]
n = 50
Ttr = 5000
output = orbitdiagram(ds, i, parameter, pvalues; n = n, Ttr = Ttr)

using PyPlot
figure()
for (j, p) in enumerate(pvalues)
    plot(p .* ones(output[j]), output[j], lw = 0,
    marker = "o", ms = 0.5, color = "black")
end
```

See also [`poincaresos`](@ref) and [`produce_orbitdiagram`](@ref).
"""
function orbitdiagram(ds::DiscreteDynamicalSystem, i::Int, parameter::Symbol, pvalues;
    n::Int = 100, Ttr::Int = 1000, ics = [ds.state])

    if typeof(ds) <: DiscreteDS1D
        i != 1 &&
        error("You have a 1D system and yet you gave `i=$i`. What's up with that!?")
    end

    T = eltype(ds.state)
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





"""
    poincaresos(ds::ContinuousDS, j, tfinal = 100.0; kwargs...)
Calculate the Poincaré surface of section (also called Poincaré map) [1,2]
of the given
system on the plane of the `j`-th variable of the system. The system is evolved
for total time of `tfinal`.

Returns a [`Dataset`](@ref) of the points that are on the surface of section.

## Keyword Arguments
* `direction = 1` and `offset = 0.0` : The surface of section is defined as the
  (hyper-) plane where `state[j] = offset`. Only crossings of the plane that
  have direction `sign(direction)` are considered to belong to the surface of section.
* `Ttr = 0.0` : Transient time to evolve the system before starting
  to compute the PSOS.
* `diff_eq_kwargs = Dict()` : See [`trajectory`](@ref)
* `callback_kwargs = Dict(:abstol=>1e-6)` : Keyword arguments passed into
  the `ContinuousCallback` type of `DifferentialEquations`. The option
  `callback_kwargs[:idxs] = j` is enforced.

## References
[1] : H. Poincaré, *Les Methods Nouvelles de la Mécanique Celeste*,
Paris: Gauthier-Villars (1892)

[2] : M. Tabor, *Chaos and Integrability in Nonlinear Dynamics: An Introduction*,
§4.1, in pp. 118-126, New York: Wiley (1989)

See also [`orbitdiagram`](@ref), [`produce_orbitdiagram`](@ref).
"""
function poincaresos(ds::ContinuousDS, j::Int, tfinal = 100.0;
    direction = +1, offset::Real = 0,
    diff_eq_kwargs = Dict(), callback_kwargs = Dict(:abstol=>1e-6),
    Ttr::Real = 0.0)

    # Transient
    if Ttr > 0
        state = evolve(ds, Ttr; diff_eq_kwargs = diff_eq_kwargs)
    else
        state = ds.state
    end

    prob = ODEProblem(ds, tfinal, state)

    # Prepare callback:
    s = sign(direction)
    cond = (t,u,integrator) -> s*(u - offset)
    affect! = (integrator) -> nothing
    cb = DiffEqBase.ContinuousCallback(cond, affect!, nothing; callback_kwargs...,
    save_positions = (true,false), idxs = j)

    solver, newkw = DynamicalSystemsBase.get_solver(diff_eq_kwargs)

    sol = solve(prob, solver; newkw...,
    save_everystep=false, callback = cb, save_start=false, save_end=false)

    if length(sol.u) == 0
        error("The poincare s.o.s. did not have any points!"*
        " Check: If the variable chosen crosses the `offset`,"*
        " if the parameters used make it happen"*
        " and be sure you have integrated for enough time.")
    end

    return Dataset(sol.u)
end





"""
    produce_orbitdiagram(ds::ContinuousDS, j, i, parameter::Symbol, pvalues; kwargs...)
Produce an orbit diagram (also called bifurcation diagram)
for the `i`-th variable of the given continuous
system by computing Poincaré surfaces of section
of the `j`-th variable of the system for the given parameter values.

## Keyword Arguments
* `direction`, `offset`, `diff_eq_kwargs`, `callback_kwargs`, `Ttr` : Passed into
  [`poincaresos`](@ref).
* `printparams::Bool = false` : Whether to print the parameter used during computation
  in order to keep track of running time.
* `ics = [ds.state]` : Collection of initial conditions. For every `state ∈ ics` a PSOS
  will be produced.

## Description
`parameter` is a symbol that indicates which parameter of `ds.eom!` should be updated.
`pvalues` is a collection with all the parameter values that the orbit diagram
will be computed for.

For each parameter, a PSOS reduces the system from a flow to a map. This then allows
the formal computation of an "orbit diagram" for one of the `i ≠ j` variables
of the system, just like it is done in [`orbitdiagram`](@ref).

The returned `output` is a vector of vectors. `output[k]` are the
"orbit diagram" points of the
`i`-th variable of the system, at parameter value `pvalues[k]`.


## Performance Notes
The total amount of PSOS produced will be `length(ics)×length(pvalues)`.

See also [`poincaresos`](@ref), [`orbitdiagram`](@ref).
"""
function produce_orbitdiagram(
    ds::ContinuousDS,
    j::Int,
    i::Int,
    parameter::Symbol,
    pvalues;
    tfinal::Real = 100.0,
    ics = [ds.state],
    direction = +1,
    offset::Real = 0,
    diff_eq_kwargs = Dict(),
    callback_kwargs = Dict{Symbol, Any}(:abstol=>1e-6),
    printparams::Bool = false,
    Ttr::Real = 0.0
    )

    T = eltype(ds.state)
    output = Vector{Vector{T}}(length(pvalues))

    # Prepare callback:
    s = sign(direction)
    cond = (t,u,integrator) -> s*(u - offset)
    affect! = (integrator) -> nothing
    cb = DiffEqBase.ContinuousCallback(cond, affect!, nothing; callback_kwargs...,
    save_positions = (true,false), idxs = j)

    solver, newkw = DynamicalSystemsBase.get_solver(diff_eq_kwargs)

    for (n, p) in enumerate(pvalues)
        setfield!(ds.eom!, parameter, p)
        printparams && println(parameter, " = $p")
        for (m, state) in enumerate(ics)


            if Ttr > 0
                state = evolve(ds, Ttr, state, diff_eq_kwargs = diff_eq_kwargs)
            end

            prob = ODEProblem(ds, tfinal, state)

            sol = solve(prob, solver; diff_eq_kwargs...,
            save_everystep=false, callback = cb, save_start=false, save_end=false)

            if length(sol.u) == 0
                error("The poincare s.o.s. did not have any points!"*
                " Check: If the variable chosen crosses the `offset`,"*
                " if the parameters used make it happen"*
                " and be sure you have integrated for enough time.")
            end

            out = sol[i, :]

            if m == 1
                output[n] = out
            else
                append!(output[n], out)
            end

        end
    end
    return output
end
