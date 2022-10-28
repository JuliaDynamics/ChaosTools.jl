export produce_orbitdiagram

"""
```julia
produce_orbitdiagram(
	ds::ContinuousDynamicalSystem, plane, i::Int, p_index, pvalues; kwargs...
)
```
Produce an orbit diagram (sometimes wrongly called bifurcation diagram)
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
* `direction, warning, Ttr, rootkw, diffeq` :
  Propagated into [`poincaresos`](@ref).
* `u0 = nothing` : Specify an initial state. If `nothing`, the previous state after each
  parameter is used to seed the new initial condition at the new parameter
  (with the very first state being the system's state). This makes convergence to the
  attractor faster, necessitating smaller `Ttr`. Otherwise `u0` can be a standard state,
  or a vector of states, so that a specific state is used for each parameter.

## Description
For each parameter, a PSOS reduces the system from a flow to a map. This then allows
the formal computation of an "orbit diagram" for the `i` variable
of the system, just like it is done in [`orbitdiagram`](@ref).

The parameter change is done as `p[p_index] = value` taking values from `pvalues`
and thus you must use a parameter container that supports this
(either `Array`, `LMArray`, dictionary or other).

See also [`poincaresos`](@ref), [`orbitdiagram`](@ref).
"""
function produce_orbitdiagram(
        ds::CDS{IIP, S, D}, plane, idxs, p_index, pvalues;
        tfinal::Real = 100.0, direction = -1, printparams = false, warning = true,
        Ttr = 0.0, u0 = nothing, rootkw = (xrtol = 1e-6, atol = 1e-6),
        diffeq = NamedTuple(), kwargs...
    ) where {IIP, S, D}

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

    i = typeof(idxs) <: Int ? idxs : SVector{length(idxs), Int}(idxs...)

    _check_plane(plane, D)
    typeof(u0) <: Vector{<:AbstractVector} && @assert length(u0)==length(p)
    integ = integrator(ds; diffeq)
    planecrossing = PlaneCrossing(plane, direction > 0)
    p0 = ds.p[p_index]
    output = Vector{typeof(ds.u0[i])}[]
	plane_distance = (t) -> planecrossing(integ(t))

    for (n, p) in enumerate(pvalues)
        integ.p[p_index] = p
        printparams && println("parameter = $p")
        if typeof(u0) <: Vector{<:AbstractVector}
            st = u0[n]
        elseif isnothing(u0)
            st = integ.u
        else
            st = u0
        end
        reinit!(integ, st)
		Ttr != 0 && step!(integ, Ttr)
		data = _poincaresos(integ, plane_distance, planecrossing, tfinal+Ttr, i, rootkw)
		push!(output, data)
        warning && length(output[end]) == 0 && @warn "For parameter $p $PSOS_ERROR"
    end
    # Reset the parameter of the system:
    ds.p[p_index] = p0
    return output
end