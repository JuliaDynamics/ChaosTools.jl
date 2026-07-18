export orbitdiagram
import ProgressMeter

"""
    orbitdiagram(ds::DynamicalSystem, i, p_index, pvalues; kwargs...) → od
    orbitdiagram(ds::DynamicalSystem, i, pcurve; kwargs...) → od

Compute the orbit diagram (sometimes wrongly called bifurcation diagram)
of the given dynamical system,
saving the `i` variable(s) for parameter values `pvalues`.
The `p_index` specifies
which parameter to change via `set_parameter!(ds, p_index, pvalue)`.
Alternative, you can provide an arbitrary curve in parameter space
(same as in global continuation of Attractors.jl), by providing a
vector of dictionaries mapping parameter indices to values.

## Description

An orbit diagram is simply a collection of the last `n` states of `ds` as `ds` is
evolved. This is done for each parameter value.
It works for any kind of `DynamicalSystem`, although it mostly makes sense with
one of `DeterministicIteratedMap, StroboscopicMap, PoincareMap`.

`i` can be an integer or a vector of anything acceptable by `observe_state`.
If `i` is `Int`, `od` is a vector of vectors.
Else `od` is a vector of vectors of vectors.
Each entry of `od` are the points at each parameter curve entry,
so that `length(od) == length(pcurve)` and `length(od[j]) == n, ∀ j`.

## Keyword arguments

- `n::Int = 100`: Amount of points to save for each parameter value.
- `Δt = 1`: Stepping time between saving points.
- `u0 = nothing`: Specify an initial state. If `nothing`, the previous state after each
  parameter is used to seed the new initial condition at the new parameter
  (with the very first state being the system's state). This makes convergence to the
  attractor faster, necessitating smaller `Ttr`. Otherwise `u0` can be a standard state,
  or a vector of states, so that a specific state is used for each parameter.
- `Ttr::Int = 10`: Each orbit is evolved for `Ttr` first before saving output.
- `ulims = (-Inf, Inf)`: only record system states within `ulims`
  (only valid if `i isa Int`). Iteration continues until `n` states fall within `ulims`.
- `show_progress = false`: Display a progress bar (counting the parameter values).
- `periods = nothing`: Only valid if `ds isa StroboscopicMap`. If given, it must be a
  a container with same layout as `pvalues`. Provides a value for the `period` for each
  parameter value. Useful in case the orbit diagram is produced versus a driving frequency.
"""
function orbitdiagram(ds::DynamicalSystem, idxs, p_index, pvalues; kw...)
    pcurve = [[p_index => p] for p in pvalues]
    return orbitdiagram(ds, idxs, pcurve; kw...)
end

function orbitdiagram(
        ds::DynamicalSystem, idxs, pcurve::AbstractVector; n::Int = 100, kwargs...
    )
    output = initialize_od_output(current_state(ds), idxs, n, length(pcurve))
    fill_orbitdiagram!(output, ds, idxs, pcurve; n, kwargs...)
    return output
end

function initialize_od_output(u, i::Int, n, l)
    return [zeros(eltype(u), n) for _ in 1:l]
end
function initialize_od_output(u, i::AbstractVector, n, l)
    return [[zeros(eltype(u), length(i)) for _ in 1:n] for _ in 1:l]
end

function fill_orbitdiagram!(output, ds::DynamicalSystem, idxs, pcurve;
        n::Int = 100, Ttr::Int = 10, u0 = nothing, Δt = 1, ulims = nothing,
        show_progress = false, periods = nothing,
    )

    # Sanity check
    if u0 isa AbstractVector{<:AbstractVector}
        length(u0) != length(pcurve) && error("Need length(u0) == length(pcurve)")
    end

    progress = ProgressMeter.Progress(length(pcurve);
        desc = "Orbit diagram: ", dt = 1.0, enabled = show_progress
    )

    for (j, pidx) in enumerate(eachindex(pcurve))
        # reset to current parameter/state/Ttr
        set_parameters!(ds, pcurve[pidx])
        if !isnothing(periods) && ds isa StroboscopicMap
            set_period!(ds, periods[pidx])
        end
        st = orbitdiagram_starting_state(u0, j)
        reinit!(ds, st)
        Ttr > 0 && step!(ds, Ttr)
        # collect `n` states
        if idxs isa AbstractVector
            @inbounds for k in 1:n
                step!(ds, Δt)
                for (ii, i) in enumerate(idxs)
                    output[j][k][ii] = observe_state(ds, i)
                end
            end
        else
            k = 1
            while k ≤ n
                step!(ds, Δt)
                u = current_state(ds)[i]
                @inbounds if ulims[1] ≤ u ≤ ulims[2]
                    output[j][k] = u
                    k += 1
                end
            end
        end
        ProgressMeter.update!(progress, j)
    end
    return output
end

function orbitdiagram_starting_state(u0, j)
    if u0 isa AbstractVector{<:AbstractVector}
        u0[j]
    else
        u0 # this can also be `nothing` (default), which resumes from prior state
    end
end
