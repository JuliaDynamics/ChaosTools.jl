export orbitdiagram
import ProgressMeter

"""
    orbitdiagram(ds::DynamicalSystem, i, p_index, pvalues; kwargs...) → od

Compute the orbit diagram (sometimes wrongly called bifurcation diagram)
of the given dynamical system,
saving the `i` variable(s) for parameter values `pvalues`. The `p_index` specifies
which parameter to change via `set_parameter!(ds, p_index, pvalue)`.
Works for any kind of `DynamicalSystem`, although it mostly makes sense with
one of `DeterministicIteratedMap, StroboscopicMap, PoincareMap`.

An orbit diagram is simply a collection of the last `n` states of `ds` as `ds` is
evolved. This is done for each parameter value.

`i` can be `Int` or `AbstractVector{Int}`. If `i` is `Int`, `od` is a vector of vectors.
Else `od` is a vector of vectors of vectors.
Each entry od `od` are the points at each parameter value,
so that `length(od) == length(pvalues)` and `length(od[j]) == n, ∀ j`.

See also [`produce_orbitdiagram`](@ref).

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
function orbitdiagram(
        ds::DynamicalSystem, idxs, p_index, pvalues; n::Int = 100, kwargs...
    )

    i = idxs isa Int ? idxs : SVector{length(idxs), Int}(idxs...)
    output = initialize_od_output(current_state(ds), i, n, length(pvalues))
    fill_orbitdiagram!(output, ds, i, pvalues, p_index; n, kwargs...)
    return output
end

function initialize_od_output(u::S, i::Int, n, l) where {S}
    return [zeros(typeof(u[i]), n) for k in 1:l]
end
function initialize_od_output(u::S, i::SVector, n, l) where {S}
    s = u[i]
    return [Vector{typeof(s)}(undef, n) for k in 1:l]
end

function fill_orbitdiagram!(output, ds::DynamicalSystem, i, pvalues, p_index;
        n::Int = 100, Ttr::Int = 10, u0 = nothing, Δt = 1, ulims = nothing,
        show_progress = false, periods = nothing,
    )

    # Sanity check
    if u0 isa AbstractVector{<:AbstractVector}
        length(u0) != length(pvalues) && error("Need length(u0) == length(pvalues)")
    end

    progress = ProgressMeter.Progress(length(pvalues);
        desc = "Orbit diagram: ", dt = 1.0, enabled = show_progress
    )

    for (j, pidx) in enumerate(eachindex(pvalues))
        p = pvalues[pidx]
        # reset to current parameter/state/Ttr
        set_parameter!(ds, p_index, p)
        if !isnothing(periods) && ds isa StroboscopicMap
            set_period!(ds, periods[pidx])
        end
        st = orbitdiagram_starting_state(u0, j)
        reinit!(ds, st)
        Ttr > 0 && step!(ds, Ttr)
        # collect `n` states
        if i isa AbstractVector || isnothing(ulims) # if-clause gets compiled away
            @inbounds for k in 1:n
                step!(ds, Δt)
                output[j][k] = current_state(ds)[i]
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
    elseif isnothing(u0)
        nothing # `reinit!` accepts `nothing`!
    else
        u0
    end
end
