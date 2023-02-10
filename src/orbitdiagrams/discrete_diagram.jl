export orbitdiagram
export produce_orbitdiagram

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
so that `length(od) == length(pvalues)`.

## Keyword arguments

* `n::Int = 100`: Amount of points to save for each parameter value.
* `Δt = 1`: Stepping time between saving points.
* `u0 = nothing`: Specify an initial state. If `nothing`, the previous state after each
  parameter is used to seed the new initial condition at the new parameter
  (with the very first state being the system's state). This makes convergence to the
  attractor faster, necessitating smaller `Ttr`. Otherwise `u0` can be a standard state,
  or a vector of states, so that a specific state is used for each parameter.
* `Ttr::Int = 10`: Each orbit is evolved for `Ttr` first before saving output.
* `ulims = (-Inf, Inf)`: only record system states within `ulims`
  (only valid if `i isa Int`). Iteration continues until
  `n` states fall within `ulims`.

See also [`poincaresos`](@ref) and [`produce_orbitdiagram`](@ref).
"""
function orbitdiagram(
        ds::DynamicalSystem, idxs, p_index, pvalues; kwargs...
    )

    i = idxs isa Int ? idxs : SVector{length(idxs), Int}(idxs...)
    !isnothing(ulims) && i isa SVector && error("If `i` is a vector, you can't use `ulims`.")

    output = initialize_od_output(current_state(ds), i, n, length(pvalues))
    fill_orbitdiagram!(output, ds, i, pvalues, p_index; kwargs...)
    return output
end

function initialize_od_output(u::S, i::Int, n, l) where {S}
    return [zeros(eltype(S), n) for k in 1:l]
end
function initialize_od_output(u::S, i::SVector, n, l) where {S}
    s = u[i]
    return [Vector{typeof(s)}(undef, n) for k in 1:l]
end

function fill_orbitdiagram!(output, ds, i, pvalues, p_index;
        n::Int = 100, Ttr::Int = 10, u0 = nothing, Δt = 1, ulims = nothing,
    )
    for (j, p) in enumerate(pvalues)
        set_parameter!(ds, p_index, p)
        st = orbitdiagram_starting_state(u0, j)
        reinit!(ds, st)
        Ttr > 0 && step!(ds, Ttr)
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


"""
    produce_orbitdiagram(ds::CoupledODEs, P, args...; kwargs...)

Shortcut function for producing an orbit diagram for a `CoupledODEs`.
The function simply transforms `ds` to either a [`PoincareMap`](@ref) or a
[`StroboscopicMap`](@ref), based on `P`, and then calls
[`orbitdiagram](@ref)`(map, args...; kwargs...)`.

If `P isa Union{Tuple, Vector}` then a [`PoincareMap`](@ref) is created with
`P` as the `Plane`. If `P isa Real`, then a [`StroboscopicMap`](@ref) is created
with `P` the period.

See [^DatserisParlitz2022] chapter 4 for a discussion on why making a map is useful.

[^DatserisParlitz2022]:
    Datseris & Parlitz 2022, _Nonlinear Dynamics: A Concise Introduction Interlaced with Code_,
    [Springer Nature, Undergrad. Lect. Notes In Physics](https://doi.org/10.1007/978-3-030-91032-7)
"""
function produce_orbitdiagram(ds::CoupledODEs, args...; kwargs...)
    if P isa Union{Tuple, AbstractVector}
        map = PoincareMap(ds, P)
    elseif P isa Real
        map = StroboscopicMap(ds, P)
    else
        error("Unknown type for `P`.")
    end
    return orbitdiagram(map, args...; kwargs...)
end