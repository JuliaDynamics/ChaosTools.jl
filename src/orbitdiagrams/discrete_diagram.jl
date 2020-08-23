export orbitdiagram

"""
    orbitdiagram(ds::DiscreteDynamicalSystem, i, p_index, pvalues; kwargs...)
Compute the orbit diagram (also called bifurcation diagram) of the given system,
saving the `i` variable(s) for parameter values `pvalues`. The `p_index` specifies
which parameter of the equations of motion is to be changed.

`i` can be `Int` or `AbstractVector{Int}`.
If `i` is `Int`, returns a vector of vectors. Else
it returns vectors of vectors of vectors.
Each entry are the points at each parameter value.

## Keyword Arguments
* `Ttr::Int = 1000` : Transient steps;
  each orbit is evolved for `Ttr` first before saving output.
* `n::Int = 100` : Amount of points to save for each initial condition.
* `dt = 1` : Stepping time. Changing this will give you the orbit diagram of
  the `dt` order map.
* `u0 = get_state(ds)` : Initial condition. Besides a vector you can also give
  a vector of vectors such that `length(u0) == length(pvalues)`. Then each parameter
  has a different initial condition.
* `ulims = (-Inf, Inf)` : only record system states within `ulims`
  (only valid if `i isa Int`).

See also [`poincaresos`](@ref) and [`produce_orbitdiagram`](@ref).
"""
function orbitdiagram(
        ds::DDS{IIP, S, D}, idxs, p_index, pvalues;
        n::Int = 100, Ttr::Int = 1000, u0 = get_state(ds), dt = 1, ulims = nothing
    ) where {IIP, S, D}

    p0 = ds.p[p_index]
    if D == 1
        idxs != 1 &&
        error("You have a 1D system and yet you gave `i=$i`. What's up with that!?")
    end

    typeof(u0) <: Vector{<:AbstractVector} && @assert length(u0)==length(pvalues)
    i = idxs isa Int ? idxs : SVector{length(idxs), Int}(idxs...)
    !isnothing(ulims) && i isa SVector && error("If `i` is a vector, you can't use `ulims`.")

    output = _initialize_od_output(ds.u0, i, n, length(pvalues))
    integ = integrator(ds)
    _fill_orbitdiagram!(output, integ, i, pvalues, p_index, n, Ttr, u0, dt, ulims)
    ds.p[p_index] = p0
    return output
end

function _initialize_od_output(u::S, i::Int, n, l) where {S}
    output = [zeros(eltype(S), n) for k in 1:l]
end
function _initialize_od_output(u::S, i::SVector, n, l) where {S}
    s = u[i]
    output = [Vector{typeof(s)}(undef, n) for k in 1:l]
end

function _fill_orbitdiagram!(output, integ, i, pvalues, p_index, n, Ttr, u0, dt, ulims)
    isavector = i isa AbstractVector
    for (j, p) in enumerate(pvalues)
        integ.p[p_index] = p
        st = (u0 isa Vector{ <: _innertype(integ.u) }) ? u0[j] : u0
        reinit!(integ, st)
        step!(integ, Ttr)
        if isavector || isnothing(ulims) # if-clause gets compiled away (I hope)
            @inbounds for k in 1:n
                step!(integ, dt)
                output[j][k] = integ.u[i]
            end
        else
            k = 1
            while k ≤ n
                step!(integ, dt)
                u = integ.u[i]
                @inbounds if ulims[1] ≤ u ≤ ulims[2]
                    output[j][k] = u
                    k += 1
                end
            end
        end
    end
    return output
end

_innertype(::Real) = Real
_innertype(::AbstractVector) = AbstractVector
