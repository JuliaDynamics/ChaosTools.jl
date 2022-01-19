# Utilities for re-initializing initial conditions on the grid
"""
    CompleteAndReinit(complete_state, idxs, D)
Helper struct that completes a state and reinitializes the integrator once called
as a function with arguments `f(integ, y)` with `integ` the initialized dynamical
system integrator and `y` the projected initial condition on the grid.
"""
struct CompleteAndReinit{C, Y, R}
    complete_state::C
    u::Vector{Float64} # dummy variable for a state in full state space
    idxs::SVector{Y, Int}
    remidxs::R
end
function CompleteAndReinit(complete_state, idxs, D::Int)
    remidxs = setdiff(1:D, idxs)
    remidxs = isempty(remidxs) ? nothing : SVector(remidxs...)
    u = zeros(D)
    if complete_state isa AbstractVector
        @assert eltype(complete_state) <: Number
    end
    return CompleteAndReinit(complete_state, u, idxs, remidxs)
end
function (c::CompleteAndReinit{<: AbstractVector})(integ, y)
    c.u[c.idxs] .= y
    if !isnothing(c.remidxs)
        c.u[c.remidxs] .= c.complete_state
    end
    reinit!(integ, c.u)
end
function (c::CompleteAndReinit)(integ, y) # case where `complete_state` is a function
    c.u[c.idxs] .= y
    if !isnothing(c.remidxs)
        c.u[c.remidxs] .= c.complete_state(y)
    end
    reinit!(integ, c.u)
end



@generated function generate_ic_on_grid(grid::NTuple{B, T}, ind) where {B, T}
    gens = [:(grid[$k][ind[$k]]) for k=1:B]
    quote
        Base.@_inline_meta
        @inbounds return SVector{$B, Float64}($(gens...))
    end
end
