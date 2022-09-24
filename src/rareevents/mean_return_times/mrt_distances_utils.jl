# Utility functions for **signed** distance definitions used in mean return time estimation
# if ε is Real, we have Euclidean distance. If it is Vector, we have Chebyshev.
function check_εs_sorting(εs, L)
    correct = if εs[1] isa Real
        issorted(εs; rev = true)
    elseif εs[1] isa AbstractVector
        @assert all(e -> length(e) == L, εs) "Boxes must have same dimension as state space!"
        for j in 1:L
            if !issorted([εs[i][j] for i in eachindex(εs)]; rev = true)
                return false
            end
        end
        true
    end
    if !correct
        throw(ArgumentError("`εs` must be sorted from largest to smallest ball/box size."))
    end
    return correct
end

# Support both types of sets: balls and boxes
"Return `true` if state is outside ε-ball.
Can also accept pre-computed **minimum** distance (according to metric)."
function isoutside(u, u0, ε::AbstractVector)
    @inbounds for i in eachindex(u)
        abs(u[i] - u0[i]) > ε[i] && return true
    end
    return false
end
isoutside(d::Real, ε::AbstractVector) = d > maximum(ε)
isoutside(u, u0, ε::Real) = euclidean(u, u0) > ε
isoutside(d::Real, ε::Real) = d > ε

"Find the (index of the) outermost ε-ball the trajectory is not in."
function first_outside_index(u::AbstractVector, u₀, εs, E = length(εs))::Int
    i = findfirst(e -> isoutside(u, u₀, e), εs)
    return isnothing(i) ? E+1 : i
end
"Find the (index of the) outermost ε-ball the distance is not in."
function first_outside_index(d::Real, εs, E = length(εs))::Int
    i = findfirst(e -> isoutside(d, e), εs)
    out_idx::Int = isnothing(i) ? E+1 : i
    return out_idx
end

"Return the **signed** distance of state to ε-ball (negative means inside set)."
function signed_distance(u, u0, ε::AbstractVector)
    m = eltype(u)(-Inf)
    @inbounds for i in eachindex(u)
        m2 = abs(u[i] - u0[i]) - ε[i]
        m2 > m && (m = m2)
    end
    return m
end
signed_distance(u, u0, ε::Real) = euclidean(u, u0) - ε
