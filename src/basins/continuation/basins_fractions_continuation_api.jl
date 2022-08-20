export basins_fractions_continuation

# In the end, it is better to have a continuation type that contains
# how to match, because there are other keywords that always go into the
# continuation... Like in the recurrences the keyword of seeds per attractor,
# or in the clustering some other stuff that Max will add...
abstract type BasinsFractionContinuation end

"""
basins_fractions_continuation(continuation::BasinsFractionContinuation, parameter; kwargs...)
Continiation API. TODO: Write it.

Current continiation methods are:
- [`RecurrencesSeedingContinuation`](@ref).
"""
function basins_fractions_continuation(continuation::BasinsFractionContinuation, parameter; kwargs...)
    # code
    # Return values:
    # the fractions
    fracs <: Vector{Dict{Int, Float64}}
    # some info on the attractors which depends on the method.
    # Could be the actual attractors (possible in recurrences),
    # or the attractor features...?
    attractor_summary <: Vector{Dict{Int, <:Any}}
end

function _example(; kwargs...)
    # Decide the kind of dynamical system
    α = 0.2; ω = 1.0; d = 0.3
    ma = Systems.magnetic_pendulum(; α, ω, d)
    proj = projected_integrator(ma, [1,2], [0,0])
    # Decide the attractor mapping
    gx = gy = range(-5, 5; length = 1500)
    mapper = AttractorsViaRecurrences(proj, (gx, gy))
    # Decide how to match attractors
    continuation = RecurrencesSeedingContinuation(mapper, kwargs...)
    # What parameter to continue over
    parameter = (0:0.01:1, 1) # index, range
    # Call function
    basins_fractions_continuation(continuation, parameter; kwargs...)
end

include("match_attractor_ids.jl")
include("continuation_recurrences.jl")