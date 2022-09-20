export basins_fractions_continuation

# In the end, it is better to have a continuation type that contains
# how to match, because there are other keywords that always go into the
# continuation... Like in the recurrences the keyword of seeds per attractor,
# or in the clustering some other stuff that Max will add...
abstract type BasinsFractionContinuation end

"""
    basins_fractions_continuation(continuation::BasinsFractionContinuation, parameter; kwargs...)
Continiation API. TODO: Write it.

Return:
1. `fracs <: Vector{Dict{Int, Float64}}`. The fractions of basins of attraction.
   `fracs[i]` is a dictionary mapping attractor IDs to their basin fraction
   at the `i`-th parameter.
2. `attractors_info <: Vector{Dict{Int, <:Any}}`. Information about the attractors.
   `attractors_info[i]` is a dictionary mapping attractor ID to information about the
   attractor. The type of information stored depends on the chosen continuation method.

Current continiation methods are:
- [`RecurrencesSeedingContinuation`](@ref).
"""
function basins_fractions_continuation end

include("match_attractor_ids.jl")
include("continuation_recurrences.jl")