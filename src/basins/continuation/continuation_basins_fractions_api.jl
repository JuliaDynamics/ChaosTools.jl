export basins_fractions_continuation

"""
    AttractorMatcher
Supertype governing how attractors are matched in [`basins_fractions_continuation`](@ref).
Some matching methods can only be used with some specific [`AttractorMapper`](@ref) types.
At the moment we have:

- [`MatchByDistance`](@ref) which can be used with [`AttractorsViaRecurrences`](@ref).
"""
abstract type AttractorMatcher end


# Design of the API:
function basins_fractions_continuation(mapper::AttractorMapper, maching_method::AttractorMatcher, parameter; kwargs...)
    # code
end
# Return values:
fracs <: Vector{Dict{Int, Float64}}
attractor_summary <: Vector{<:Any}
# some info on the attractors which depends on the method.
# Could be the actual attractors (possible in recurrences),
# or the attractor features...?


# Decide the kind of dynamical system
α = 0.2; ω = 1.0; d = 0.3
ma = Systems.magnetic_pendulum(; α, ω, d)
proj = projected_integrator(ma, [1,2], [0,0])
# Decide the attractor mapping
gx = gy = range(-5, 5; length = 1500)
mapper = AttractorsViaRecurrences(proj, (gx, gy))
# Decide how to match attractors
matching_method = StateSpaceDistance(kwargs...)
# What parameter to continue over
parameter = (0:0.01:1, 1) # index, range
# Call function
basins_fractions_continuation(mapper, matching_method, parameter; ...)

# Make matching subtype an "AttractorMatcher" interface.
# And perhaps teh `threshold` doesn't need to be given to the matchers,
# but only to the matching function? And the matchers provide only the
# similarity measure.
