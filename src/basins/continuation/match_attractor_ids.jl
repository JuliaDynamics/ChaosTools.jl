include("dict_utils.jl")
###########################################################################################
# Matching attractors and key swapping business
###########################################################################################
# TODO: allow giving in any kind of similarity function.
"""
    match_attractor_ids!(a₊::AbstractDict, a₋; metric = Euclidean(), threshold = Inf)
Given dictionaries `a₊, a₋` mapping IDs to attractors (`Dataset` instances),
match attractor IDs in dictionary `a₊` so that its attractors that are the closest to
those in dictionary `a₋` get assigned the same key as in `a₋`.
Typically the +,- mean after and before some change of parameter of a system.
Distance in attractor space is quantified by the `metric`.

Return the `replacement_map`, a dictionary mapping old keys of `a₊` to
the new ones that they were mapped to.

When finding attractors and their fractions in DynamicalSystems.jl,
different attractors get assigned different IDs. However
which attractor gets which ID is somewhat arbitrary. Finding the attractors of the
same system for slightly different parameters could label "similar" attractors (at
the different parameters) with different IDs.
`match_attractors!` tries to "match" them by modifying the attractor IDs.

Optionally, you can provide a `threshold` value. If the distance between two attractors
is larger than this `threshold`, then it is guaranteed that the attractors will get assigned
different key in the dictionary `a₊`.

This function is utilized in [`basins_fractions_continuation`](@ref).

    match_attractor_ids!(as::AbstractVector{<:AbstractDict}; kwargs...)
When given a vector of dictionaries, iteratively perform the above method for each
consecutive two dictionaries in the vector.
"""
function match_attractor_ids!(a₊::AbstractDict, a₋; metric = Euclidean(), threshold = Inf)
    distances = _similarity_from_distance(a₊, a₋, metric)
    mdc = minimal_distance_combinations(distances)
    rmap = replacement_map(a₊, a₋, mdc, threshold)
    swap_dict_keys!(a₊, rmap)
    return rmap
end

function match_attractor_ids!(as::AbstractVector{<:AbstractDict}; kwargs...)
    for i in 1:length(as)-1
        a₋ = as[i]; a₊ = as[i+1]
        match_attractor_ids!(a₊, a₋; kwargs...)
    end
end


function _similarity_from_distance(a₊, a₋, metric::Metric = Euclidean())
    ids₊, ids₋ = keys(a₊), keys(a₋)
    distances = Dict{eltype(ids₊), Dict{eltype(ids₋), Float64}}()
    for i in ids₊
        d = valtype(distances)()
        for j in ids₋
            # TODO: create and use `dataset_distance` function in delay embeddings.jl
            # TODO: Use KD-trees or `pairwise`
            d[j] = minimum(metric(x, y) for x ∈ a₊[i] for y ∈ a₋[j])
        end
        distances[i] = d
    end
    return distances
end

"""
    minimal_distance_combinations(distances)
Using the distances (dict of dicts), create a vector that only keeps
the minimum distance for each attractor. The output is a vector of
`Tuple{Int, Int, Float64}` meaning `(oldkey, newkey, min_distance)`
where the `newkey` is the key whose attractor has the minimum distance.
"""
function minimal_distance_combinations(distances)
    min_dist_combs = Tuple{Int, Int, Float64}[]
    for i in keys(distances)
        s = distances[i] # dict with distances of i to all in "-"
        j = argmin(s)
        push!(min_dist_combs, (i, j, s[j]))
    end
    sort!(min_dist_combs; by = x -> x[3])
    return min_dist_combs
end


"""
    replacement_map(a₊, a₋, min_dist_combs, threshold)
Given the output of [`minimal_distance_combinations`](@ref), generate the replacement
map mapping old to new keys for `a₊`.
"""
function replacement_map(a₊, a₋, minimal_distance_combinations, threshold)
    rmap = Dict{keytype(a₊), keytype(a₋)}()
    next_id = max(maximum(keys(a₊),), maximum(keys(a₋))) + 1
    # In the same loop we do the logic that matches keys according to distance of values,
    # but also ensures that keys that have too high of a value distance are guaranteeed
    # to have different keys.
    for (oldkey, newkey, mindist) in minimal_distance_combinations
        if mindist > threshold
            # The distance exceeds threshold, so we will assign a new key
            newkey = next_id
            next_id += 1
        end
        rmap[oldkey] = newkey
    end
    return rmap
end


###########################################################################################
# Matching with basins and possibly overlaps
###########################################################################################
"""
```julia
match_basins_ids!(b₊::AbstractArray, b₋; threshold = Inf)
```
Similar to [`match_attractor_ids!`](@ref) but operate on basin arrays instead.

This method matches IDs of attractors whose basins of attraction before and after `b₋,b₊`
have the most overlap (in pixels). This overlap is normalized in 0-1 (with 1 meaning
100% overlap of pixels). The `threshold` in this case is compared to the inverse
of the overlap (so, `threshold = 2` attractors that have less than 50% overlap get
different IDs guaranteed).
"""
function match_basins_ids!(b₊::AbstractArray, b₋; threshold = Inf)
    ids₊, ids₋ = unique(b₊), unique(b₋)
    distances = _similarity_from_overlaps(b₊, ids₊, b₋, ids₋)
    mdc = minimal_distance_combinations(distances)
    rmap = replacement_map(a₊, a₋, mdc, threshold)
    replace!(b₊, rmap...)
    return rmap
end

function _similarity_from_overlaps(b₊, ids₊, b₋, ids₋)
    @assert size(b₋) == size(b₊)
    distances = Dict{eltype(ids₊), Dict{eltype(ids₋), Float64}}()
    for i in ids₊
        Bi = findall(isequal(i), b₊)
        d = valtype(distances)()
        # Compute normalized overlaps of each basin with each other basis
        for j in ids₋
            Bj = findall(isequal(j), b₋)
            overlap = length(Bi ∩ Bj)/length(Bj)
            d[j] = 1 / overlap # distance is inverse overlap
        end
        distances[i] = d
    end
    return distances
end
