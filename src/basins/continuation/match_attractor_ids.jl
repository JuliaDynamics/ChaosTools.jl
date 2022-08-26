# Notice this file uses heavily `dict_utils.jl`!
export match_attractor_ids!, match_basins_ids!, replacement_map
###########################################################################################
# Matching attractors and key swapping business
###########################################################################################
"""
    match_attractor_ids!(a₊::AbstractDict, a₋; metric = Euclidean(), threshold = Inf)
Given dictionaries `a₊, a₋` mapping IDs to attractors (`Dataset` instances),
match attractor IDs in dictionary `a₊` so that its attractors that are the closest to
those in dictionary `a₋` get assigned the same key as in `a₋`.
Typically the +,- mean after and before some change of parameter of a system.

Return the `replacement_map`, a dictionary mapping old keys of `a₊` to
the new ones that they were mapped to. You can obtain this map, without modifying
the dictionaries, by directly calling the [`replacement_map`](@ref) function
with the output of [`datasets_sets_distances`](@ref) for given `metric`.

## Description

When finding attractors and their fractions in DynamicalSystems.jl,
different attractors get assigned different IDs. However
which attractor gets which ID is somewhat arbitrary. Finding the attractors of the
same system for slightly different parameters could label "similar" attractors (at
the different parameters) with different IDs.
`match_attractors!` tries to "match" them by modifying the attractor IDs.

Distance in attractor space uses the [`datasets_sets_distances`](@ref) function,
and hence the keyword `metric` can be whatever that function accepts, such as
an actual `Metric` instance, or an arbitrary user-defined function that computes
an arbitrary "distance" between two datasets.

Additionally, you can provide a `threshold` value. If the distance between two attractors
is larger than this `threshold`, then it is guaranteed that the attractors will get assigned
different key in the dictionary `a₊`.
"""
function match_attractor_ids!(a₊::AbstractDict, a₋; metric = Euclidean(), threshold = Inf)
    distances = datasets_sets_distances(a₊, a₋, metric)
    # mdc = minimal_distance_combinations(distances)
    # rmap = replacement_map(a₊, a₋, mdc, threshold)
    rmap = replacement_map(a₊, a₋, distances, threshold)
    swap_dict_keys!(a₊, rmap)
    return rmap
end

"""
    match_attractor_ids!(as::AbstractVector{<:AbstractDict}; kwargs...)
When given a vector of dictionaries, iteratively perform the above method for each
consecutive two dictionaries in the vector.
This method is utilized in [`basins_fractions_continuation`](@ref).
"""
function match_attractor_ids!(as::AbstractVector{<:AbstractDict}; kwargs...)
    for i in 1:length(as)-1
        a₋ = as[i]; a₊ = as[i+1]
        match_attractor_ids!(a₊, a₋; kwargs...)
    end
end

"""
    minimal_distance_combinations(distances)
Using the distances (dict of dicts), create a vector that only keeps
the minimum distance for each attractor. The output is a vector of
`Tuple{Int, Int, Float64}` meaning `(oldkey, newkey, min_distance)`
where the `newkey` is the key whose attractor has the minimum distance.
The vector is also sorted according to the distance.
"""
function minimal_distance_combinations(distances)
    # Here we assume that the dictionary keys are integers
    min_dist_combs = Tuple{Int, Int, Float64}[]
    for i in keys(distances)
        s = distances[i] # dict with distances of i to all in "-"
        # j is already the correct key, because `s` is a dictionary
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
    next_id = max(maximum(keys(a₊)), maximum(keys(a₋))) + 1
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


"""
    replacement_map(a₊, a₋, distances, threshold) → map
Return a dictionary mapping keys in `a₊` to new keys in `a₋`.
"""
function replacement_map(a₊::Dict, a₋::Dict, distances::Dict, threshold)
    # Transform distances to sortable collection. Sorting by distance
    # ensures we prioritize the most closest matches
    sorted_keys_with_distances = Tuple{Int, Int, Float64}[]
    for i in keys(distances)
        for j in keys(distances[i])
            push!(sorted_keys_with_distances, (i, j, distances[i][j]))
        end
    end
    sort!(sorted_keys_with_distances; by = x -> x[3])

    # Iterate through minimal distances, find match, and remove
    # all remaining same indices a'la Eratosthenis sieve
    # In the same loop we match keys according to distance of values,
    # but also ensures that keys that have too high of a value distance are guaranteeed
    # to have different keys, and ensure that there is unique mapping happening!
    rmap = Dict{keytype(a₊), keytype(a₋)}()
    next_id = max(maximum(keys(a₊)), maximum(keys(a₋))) + 1
    done_keys₊ = keytype(a₊)[] # stores keys of a₊ already processed
    used_keys₋ = keytype(a₋)[] # stores keys of a₋ already used
    for (oldkey, newkey, mindist) in sorted_keys_with_distances
        (oldkey ∈ done_keys₊ || newkey ∈ used_keys₋) && continue
        if  mindist < threshold
            # mapping key is alright, but now need to delete all entries with same key
            push!(used_keys₋, newkey)
        else
            # The distance exceeds threshold, so we will assign a new key
            # (notice that this assumes the sorting by distance we did above,
            # otherwise it wouldn't work!)
            newkey = next_id
            next_id += 1
        end
        # and we also delete all entries with old key
        # entries = findall(x -> x[1] == oldkey, sorted_keys_with_distances)
        # deleteat!(sorted_keys_with_distances, entries)
        push!(done_keys₊, oldkey)
        # and assign the change in the replacement map
        rmap[oldkey] = newkey
    end

    # if not all keys were processed, we map them to the next available integers
    if length(done_keys₊) ≠ length(keys(a₊))
        unprocessed = setdiff(collect(keys(a₊)), done_keys₊)
        for oldkey in unprocessed
            rmap[oldkey] = next_id
            next_id += 1
        end
    end
    return rmap
end



###########################################################################################
# Matching with basins and possibly overlaps
###########################################################################################
"""
    match_basins_ids!(b₊::AbstractArray, b₋; threshold = Inf)
Similar to [`match_attractor_ids!`](@ref) but operate on basin arrays instead
(the arrays typically returned by [`basins_of_attraction`](@ref)).

This method matches IDs of attractors whose basins of attraction before and after `b₋,b₊`
have the most overlap (in pixels). This overlap is normalized in 0-1 (with 1 meaning
100% overlap of pixels). The `threshold` in this case is compared to the inverse
of the overlap (so, for `threshold = 2` attractors that have less than 50% overlap get
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
