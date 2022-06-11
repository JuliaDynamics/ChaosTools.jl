export basins_fractions, match_attractor_ids!, unique_attractor_ids!

"""
    basins_fractions(basins::Array) → fs::Dict
Calculate the state space fraction of the basins of attraction encoded in `basins`.
The elements of `basins` are integers, enumerating the attractor that the entry of
`basins` converges to (i.e., like the output of [`basins_of_attraction`](@ref)).
Return a dictionary that maps attractor IDs to their relative fractions.

In[^Menck2013] the authors use these fractions to quantify the stability of a basin of
attraction, and specifically how it changes when a parameter is changed.

[^Menck2013]:
    Menck, Heitzig, Marwan & Kurths. How basin stability complements the linear
    stability paradigm. [Nature Physics, 9(2), 89–92](https://doi.org/10.1038/nphys2516)
"""
function basins_fractions(basins::AbstractArray)
    fs = Dict{eltype(basins), Float64}()
    ids = unique(basins)
    N = length(basins)
    for ξ in ids
        B = count(isequal(ξ), basins)
        fs[ξ] = B/N
    end
    return fs
end


###########################################################################################
# Matching attractors and key swapping business
###########################################################################################
# This code is one of the most complicated pieces of code I've ever had to write.
# It is like 50 lines of code, yet it took me 6 full hours. Shit.
# Thanks a lot to Valentin @Seelengrab for generous help in the key swapping code.
"""
    match_attractor_ids!(a₊::AbstractDict, a₋; metric = Euclidean(), threshold = Inf)
Given dictionaries `a₊, a₋` mapping IDs to attractors,
match attractor IDs in dictionary `a₊` so that its attractors that are the closest to
those in dictionary `a₋` get assigned the same key as in `a₋`.
Typically the +,- mean after and before some change of parameter of a system.
Distance is quantified by the `metric`.

Return the `replacement_map`, a dictionary mapping previous indices of `a₊` to
the new ones that they were mapped to.

When finding attractors and their fractions in DynamicalSystems.jl,
different attractors get assigned different IDs. However
which attractor gets which ID is somewhat arbitrary. Finding the attractors of the
same system for slightly different parameters could label the "same" attractors (at
the different parameters) with different IDs. `match_attractors!` tries to "match" them
by modifying the attractor IDs.

Optionally, you can provide a `threshold` value. If the distance between two attractors
is larger than this `threshold`, then it is guaranteed that the attractors will get assigned
different key in the dictionary `a₊`.

This function is always called at the end of [`basins_fractions_continuation`](@ref).

    match_attractor_ids!(as::AbstractVector{<:AbstractDict}; kwargs...)
When given a vector of dictionaries, iteratively perform the above method for each
consecutive two dictionaries in the vector.
"""
function match_attractor_ids!(a₊::AbstractDict, a₋; metric = Euclidean(), threshold = Inf)
    distances = _similarity_from_distance(a₊, a₋, metric)
    mdc = _minimal_distance_combinations(distances)
    replacement_map = _swap_dict_keys!(a₊, a₋, mdc, threshold)
    return replacement_map
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
        d = Dict{eltype(ids₋), Float64}()
        for j in ids₋
            # TODO: create and use `dataset_distance` function in delay embeddings.jl
            # TODO: Use KD-trees or `pairwise`
            d[j] = minimum(metric(x, y) for x ∈ a₊[i] for y ∈ a₋[j])
        end
        distances[i] = d
    end
    return distances
end

function _minimal_distance_combinations(distances)
    # Prioritize mappings that have the least distance
    minimal_distance_combinations = Tuple{Int, Int, Float64}[]
    for i in keys(distances)
        s = distances[i] # dict with distances of i to all in "-"
        j = argmin(s)
        push!(minimal_distance_combinations, (i, j, s[j]))
    end
    sort!(minimal_distance_combinations; by = x -> x[3])
end

function _swap_dict_keys!(a₊, a₋, minimal_distance_combinations, threshold = Inf)
    replacement_map = Dict{keytype(a₊), keytype(a₋)}()
    next_id = max(maximum(keys(a₊),), maximum(keys(a₋))) + 1
    # In the same loop we do the logic that matches keys according to distance of values,
    # but also ensures that keys that have too high of a value distance are guaranteeed
    # to have different keys.
    cache = Tuple{keytype(a₊), valtype(a₊)}[]
    for (oldkey, newkey, mindist) in minimal_distance_combinations
        if mindist > threshold
            # The distance exceeds threshold, so we will assign a new key
            newkey = next_id
            next_id += 1
        end
        tmp = pop!(a₊, oldkey)
        if !haskey(a₊, newkey)
            a₊[newkey] = tmp
        else
            push!(cache, (newkey, tmp))
        end
        replacement_map[oldkey] = newkey
    end
    for (k, v) in cache
        a₊[k] = v
    end
    return replacement_map
end




###########################################################################################
# Matching with basins and possibly overlaps
###########################################################################################
"""
```julia
match_attractor_ids!(b₊, a₊, b₋, a₋, [, method = :distance];
                     metric = Euclidean(), threshold = Inf)
```
An enhancement of `match_attractor_ids!(a₊, a₋, ...)` that also takes in calculated
basins of attraction `b::AbstractArray`. Once again match attractor IDs, now in both
the dictionary `a₊` and the basins of attraction `b₊`. In addition, the
`method` decides the matching process:
* `method = :overlap` matches attractors whose basins of attraction before and after
  have the most overlap (in pixels). This overlap is normalized in 0-1 (with 1 meaning
  100% overlap of pixels). The `threshold` in this case is compared to the inverse
  of the overlap.
* `method = :distance` is the same as in `match_attractor_ids!(a₊, a₋, ...)`.
"""
function match_attractor_ids!(b₊::AbstractArray, a₊::AbstractDict, b₋, a₋,
    method = :distance; metric = Euclidean(), threshold = Inf)

    if method == :overlap
        ids₊, ids₋ = keys(a₊), keys(a₋)
        distances = _similarity_from_overlaps(b₊, ids₊, b₋, ids₋)
    elseif method == :distance
        distances = _similarity_from_distance(a₊, a₋, metric)
    else
        error("Unknown `method` for `match_attractors!`.")
    end
    mdc = _minimal_distance_combinations(distances)
    replacement_map = _swap_dict_keys!(a₊, a₋, mdc, threshold)
    replace!(b₊, replacement_map...)
    return replacement_map
end

function _similarity_from_overlaps(b₊, ids₊, b₋, ids₋)
    @assert size(b₋) == size(b₊)
    distances = Dict{eltype(ids₊), Dict{eltype(ids₋), Float64}}[]
    for i in ids₊
        Bi = findall(isequal(i), b₊)
        d = valtype(distances)[]
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
