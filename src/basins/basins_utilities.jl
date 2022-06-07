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


"""
    match_attractor_ids!(b₋, a₋, b₊, a₊, [, method = :distance]; metric = Euclidean())
Match the attractors in basins/attractors `b₊, a₊` with those at `b₋, a₋`.
`b` is an array whose values encode the attractor ID, while `a` is a dictionary mapping
IDs to `Dataset`s containing the attractors (i.e, output of [`basins_of_attraction`](@ref)).
Typically the +,- mean after and before some change of parameter for a system.

In [`basins_of_attraction`](@ref) different attractors get assigned different IDs, however
which attractor gets which ID is somewhat arbitrary, and computing the basins of the
same system for slightly different parameters could label the "same" attractors (at
the different parameters) with different IDs. `match_attractors!` tries to "match" them
by modifying the attractor IDs.

The modification of IDs is always done on the `b, a` that have *less* attractors.
If they have equal, modification is done on the `₊` values.

`method` decides the matching process:
* `method = :overlap` matches attractors whose basins of attraction before and after
  have the most overlap (in pixels).
* `method = :distance` matches attractors whose state space distance the smallest.
  The keyword `metric` decides the metric for the distance (anything from Distances.jl).
"""
function match_attractor_ids!(b₋, a₋, b₊, a₊, method = :distance; metric = Euclidean())
    if length(a₊) > length(a₋)
        # Set it up so that modification is always done on `+` attractors
        a₋, a₊ = a₊, a₋
        b₋, b₊ = b₊, b₋
    end
    ids₊, ids₋ = sort!(collect(keys(a₊))), sort!(collect(keys(a₋)))
    if method == :overlap
        similarity = _similarity_from_overlaps(b₋, ids₋, a₊, ids₊)
    elseif method == :distance
        similarity = _similarity_from_distance(a₋, ids₋, a₊, ids₊, metric)
    else
        error("Unknown `method` for `match_attractors!`.")
    end

    replacement_map = _replacement_map(similarity, ids₋, ids₊)

    # Do the actual replacing; easy for the basin arrays
    if b₊ isa AbstractArray
        replace!(b₊, replacement_map...)
    end
    # But a bit more involved for the attractor dictionaries
    aorig = copy(a₊)
    for (k, v) ∈ replacement_map
        a₊[v] = aorig[k]
    end
    # delete unused keys!
    for k ∈ keys(a₊)
        if k ∉ values(replacement_map); delete!(a₊, k); end
    end
    return similarity
end

function _similarity_from_overlaps(b₋, ids₋, b₊, ids₊)
    @assert size(b₋) == size(b₊)
    # Compute normalized overlaps of each basin with each other basin
    overlaps = zeros(length(ids₊), length(ids₋))
    for (i, ι) in enumerate(ids₊)
        Bi = findall(isequal(ι), b₊)
        for (j, ξ) in enumerate(ids₋)
            Bj = findall(isequal(ξ), b₋)
            overlaps[i, j] = length(Bi ∩ Bj)/length(Bj)
        end
    end
    return overlaps
end

function _similarity_from_distance(a₋, ids₋, a₊, ids₊, metric::Metric)
    distances = zeros(length(ids₊), length(ids₋))
    for (i, ι) in enumerate(ids₊)
        aι = a₊[ι]
        for (j, ξ) in enumerate(ids₋)
            aξ = a₋[ξ]
            distances[i, j] = minimum(metric(x, y) for x ∈ aι for y ∈ aξ)
        end
    end
    return 1 ./ distances
end


"""
    _replacement_map(similarity, ids₋, ids₊)
Return a dictionary mapping old IDs to new IDs for `ids₊` given the `similarity`.
Closest keys (according to similarity) in `ids₋` become keys for `ids₊`.
"""
function _replacement_map(similarity, ids₋, ids₊)
    replacement_map = Dict{Int, Int}()
    for (i, ι) in enumerate(ids₊)
        v = similarity[i, :]
        for j in sortperm(v) # go through the closeness metric in sorted order
            if ids₋[j] ∈ values(replacement_map)
                continue # do not use keys that have been used
            else
                replacement_map[ι] = ids₋[j]
            end
        end
    end
    return replacement_map
end


"""
    unique_attractor_ids!(a₋, a₊, threshold::Real; metric = Euclidean())
This is a stricter version of [`match_attractor_ids!`](@ref).
First, attractors are matched by distance by calling [`match_attractor_ids`](@ref).
Then, there is an extra step that ensures that attractors whose
distance is greater than `threshold` are explicitly assigned different IDs.
The new IDs used are always higher integers than the existing IDs in either `a₋, a₊`.

For example, assume that both `a₋, a₊` have three attractors, and (after matching)
attractors with IDs 2, 3 are closer than `threshold` to each other, but attractors
with ID 1 are not within `threshold` distance. Keys 2, 3 remain as is in both `a₋, a₊`
but key 1 will become 4 in `a₊`.

This is used in [`continuation_basins_fractions`](@ref).
"""
function unique_attractor_ids!(a₋, a₊, threshold::Real; metric = Euclidean())
    # We utilize duck-typing of `match_attractor_ids!` for the `b` values
    similarity = match_attractor_ids!(nothing, a₋, nothing, a₊, :distance; metric)
    if length(a₊) > length(a₋) # because of the swapping in `match_attractor_ids!`
        similarity = permutedims(similarity) # like transpose, but not lazy
    end
    distances = 1 ./ similarity
    ids₊, ids₋ = sort!(collect(keys(a₊))), sort!(collect(keys(a₋)))
    next_id = max(maximum(ids₊), maximum(ids₋)) + 1
    # Go through attractors; for same ID, check distance
    for (i, ι) in enumerate(ids₊)
        if ι ∈ ids₋
            j = findfirst(isequal(ι), ids₋)
            d = distances[i, j]
            if d > threshold # replace key with `next_id`
                a₊[next_id] = a₊[ι]
                delete!(a₊, ι)
                next_id += 1
            end
        else
            continue
        end
    end
end