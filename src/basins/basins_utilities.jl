export match_attractors!


"""
    match_attractors!(b₋, a₋, b₊, a₊, [, method = :distance])
Attempt to match the attractors in basins/attractors `b₊, a₊` with those at `b₋, a₋`.
`b` is an array whose values encode the attractor ID, while `a` is a dictionary mapping
IDs to `Dataset`s containing the attractors (e.g. output of [`basins_general`](@ref)).
Typically the +,- mean after and before some change of parameter for a system.

In [`basins_general`](@ref) different attractors get assigned different IDs, however
which attractor gets which ID is somewhat arbitrary, and computing the basins of the
same system for slightly different parameters could label the "same" attractors (at
the different parameters) with different IDs. `match_attractors!` tries to "match" them
by modifying the attractor IDs.

The modification of IDs is always done on the `b, a` that have less attractors.

`method` decides the matching process:
* `method = :overlap` matches attractors whose basins before and after have the most
  overlap (in pixels).
* `method = :distance` matches attractors whose state space distance the smallest.
"""
function match_attractors!(b₋, a₋, b₊, a₊, method = :distance)
    @assert size(b₋) == size(b₊)
    if length(a₊) > length(a₋)
        # Set it up so that modification is always done on `+` attractors
        a₋, a₊ = a₊, a₋
        b₋, b₊ = b₊, b₋
    end
    ids₊, ids₋ = sort!(collect(keys(a₊))), sort!(collect(keys(a₋)))
    if method == :overlap
        match_metric = _match_from_overlaps(b₋, a₋, ids₋, b₊, a₊, ids₊)
    elseif method == :distance
        match_metric = _match_from_distance(b₋, a₋, ids₋, b₊, a₊, ids₊)
    else
        error("Unknown method")
    end

    replacement_mapping = Dict{Int, Int}()
    for (i, ι) in enumerate(ids₊)
        v = match_metric[i, :]
        for j in sortperm(v) # go through the match metric in sorted order
            if ids₋[j] ∈ values(replacement_mapping)
                continue # do not use keys that have been used
            else
                replacement_mapping[ι] = ids₋[j]
            end
        end
    end

    # Do the actual replacing
    replace!(b₊, replacement_mapping...)
    aorig = copy(a₊)
    for (k, v) ∈ replacement_mapping
        a₊[v] = aorig[k]
    end
    # delete unused keys
    for k ∈ keys(a₊)
        if k ∉ values(replacement_mapping); delete!(a₊, k); end
    end
    return
end

function _match_from_overlaps(b₋, a₋, ids₋, b₊, a₊, ids₊)
    # Compute normalized overlaps of each basin with each other basin
    overlaps = zeros(length(ids₊), length(ids₋))
    for (i, ι) in enumerate(ids₊)
        Bi = findall(isequal(ι), b₊)
        for (j, ξ) in enumerate(ids₋)
            Bj = findall(isequal(ξ), b₋)
            overlaps[i, j] = length(Bi ∩ Bj)/length(Bj)
        end
    end
    overlaps
end

using LinearAlgebra
function _match_from_distance(b₋, a₋, ids₋, b₊, a₊, ids₊)
    closeness = zeros(length(ids₊), length(ids₋))
    for (i, ι) in enumerate(ids₊)
        aι = a₊[ι]
        for (j, ξ) in enumerate(ids₋)
            aξ = a₋[ξ]
            closeness[i, j] = 1 / minimum(norm(x .- y) for x ∈ aι for y ∈ aξ)
        end
    end
    closeness
end