export basins_fractions, match_attractors!

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


# TODO: Somehow needs to make sure that attractors that are too far get different key.

"""
    match_attractors!(b₋, a₋, b₊, a₊, [, method = :distance])
Attempt to match the attractors in basins/attractors `b₊, a₊` with those at `b₋, a₋`.
`b` is an array whose values encode the attractor ID, while `a` is a dictionary mapping
IDs to `Dataset`s containing the attractors (i.e, output of [`basins_of_attraction`](@ref)).
Typically the +,- mean after and before some change of parameter for a system.

In [`basins_of_attraction`](@ref) different attractors get assigned different IDs, however
which attractor gets which ID is somewhat arbitrary, and computing the basins of the
same system for slightly different parameters could label the "same" attractors (at
the different parameters) with different IDs. `match_attractors!` tries to "match" them
by modifying the attractor IDs.

The modification of IDs is always done on the `b, a` that have *less* attractors.

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
        closeness_metric = _match_from_overlaps(b₋, ids₋, a₊, ids₊)
    elseif method == :distance
        closeness_metric = _match_from_distance(a₋, ids₋, a₊, ids₊)
    else
        error("Unknown `method` for `match_attractors!`.")
    end

    replacement_map = _replacement_map(closeness_metric, ids₋, ids₊)

    # Do the actual replacing; easy for the basin arrays
    replace!(b₊, replacement_map...)
    # But a bit more involved for the attractor dictionaries
    aorig = copy(a₊)
    for (k, v) ∈ replacement_map
        a₊[v] = aorig[k]
    end
    # delete unused keys!
    for k ∈ keys(a₊)
        if k ∉ values(replacement_map); delete!(a₊, k); end
    end
    return
end

function _match_from_overlaps(b₋, ids₋, b₊, ids₊)
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

using LinearAlgebra: norm
function _match_from_distance(a₋, ids₋, a₊, ids₊)
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


"""
    _replacement_map(closeness_metric, ids₋, ids₊)
Return a dictionary mapping old IDs to new IDs for `ids₊` given the `closeness_metric`.
Closest keys in `ids₋` become keys for `ids₊`.
"""
function _replacement_map(closeness_metric, ids₋, ids₊)
    replacement_map = Dict{Int, Int}()
    for (i, ι) in enumerate(ids₊)
        v = closeness_metric[i, :]
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
