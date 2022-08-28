# Utility functions for managing dictionary keys that are useful
# in continuation and attractor matching business
# Thanks a lot to Valentin (@Seelengrab) for generous help in the key swapping code.

"""
    swap_dict_keys!(d::Dict, replacement_map::Dict)

Swap the keys of a dictionary `d` given the `replacement_map`
which maps old keys to new keys.
"""
function swap_dict_keys!(fs::Dict, rmap::Dict)
    isempty(rmap) && return
    cache = Tuple{keytype(fs), valtype(fs)}[]
    for (oldkey, newkey) in rmap
        haskey(fs, oldkey) || continue
        tmp = pop!(fs, oldkey)
        if !haskey(fs, newkey)
            fs[newkey] = tmp
        else
            push!(cache, (newkey, tmp))
        end
    end
    for (k, v) in cache
        fs[k] = v
    end
    return
end

"""
    overwrite_dict!(old::Dict, new::Dict)
In-place overwrite the `old` dictionary for the key-value pairs of the `new`.
"""
function overwrite_dict!(old::Dict, new::Dict)
    empty!(old)
    for (k, v) in new
        old[k] = v
    end
end

"""
    additive_dict_merge!(d1::Dict, d2::Dict)
Merge keys and values of `d2` into `d1` additively: the values of the same keys
are added together in `d1` and new keys are given to `d1` as-is.
"""
function additive_dict_merge!(d1::Dict, d2::Dict)
    for (k, v) in d2
        d1[k] = get(d1, k, 0) + v
    end
    return d1
end


"""
    retract_keys_to_consecutive!(v::Vector{<:Dict}) → rmap
Given a vector of dictionaries with various positive integer keys, retract all keys so that
consecutive integers are used. So if the dictionaries have overall keys 2, 3, 42,
then they will transformed to have 1, 2, 3.

Return the replacement map used to replace keys in all dictionaries with
[`swap_dict_keys!`](@ref).

As this function is used in attractor matching in [`basins_fractions_continuation`](@ref)
it skips the special key `-1`.
"""
function retract_keys_to_consecutive!(v::Vector{<:Dict})
    ukeys = unique_keys(v)
    ukeys = setdiff(ukeys, [-1]) # skip key -1 if it exists
    rmap = Dict(k => i for (i, k) in enumerate(ukeys))
    for d in v
        swap_dict_keys!(d, rmap)
    end
    return rmap
end

"""
    unique_keys(v::Vector{<:Dict})
Given a vector of dictionaries, return a sorted vector of the unique keys
that are present across all dictionaries.
"""
function unique_keys(v::Vector{<:Dict})
    unique_keys = Set(keytype(first(v))[])
    for d in v
        for k in keys(d)
            push!(unique_keys, k)
        end
    end
    return sort!(collect(unique_keys))
end
