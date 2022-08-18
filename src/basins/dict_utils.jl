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