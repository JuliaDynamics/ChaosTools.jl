using StatsBase
export estimate_period

"""
    estimate_period(v, t, method; kwargs...)
Estimate the period of the signal `v`, with accompanying time vector `t`,
using the given `method`:

* `"ac"` : Use the autocorrelation function (AC). The value where the AC first
  comes back close to 1 is the period of the signal. The keyword `ε = 0.05`
  here identifies how close to 1 actually counts as 1.
"""
function estimate_period(v, t, method; kwargs...)
    @assert length(v) == length(t)
    if method ∉ ("ac",)
        error("Unknown method given to `estimate_period`.")
    elseif method == "ac"
        period = _ac_period(v, t; kwargs...)

    return period
end

function _ac_period(v, t; ε = 0.05)
    err = "The autocorrelation did not become close to 1."
    ac = autocor(v)
    j = 1
    # find next local maximum of ac
    while j < length(ac) && ac[j] ≥ ac[j+1]
        j += 1
    end
    # now j is a local minimum
    j == length(ac) && error(err)
    while j < length(ac) && ac[j] < ac[j+1]
        j += 1
    end
    # now j is a local maximum
    if j == length(ac) || ac[j] < 1 - ε
        error(err)
    end
    # since now it holds that ac[j] is next local maximum within 1-ε:
    period = t[j] - t[1]
    return period
end
