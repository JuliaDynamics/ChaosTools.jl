using StatsBase
export estimate_period

"""
    estimate_period(v, method, t=0:length(v)-1; kwargs...)
Estimate the period of the signal `v`, with accompanying time vector `t`,
using the given `method`:

* `"ac"` : Use the autocorrelation function (AC). The value where the AC first
  comes back close to 1 is the period of the signal. The keyword
  `L = length(v)÷10` denotes the length of the AC (thus, given the default
  setting, this method will fail if there less than 10 periods in the signal).
  The keyword `ε = 0.2` means that `1-ε` counts as "1" for the AC.
"""
function estimate_period(v, method, t = 0:length(v)-1; kwargs...)
    @assert length(v) == length(t)
    if method ∉ ("ac",)
        error("Unknown method given to `estimate_period`.")
    elseif method == "ac"
        period = _ac_period(v, t; kwargs...)
    end
    return period
end

function _ac_period(v, t; ε = 0.2, L = length(v)÷10)
    err = "The autocorrelation did not become close to 1."
    ac = autocor(v, 0:L)
    j = 0
    local_maxima = findall(i -> ac[i-1] < ac[i] ≥ ac[i+1], 2:length(ac)-1)
    isempty(local_maxima) && error("AC did not have any local maxima")
    # progressively scan the local maxima and find the first within 1-ε
    for i in local_maxima
        if ac[i] > 1-ε
            j = i
            break
        end
    end
    j == 0 && error("No local maximum of the AC exceeded 1-ε")
    # since now it holds that ac[j] is a local maximum within 1-ε:
    period = t[j+1] - t[1]
    return period
end
