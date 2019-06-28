using StatsBase
import DSP
import DSP: Periodograms
export estimate_period

"""
    estimate_period(v, t=0:length(v)-1, method; kwargs...)
Estimate the period of the signal `v`, with accompanying time vector `t`,
using the given `method`:

* `:ac` : Use the autocorrelation function (AC). The value where the AC first
  comes back close to 1 is the period of the signal. The keyword
  `L = length(v)÷10` denotes the length of the AC (thus, given the default
  setting, this method will fail if there less than 10 periods in the signal).
  The keyword `ε = 0.2` means that `1-ε` counts as "1" for the AC.

* `:periodogram` or `:pg`: Use the fast Fourier transform to compute a
   periodogram (power-spectrum) of the given data.  Data must be evenly sampled.

* `:bartlett`: The Bartlett periodogram is aimed at tackling noisy or
  undersampled data, by splitting the signal up into segments, then averaging
  their periodograms.  The keyword `n`  controls the number of segments.
  Using this method reduces the variance of the periodogram compared to the
  standard method.

* `:welch`: Use the Welch method.  The Welch periodogram is aimed at tackling
  noisy or undersampled data, by splitting the signal up into overlapping
  segments and windowing them, then averaging their periodograms. The keyword
  `n`  controls the number of segments, and `noverlap` controls the number of
  overlapping segments.  `window` is the windowing function to be used.

* `:multitaper`: The multitaper method reduces estimation bias by obtaining
  multiple independent estimates from the same sample. Data tapers are then
  windowed and the power spectra are obtained.  Available keywords follow:
  `nw` is the time-bandwidth product, and `ntapers` is the number of tapers.
  If `window` is not specified, the signal is tapered with `ntapers` discrete
  prolate spheroidal sequences with time-bandwidth product `nw`.
  Each sequence is equally weighted; adaptive multitaper is not (yet) supported.
  If `window` is specified, each column is applied as a taper. The sum of
  periodograms is normalized by the total sum of squares of `window`.

"""
function estimate_period(v, t = 0:length(v)-1, method; kwargs...)
    @assert length(v) == length(t)
    methods = Set([
                :ac, :periodogram, :pg, :welch,
                :bartlett, :multitaper, :mt, :lombscargle,
                :esprit])
    if method ∉ methods
        error("Unknown method (`$method`) given to `estimate_period`.")
    elseif method == :ac
        period = _ac_period(v, t; kwargs...)
    end
    return period
end

################################################################################
#                           Autocorrelation Function                           #
################################################################################

"""
    _ac_period(v, t; ε = 0.2, L = length(v)÷10)

Use the autocorrelation function (AC). The value where the AC first
comes back close to 1 is the period of the signal. The keyword
`L = length(v)÷10` denotes the length of the AC (thus, given the default
setting, this method will fail if there less than 10 periods in the signal).
The keyword `ε = 0.2` means that `1-ε` counts as "1" for the AC.
"""
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


################################################################################
#                                 Periodograms                                 #
################################################################################


########################################
#          Basic Periodogram           #
########################################

"""
    _periodogram_period(v, t; kwargs...)

Use the fast Fourier transform to compute a periodogram (power-spectrum) of the
given data.  Data must be evenly sampled.
"""
function _periodogram_period(v, t; kwargs...)

    kwargs = Dict()

    p = Periodograms.periodogram(v; kwargs...)

    return 1 / Periodograms.freq(p)[findmax(Periodograms.power(p))[2]] * (t[end] - t[1]) / length(t)

end