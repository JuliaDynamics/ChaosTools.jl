using StatsBase
import DSP
import DSP: Periodograms
import LombScargle
export estimate_period

"""
    estimate_period(v, method, t=0:length(v)-1; faith = false, kwargs...)
Estimate the period of the signal `v`, with accompanying time vector `t`,
using the given `method`.  If `faith` is set to `true`, and the chosen
method requires evenly sampled data, then the time data will not be checked;
otherwise, it will.  It is recommended to provide an range in this case, instead
of relying on the `faith` keyword.

# Methods requiring evenly sampled data

These methods are faster, but some are error-prone.

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

# Methods not requiring evenly sampled data

These methods tend to be slow, but versatile and low-error.

* `:ac` : Use the autocorrelation function (AC). The value where the AC first
  comes back close to 1 is the period of the signal. The keyword
  `L = length(v)÷10` denotes the length of the AC (thus, given the default
  setting, this method will fail if there less than 10 periods in the signal).
  The keyword `ε = 0.2` means that `1-ε` counts as "1" for the AC.

* `:lombscargle` or `:ls`: Use the Lomb-Scargle algorithm to compute a
  periodogram.  The advantage of the Lomb-Scargle method is that it does not
  require an equally sampled dataset and performs well on undersampled datasets.
  Constraints have been set on the period, since Lomb-Scargle tends to have
  false peaks at very low frequencies.  That being said, it's a very flexible
  method.  It is extremely customizable, and the keyword arguments that can be
  passed to it are given [in the documentation](https://juliaastro.github.io/LombScargle.jl/stable/index.html#LombScargle.plan).

For more information on the periodogram methods, see the documentation of
`DSP.jl` and `LombScargle.jl`.
"""
function estimate_period(v, method, t = 0:length(v)-1; faith = false, kwargs...)
    @assert length(v) == length(t)

    even_methods = [
            :periodogram, :pg, :welch,
            :bartlett, :multitaper, :mt
    ]

    other_methods = [:ac, :lombscargle, :ls]

    methods = union(even_methods, other_methods)
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

    p = Periodograms.periodogram(v; fs = length(t)/(t[end] - t[1]), kwargs...)

    return 1 / Periodograms.freq(p)[findmax(Periodograms.power(p))[2]]

end

########################################
#          Welch periodogram           #
########################################

"""
    _welch_period(v, t; n = length(v) ÷ 8, noverlap = n ÷ 2,
                  window = nothing, kwargs...)

The Welch periodogram is aimed at tackling noisy or undersampled data,
by splitting the signal up into overlapping segments and windowing
them, then averaging their periodograms.
`n`  controls the number of segments, and `noverlap` controls the number
of overlapping segments.  `window` is the windowing function to be used.
"""
function _welch_period(v, t;
                        n = length(v) ÷ 8,
                        noverlap = n ÷ 2,
                        window = nothing,
                        kwargs...
                        )

    p = Periodograms.welch_pgram(v, n, noverlap;
                                 fs = length(t)/(t[end] - t[1]),
                                 window = window,
                                 kwargs...
                             )

    return 1 / Periodograms.freq(p)[findmax(Periodograms.power(p))[2]]

end

########################################
#         Bartlett periodogram         #
########################################

"""
    _bartlett_period(v, t; n = length(v) ÷ 8, kwargs...)

The Bartlett periodogram is aimed at tackling noisy or undersampled data,
by splitting the signal up into segments, then averaging their periodograms.
`n`  controls the number of segments.  Using this method reduces the variance
of the periodogram compared to the standard method.
"""
_bartlett_period(v, t; n = length(v) ÷ 8, kwargs...) = _welch_period(v, t;
                                                        n = length(v) ÷ 8,
                                                        noverlap = 0, kwargs...)

########################################
#        Multitaper periodogram        #
########################################

"""
    _mt_period(v, t;
                    nw = 4, ntapers = DSP.ceil(2nw)-1,
                    window = DSP.dpss(length(s), nw, ntapers), kwargs...)

The multitaper method reduces estimation bias by obtaining multiple independent estimates from the same sample. Data tapers are then windowed and the power
spectra are obtained.
`nw` is the time-bandwidth product, and `ntapers` is the number of tapers.
If `window` is not specified, the signal is tapered with `ntapers` discrete
prolate spheroidal sequences with time-bandwidth product `nw`.
Each sequence is equally weighted; adaptive multitaper is not (yet) supported.
If `window` is specified, each column is applied as a taper. The sum of
periodograms is normalized by the total sum of squares of `window`.
"""
function _mt_period(v, t;
                        nw = 4,
                        ntapers::Integer = ceil(Int, 2nw)-1,
                        window = DSP.dpss(length(v), nw, ntapers),
                        kwargs...
                    )

    p = Periodograms.mt_pgram(v;
                                 fs = length(t)/(t[end] - t[1]),
                                 nw = nw,
                                 ntapers = ntapers,
                                 window = window,
                                 kwargs...
                             )

    return 1 / Periodograms.freq(p)[findmax(Periodograms.power(p))[2]]

end

########################################
#       Lomb-Scargle periodogram       #
########################################

"""
    _ls_period(v, t;
                minimum_period::Real = 2 * (t[end] - t[1]) / length(t),
                maximum_period::Real = (t[end] - t[1]) / 1.5,
                kwargs... # see the documentation of LombScargle.jl for these
        )

Uses the Lomb-Scargle algorithm to compute a periodogram.  The advantage of the Lomb-Scargle method is that it does not require an equally sampled dataset, and it performs well on undersampled datasets.
Constraints have been set on the period, since Lomb-Scargle tends to have false peaks at very low frequencies.  That being said, it's a very flexible method.  It is extremely customizable, and the keyword arguments that can be passed to it are given [in the documentation](https://juliaastro.github.io/LombScargle.jl/stable/index.html#LombScargle.plan).
"""
function _ls_period(v, t;
                    minimum_period::Real = 4 * (t[end] - t[1]) / length(t),
                    maximum_period::Real = (t[end] - t[1]) / 1.5,
                    kwargs...
            )

    plan = LombScargle.plan(t, v;
                            minimum_frequency = 1/maximum_period,
                            maximum_frequency = 1/minimum_period,
                            kwargs...
                        )

    p = LombScargle.lombscargle(plan)

    return LombScargle.findmaxperiod(p)[1]

end
