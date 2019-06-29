using StatsBase
import DSP
import DSP: Periodograms
import LombScargle
export estimate_period

"""
    estimate_period(v, method, t=0:length(v)-1;, method_specific_kwargs...)
Estimate the period of the signal `v`, with accompanying time vector `t`,
using the given `method`.

If `t` is an AbstractArray, then it is iterated through to ensure that it's
evenly sampled (if necessary for the algorithm).  To avoid this, you can pass
any `AbstractRange`, like a `UnitRange` or a `LinRange`, which are defined to be
evenly sampled.

## Methods requiring evenly sampled data

These methods are faster, but some are error-prone.

* `:periodogram` or `:pg`: Use the fast Fourier transform to compute a
   periodogram (power-spectrum) of the given data.  Data must be evenly sampled.

* `:multitaper` or `mt`: The multitaper method reduces estimation bias by using
  multiple independent estimates from the same sample. Data tapers are then
  windowed and the power spectra are obtained.  Available keywords follow:
  `nw` is the time-bandwidth product, and `ntapers` is the number of tapers.
  If `window` is not specified, the signal is tapered with `ntapers` discrete
  prolate spheroidal sequences with time-bandwidth product `nw`.
  Each sequence is equally weighted; adaptive multitaper is not (yet) supported.
  If `window` is specified, each column is applied as a taper. The sum of
  periodograms is normalized by the total sum of squares of `window`.

* `:autocorrelation` or `:ac`: Use the autocorrelation function (AC). The value
  where the AC first comes back close to 1 is the period of the signal. The
  keyword `L = length(v)÷10` denotes the length of the AC (thus, given the
  default setting, this method will fail if there less than 10 periods in the
  signal). The keyword `ϵ = 0.2` (`\\epsilon`) means that `1-ϵ` counts as "1" for the AC.

## Methods not requiring evenly sampled data

These methods tend to be slow, but versatile and low-error.

* `:lombscargle` or `:ls`: Use the Lomb-Scargle algorithm to compute a
  periodogram.  The advantage of the Lomb-Scargle method is that it does not
  require an equally sampled dataset and performs well on undersampled datasets.
  Constraints have been set on the period, since Lomb-Scargle tends to have
  false peaks at very low frequencies.  That being said, it's a very flexible
  method.  It is extremely customizable, and the keyword arguments that can be
  passed to it are given [in the documentation](https://juliaastro.github.io/LombScargle.jl/stable/index.html#LombScargle.plan).

*  `:zerocrossing` or `:zc`: Find the zero crossings of the data, and use the
  average difference between zero crossings as the period.  This is a naïve
  implementation, with only linear interpolation; however, it's useful as a
  sanity check.  The keyword `line` controls where the "crossing point" is.
  It deffaults to `mean(v)`.

For more information on the periodogram methods, see the documentation of
`DSP.jl` and `LombScargle.jl`.
"""
function estimate_period(v, method, t = 0:length(v)-1; kwargs...)
    @assert length(v) == length(t)

    even_methods  = [:periodogram, :pg, :multitaper, :mt, :autocorrelation, :ac]
    other_methods = [:lombscargle, :ls, :zerocrossing, :zc]
    if method ∉ even_methods && method ∉ other_methods
        error("Unknown method (`$method`) given to `estimate_period`.")
    end

    period = if method ∈ even_methods
                isevenlysampled(t) ||
                error("Your time data was not evenly sampled,
                and the algorithm `$method` requires evenly sampled data.")

                if method == :periodogram || method == :pg
                    _periodogram_period(v, t; kwargs...)
                elseif method == :multitaper || method == :mt
                    _mt_period(v, t; kwargs...)
                elseif method == :autocorrelation || method == :ac
                    _ac_period(v, t; kwargs...)
                end
            else
                if method == :lombscargle || method == :ls
                    _ls_period(v, t; kwargs...)
                elseif method == :zerocrossing || method == :zc
                    _zc_period(v, t; kwargs...)
                end
            end

    return period
end

function isevenlysampled(t::AbstractVector)
    for i in 2:length(t)-1
        if !(t[nextind(t, i)] - t[i] ≈ t[i] - t[prevind(t, i)])
            return false
            break
        end
    end
    return true
end

# AbstractRanges are defined to be evenly sampled.
isevenlysampled(::AbstractRange) = true

################################################################################
#                           Autocorrelation Function                           #
################################################################################
"""
    _ac_period(v, t; ϵ = 0.2, L = length(v)÷10)

Use the autocorrelation function (AC). The value where the AC first
comes back close to 1 is the period of the signal. The keyword
`L = length(v)÷10` denotes the length of the AC (thus, given the default
setting, this method will fail if there less than 10 periods in the signal).
The keyword `ϵ = 0.2` means that `1-ϵ` counts as "1" for the AC.
"""
function _ac_period(v, t; ϵ = 0.2, L = length(v)÷10)
    err = "The autocorrelation did not become close to 1."
    ac = autocor(v, 0:L)
    j = 0
    local_maxima = findall(i -> ac[i-1] < ac[i] ≥ ac[i+1], 2:length(ac)-1)
    isempty(local_maxima) && error("AC did not have any local maxima")
    # progressively scan the local maxima and find the first within 1-ϵ
    for i in local_maxima
        if ac[i] > 1-ϵ
            j = i
            break
        end
    end
    j == 0 && error("No local maximum of the AC exceeded 1-ϵ")
    # since now it holds that ac[j] is a local maximum within 1-ϵ:
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

    p = Periodograms.periodogram(v; fs = length(t)/(t[end] - t[1]), kwargs...)

    return 1 / Periodograms.freq(p)[findmax(Periodograms.power(p))[2]]

end

########################################
#        Multitaper periodogram        #
########################################

"""
    _mt_period(v, t;
                    nw = 4, ntapers = DSP.ceil(2nw)-1,
                    window = DSP.dpss(length(s), nw, ntapers), kwargs...)

The multitaper method reduces estimation bias by obtaining multiple independent
estimates from the same sample. Data tapers are then windowed and the power
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

    Sys.WORD_SIZE == 32 &&
        error("multitaper method doesn't work on 32-bit systems. Sorry!")

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

################################################################################
#                                Zero crossings                                #
################################################################################
"""
    _zc_period(v, t; line = mean(v))

Find the zero crossings of the data, and use the average difference between
zero crossings as the period.  This is a naïve implementation, with no
interpolation; however, it's useful as a sanity check.

The keyword `line` controls where the "zero point" is.

The implementation of the function was inspired by [this gist](https://gist.github.com/endolith/255291),
and has been modified for performance and to support arbitrary time grids.
"""
function _zc_period(v, t; line = mean(v))
    # This line might be a little opaque, so I'll provide some more detail.
    # The macro @. applies the broadcast operator to all operations in the
    # expression given to it, except for those which have an $ in front of
    # them.  It also does some optimization to fuse the broadcasts into a
    # single operation.
    # So, @. a * b + c is more like fma.(a, b, c) than a .* b .+ c.
    # The `@views` macro provides a view into an array without actually copying it,
    # so this allocates less.
    # What this is doing is finding all rising line crossings, which means all
    # pairs of consecutive points for which the first is below the line and the
    # second is above it.
    inds = findall(@. ≥(line, $@view(v[2:end])) & <(line, $@view(v[1:end-1])))
    mean(diff(t[inds]))
end
