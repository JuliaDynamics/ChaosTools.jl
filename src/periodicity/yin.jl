using LinearAlgebra, StaticArrays

################################################################################
#                                YIN algorithm                                 #
################################################################################

# Fundamental frequency estimation. Based on the YIN alorgorithm
# [1]: Patrice Guyot. (2018, April 19). Fast Python implementation of the Yin algorithm
# (Version v1.1.1). Zenodo. http://doi.org/10.5281/zenodo.1220947
# [2]: De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for
# speech and music. The Journal of the Acoustical Society of America, 111(4), 1917-1930.


"""
    differenceFunction_original(x, τmax)

Computes the difference function of `x`.
This corresponds to equation (6) in [^2]:

```math
d_t(\tau) = \sum_{j=1}^W (x_j - x_{j+\tau})^2
```

`x`: audio data
`τ_max`: integration window size

[^1]: De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for
speech and music. The Journal of the Acoustical Society of America, 111(4), 1917-1930.
"""
function differenceFunction_original(x, τmax)

    df = zeros(eltype(x), τmax)
    for τ in 2:τmax
        for j in 1:(N-τmax)
            df[tau] = (x[j] - x[j + τ]) ^ 2
        end
    end
    return df
end

"""
    getPitch(
        cmndf, τ_min, τ_max, harmonic_threshold=0.1
    )

Return the fundamental period of a frame,
based on the cumulative mean normalized difference function (CMNDF).

`cmndf`: cumulative mean normalized difference of the data

`τ_min`: minimum period

`τ_max`: maximum period

`harmonic_threshold`: harmonicity threshold to determine if it is necessary to compute pitch frequency

We define the CMNDF as follows:
```math
d_t^\\prime(\\tau) = \\begin{cases}
        1 & \\text{if} ~ \\tau=0 \\\\
        d_t(\\tau)/\\left[{(\\frac 1 \\tau) \\sum_{j=1}^{\\tau} d_{t}(j)}\\right] & \\text{otherwise}
        \\end{cases}
```

Returns the fundamental period if there are values under the threshold, and 0 otherwise.
"""
function getPitch(cmndf, τ_min, τ_max, harmonic_threshold=0.1)
    τ = τ_min

    while τ < τ_max
        if cmndf[tau] < harmonic_threshold # FIXME missing ends
            while τ + 1 < τ_max && cmndf[τ + 1] < cmndf[τ]
                τ += 1
            return τ
        τ += 1

    return 0    # if unvoiced

end


 """
Compute cumulative mean normalized difference function (CMND).
This corresponds to equation (8) in [1]
:param df: Difference function
:param N: length of data
:return: cumulative mean normalized difference function
:rtype: list
"""
function cmnd(df, N)
    cmndf = df[2:end] * range(2, length=N) / cumsum(df[2:end])
    return [1; cmndf]
end

"""
    yin(
        sig, sr;
        w_len = 512,
        w_step = 256,
        f0_min = 100,
        f0_max = 500,
        harmonic_threshold = 0.1,
        diffference_function = differenceFunction_original
    )

Computes the Yin Algorithm.
Returns fundamental frequency and harmonic rate.

## Arguments

* `sig`: Audio signal
* `sr`: sampling rate

## Keyword arguments

* `w_len`: size of the analysis window (samples)
* `w_step`: size of the lag between two consecutive windows (samples)
* `f0_min`: Minimum fundamental frequency that can be detected (hertz)
* `f0_max`: Maximum fundamental frequency that can be detected (hertz)
* `harmonic_threshold`: Threshold of detection. The algorithm returns the first
minimum of the CMND function below this treshold.
* `diffference_function`: The difference function to be used (by default [`differenceFunction_original`](@ref)).

## Returns:

* `pitches`: vector of fundamental frequencies,
* `harmonic_rates`: vector of harmonic rate values for each fundamental frequency value (confidence value)
* `argmins`: minimums of the Cumulative Mean Normalized DifferenceFunction
* `times`: list of time of each estimation

## Citations
[1]: Patrice Guyot. (2018, April 19). Fast Python implementation of the Yin algorithm
(Version v1.1.1). Zenodo. http://doi.org/10.5281/zenodo.1220947
[2]: De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for
speech and music. The Journal of the Acoustical Society of America, 111(4), 1917-1930.
"""
function yin(
            sig, sr;
            w_len = 512,
            w_step = 256,
            f0_min = 100,
            f0_max = 500,
            harmonic_threshold = 0.1,
            diffferenceFunction = differenceFunction_original
        )

    τ_min = round(Int, sr / f0_max)
    τ_max = round(Int, sr / f0_min)

    time_scale = range(0, length=(length(sig) - w_len), w_step)  # time values for each analysis window
    times = copy(time_scale) ./ eltype(sig)(sr)
    frames = [sig[(t+1):(t + w_len+1)] for t in timeScale]

    pitches = zeros(eltype(sig),length(timeScale))
    harmonic_rates = zeros(eltype(sig),length(timeScale))
    argmins = zeros(eltype(sig), length(timeScale))

    for (i, frame) in enumerate(frames):
        #Compute YIN
        df = differenceFunction(frame, w_len, τ_max)
        cmdf = cmnd(df, τ_max)
        p = getPitch(cmdf, τ_min, τ_max, harmo_thresh)

        #Get results
        if argmin(cmdf)>τ_min:
            argmins[i] = eltype(sig)(sr / argmin(cmdf))
        if p != 0: # A pitch was found
            pitches[i] = eltype(sig)(sr / p)
            harmonic_rates[i] = cmdf[p]
        else: # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = min(cmdf)
    end
    return pitches, harmonic_rates, argmins, times
end
