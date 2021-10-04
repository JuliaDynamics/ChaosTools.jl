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
This corresponds to equation (6) in [^1]:
```math
d_t(\\tau) = \\sum_{j=1}^W (x_j - x_{j+\\tau})^2
```
`x`: audio data
`N` : length of data
`τ_max`: integration window size

[^1]: De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for
speech and music. The Journal of the Acoustical Society of America, 111(4), 1917-1930.
"""
function difference_function_original(x, N, τmax)
    df = zeros(eltype(x), τmax)
    for τ in 1:τmax-1
        for j in 1:(N-τmax)
            df[τ+1] += (x[j] - x[j + τ]) ^ 2
        end
    end
    return df
end

"""
getPitch(
    cumulative_mean_normalized_difference_functionf, τ_min, τ_max, harmonic_threshold=0.1
)

Return the fundamental period of a frame,
based on the cumulative mean normalized difference function (CMNDF).

## Arguments:
* `cmndf`: cumulative mean normalized difference of the data
* `τ_min`: minimum period
* `τ_max`: maximum period
* `harmonic_threshold`: harmonicity threshold to determine if it is necessary to compute pitch frequency

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
    τ = τ_min + 1 #+1 necessary in translation to python; makes sense also if you consider that lag=0 occurs at τ=1
    while τ < τ_max
        if (cmndf[τ] < harmonic_threshold)
            while ( (τ + 1 < τ_max) && (cmndf[τ + 1] < cmndf[τ]) )
                τ += 1
            end
            return τ
        end
        τ += 1
    end
    return 0    # if unvoiced
end


 """
Compute cumulative mean normalized difference function (CMND), returned in an Float64 array.
This corresponds to equation (8) in [1]
## Arguments:
*df: difference function array
"""
function cumulative_mean_normalized_difference_function(df)
    N = length(df)
    cmndf = df[2:end] .* range(1, N-1, step=1) ./ Float64.(cumsum(df[2:end]))
    return [1.0; cmndf]
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
* `harmonic_threshold`: Threshold of detection. The algorithm returns the first minimum of the CMND function below this treshold.
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
            difference_function = difference_function_original
        )

    τ_min = floor(Int64, sr / f0_max)
    τ_max = floor(Int64, sr / f0_min)

    time_scale = range(1, length(sig) - w_len, step=w_step)  # time values for each analysis window
    times = time_scale ./ eltype(sig)(sr)  

    pitches = zeros(Float64,length(time_scale))
    harmonic_rates = zeros(Float64,length(time_scale))
    argmins = zeros(Float64, length(time_scale))

    for (i, t) in enumerate(time_scale)
        frame = sig[ (t) : (t + w_len-1) ]
        #Compute YIN
        df = difference_function(frame, w_len, τ_max)
        cmdf = cumulative_mean_normalized_difference_function(df)
        p = getPitch(cmdf, τ_min, τ_max, harmonic_threshold)

        #Get results
        if (argmin(cmdf) > τ_min)
            argmins[i] = sr / argmin(cmdf)
        end
        if (p ≠ 0) # A pitch was found
            pitches[i] = sr / p
            harmonic_rates[i] = cmdf[p]
        else # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = minimum(cmdf)
        end
    end
    return pitches, harmonic_rates, argmins, times
end
