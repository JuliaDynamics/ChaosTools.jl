export yin

"""
    yin(sig::Vector, sr::Int; kwargs...) -> F0s, frame_times

Estimate the fundamental frequency (F0) of the signal `sig` using the YIN algorithm [^1].
The signal `sig` is a vector of points uniformly sampled at a rate `sr`.

## Keyword arguments

* `w_len`: size of the analysis window [samples == number of points]
* `f_step`: size of the lag between two consecutive frames [samples == number of points]
* `f0_min`: Minimum fundamental frequency that can be detected [linear frequency]
* `f0_max`: Maximum fundamental frequency that can be detected [linear frequency]
* `harmonic_threshold`: Threshold of detection. The algorithm returns the first minimum of
  the CMNDF function below this threshold.
* `difference_function`: The difference function to be used (by default
  `ChaosTools.difference_function_original`).

## Description
The YIN algorithm [^CheveigneYIN2002] estimates the signal's fundamental frequency `F0` by basically
looking for the period `τ0`  which minimizes the signal's autocorrelation. This
autocorrelation is calculated for signal segments (frames), composed of two windows of
length `w_len`. Each window is separated by a distance `τ`, and the idea is that the
distance which minimizes the pairwise difference between each window is considered to be the
fundamental period `τ0` of that frame.

More precisely, the algorithm first computes the cumulative mean normalized difference
function (MNDF) between two windows of a frame for several candidate periods `τ` ranging
from `τ_min=sr/f0_max` to `τ_max=sr/f0_min`. The MNDF is defined as
```math
d_t^\\prime(\\tau) = \\begin{cases}
        1 & \\text{if} ~ \\tau=0 \\\\
        d_t(\\tau)/\\left[{(\\frac 1 \\tau) \\sum_{j=1}^{\\tau} d_{t}(j)}\\right] & \\text{otherwise}
        \\end{cases}
```
where `d_t` is the difference function:
```math
d_t(\\tau) = \\sum_{j=1}^W (x_j - x_{j+\\tau})^2
```

It then refines the local minima of the MNDF using parabolic (quadratic) interpolation. This
is done by taking each minima, along with their first neighbor points, and finding the
minimum of the corresponding interpolated parabola. The MNDF minima are substituted by the
interpolation minima. Finally, the algorithm chooses the minimum with the smallest period
and with a corresponding MNDF below the `harmonic threshold`. If this doesn't exist, it
chooses the period corresponding to the global minimum. It repeats this for frames starting
at the first signal point, and separated by a distance `f_step` (frames can overlap), and
returns the vector of frequencies `F0=sr/τ0` for each frame, along with the start times of
each frame.

As a note, the physical unit of the frequency is 1/[time], where [time] is decided by the
sampling rate `sr`. If, for instance, the sampling rate is over seconds, then the frequency
is in Hertz.

[^CheveigneYIN2002]: De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for
speech and music. The Journal of the Acoustical Society of America, 111(4), 1917-1930.
"""
function yin(sig::Vector, sr::Int; w_len::Int = 512, f_step::Int = 256, f0_min = 100,
    f0_max = 500, harmonic_threshold = 0.1,
    difference_function::Function = difference_function_original, kwargs...)

    τ_min = floor(Int64, sr / f0_max)
    τ_max = floor(Int64, sr / f0_min)

    frame_times = range(1, length(sig) - w_len - τ_max, step=f_step)  # time values for start of  each analysis window
    # times = frame_times ./ eltype(sig)(sr)

    F0s = zeros(Float64, length(frame_times))

    for (i, t) in enumerate(frame_times)
        frame = sig[ (t) : (t + τ_max+ w_len-1) ]
        df = difference_function(frame, w_len, τ_max)
        cmdf = cumulative_mean_normalized_difference_function(df)
        y_refined, τ_refined, τ_indices = refine_local_minima(cmdf)
        idx_localminimum = absolute_threshold(y_refined, τ_min, τ_max, harmonic_threshold)
        τ0 = τ_refined[idx_localminimum]
        F0s[i] = sr / τ0
    end
    return  F0s, frame_times
end

"""
    difference_function_original(x, W, τmax) -> df
Computes the difference function of `x`. `W` is the window size, and `τmax` is the maximum
period. This corresponds to equation (6) in [^CheveigneYIN2002]:
```math
d_t(\\tau) = \\sum_{j=1}^W (x_j - x_{j+\\tau})^2
```
"""
function difference_function_original(x, W, τmax)
    df = zeros(eltype(x), τmax+1) #df corresponds to τ values from 0 to τ_max
    for τ in 1:τmax
        for j in 1:W
            df[τ+1] += (x[j] - x[j + τ]) ^ 2
        end
    end
    return df
end

"""
Compute cumulative mean normalized difference function (CMND), starting from the difference
function `df`. This corresponds to equation (8).
"""
function cumulative_mean_normalized_difference_function(df)
    N = length(df)
    cmndf = df[2:end] .* range(1, N-1, step=1) ./ Float64.(cumsum(df[2:end]))
    return [1.0; cmndf]
end

"""
Returns the refined local minima of `y`, along with their refined `x` value and the
corresponding indices.  For each minimum, by the minimum of the parabola obtained by
interpolated the minimum with its first neighbors. Also returns the indices of `y`
(`x_nominal`) and the value `x` corresponding to each minima.
"""
function refine_local_minima(y)
    x_nominal = 0:length(y)-1 #nominal τ values: 0 to τ_max (cf. difference_function)
    idxs_local_minima = local_minima(y)
    if isempty(idxs_local_minima)
         @warn "No local minima found for the cumulative mean difference function. Adjusting
         the values of the minimum and maximum frequencies may fix this."
    end
    xv, yv = parabolic_interpolation(y, idxs_local_minima) #xv,yv are correction for the local minima
    y_locmin_refined = yv
    x_locmin_real= x_nominal[idxs_local_minima] .+ xv
    x_locmin_nominal = x_nominal[idxs_local_minima]
    return y_locmin_refined, x_locmin_real, x_locmin_nominal
end

"""
Applies the threshold step described in [^1]. It returns index (period) corresponding to the
first minimum below the `harmonic_threshod`. If that doesn't exist, it returns the index of the
global minimum.
"""
function absolute_threshold(localminima, τ_min, τ_max, harmonic_threshold=0.1)
    for (idx, localminimum) in enumerate(localminima)
        if localminimum ≤ harmonic_threshold
            return idx
        end
    end
    return argmin(localminima)
end

"""
Calculates the parabolic (quadratic) interpolation for all the indices of `y` given in
idxs_interpolate, and returns the coordinates of the respective parabola's minimum.
Assumes space all adjacent x values of `y` is 1.
"""
function parabolic_interpolation(y, idxs_interpolate)
    #separate x into triplets [x1,x2,x3].
    x1 = @view y[idxs_interpolate .- 1]
    x2 = @view y[idxs_interpolate]
    x3 = @view y[idxs_interpolate .+ 1]
    #calculate the vertex coordinates (xv, yv) for each triplet
    a = @. (x1 - 2x2 + x3)/2
    b = @. (x3 - x1) / 2
    xv = @. -b/2a
    yv = @. x2 - (b^2 /4a)
    return xv, yv
end

"""
Quickly written version for finding local minima by comparing first neighbors only.
#TODO: More detailed implementations available in Peaks.jl or Images.jl, maybe this should
be replaced by them.
"""
function local_minima(x)
    x1 = @view x[1:end - 2]
    x2 = @view x[2:end - 1]
    x3 = @view x[3:end]
    collect(1:length(x2))[x1 .> x2 .< x3] .+ 1
end
