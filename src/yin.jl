using LinearAlgebra, StaticArrays

export yin
"""
##########################################################################################
# Functions and Methods to estimate Dominant Period from YIN
##########################################################################################
Fundamental frequency estimation. Based on the YIN alorgorithm
[1]: Patrice Guyot. (2018, April 19). Fast Python implementation of the Yin algorithm 
(Version v1.1.1). Zenodo. http://doi.org/10.5281/zenodo.1220947
[2]: De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for
speech and music. The Journal of the Acoustical Society of America, 111(4), 1917-1930.



"""
"""
Compute difference function of data x. This corresponds to equation (6) in [1]
Original algorithm.
:param x: audio data
:param N: length of data
:param tau_max: integration window size
:return: difference function
:rtype: list
"""
function differenceFunction_original(x, N, tau_max)
    
    df = zeros(eltype(x), tau_max)
    for tau in 2:tau_max
         for j in 1:(N-tau_max)
             df[tau] += (x[j] - x[j + tau]) ^ 2
    end
    return df
end

 """
Return fundamental period of a frame based on CMND function.
:param cmdf: Cumulative Mean Normalized Difference function
:param tau_min: minimum period for speech
:param tau_max: maximum period for speech
:param harmo_th: harmonicity threshold to determine if it is necessary to compute pitch frequency
:return: fundamental period if there is values under threshold, 0 otherwise
:rtype: float
"""
function getPitch(cmdf, τ_min, τ_max, harmo_threshold=0.1)
    τ = τ_min
    while τ < τ_max
        if cmdf[tau] < harmonic_threshold
            while τ + 1 < τ_max and cmdf[τ + 1] < cmdf[τ]
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
function cumlativeMeanNormalizedDifferenceFunction(df, N)
    cmndf = df[2:end] * range(2, length=N) / cumsum(df[2:end])
    return vcat([1],cmndf)
end

 """
Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.
:param sig: Audio signal (list of float)
:param sr: sampling rate (int)
:param w_len: size of the analysis window (samples)
:param w_step: size of the lag between two consecutives windows (samples)
:param f0_min: Minimum fundamental frequency that can be detected (hertz)
:param f0_max: Maximum fundamental frequency that can be detected (hertz)
:param harmo_tresh: Threshold of detection. The yalgorithmù return the first
minimum of the CMND fubction below this treshold.
:returns:
    * pitches: list of fundamental frequencies,
    * harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
    * argmins: minimums of the Cumulative Mean Normalized DifferenceFunction
    * times: list of time of each estimation
:rtype: tuple
"""
function compute_yin(sig::Float64, sr::Integer;
                     w_len=512,
                     w_step=256,
                     f0_min=100,
                     f0_max=500,
                     harmo_thresh=0.1)
    τ_min = Integer(sr / f0_max)
    τ_max = Integer(sr / f0_min)

    time_scale = range(0, length=(length(sig) - w_len, w_step))  # time values for each analysis window
    times = copy(time_scale) ./ eltype(sig)(sr)
    frames = [sig[(t+1):(t + w_len+1)] for t in timeScale]

    pitches = zeros(eltype(sig),length(timeScale))
    harmonic_rates = zeros(eltype(sig),length(timeScale))
    argmins = zeros(eltype(sig), length(timeScale))

    for (i, frame) enumerate(frames):
        #Compute YIN
        df = differenceFunction(frame, w_len, τ_max)
        cmdf = cumulativeMeanNormalizedDifferenceFunction(df, τ_max)
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
