"""
threshold_pos(timeseries, k)
Compute threshold above which are extreme events.
it was first described in Oceanic Rogue Waves DOI: https://doi.org/10.1146/annurev.fluid.40.111406.102203

threshold_neg(timeseries, k)
Compute threshold below which are extreme events.
It was described in Intermittent large amplitude bursting in Hindmarsh-Rose neuron model DOI: 10.1109/DCNA53427.2021.9586844
"""
threshold_pos(timeseries, k) = Statistics.mean(timeseries) + k*Statistics.std(timeseries)
threshold_neg(timeseries, k) = Statistics.mean(timeseries) - k*Statistics.std(timeseries)

"""
peaks(x)
Compute peaks for timeseries
"""
function peaks(timeseries)

    peaks_ = Float64[]
    len_ = length(timeseries)
    for i in range(2, len_ - 1, step = 1)
        if timeseries[i-1] < timeseries[i] > timeseries[i+1]
            push!(peaks_, timeseries[i])
        end
    end
    return peaks_
end

"""
detect_number_ee(timeseries, k, type_threshold = "pos")
Compute number of extreme events in timeseries
"""
function detect_number_ee(timeseries, k, type_threshold = "pos")

    if type_threshold == "pos"
        Hs = threshold_pos(timeseries, k)
    else
        Hs = threshold_neg(timeseries, k)
    end

    df_peaks, _ = peaks(timeseries);
    counts = length(df_peaks[df_peaks.>=threshold_pos(df_peaks, 6)])
    return counts
end