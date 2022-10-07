export ClusteringAcrossParametersContinuation
import ProgressMeter

# TODO: Make this into a struct
function ClusteringAcrossParametersContinuation(
        mapper::AttractorsViaFeaturizing;
        info_extraction = mean_across_features
        # TODO: Here we can add more keywords regarding how to cluster across parameters.
    )
    return (; mapper, info_extraction)
end

function mean_across_features(fs)
    means = zeros(length(first(fs)))
    N = length(fs)
    for f in fs
        for i in eachindex(f)
            means[i] += f[i]
        end
    end
    return means ./ N
end

function basins_fractions_continuation(
        continuation::NamedTuple, prange, pidx, ics::Function;
        samples_per_parameter = 100, show_progress = true, w = 1
    )
    spp, n = samples_per_parameter, length(prange)
    (; mapper, info_extraction) = continuation
    progress = ProgressMeter.Progress(n;
        desc="Continuating basins fractions:", enabled=show_progress
    )
    # Extract the first possible feature to initialize the features container
    feature = extract_features(mapper, ics; N = 1)
    features = Vector{typeof(feature[1])}(undef, n*spp)
    # Collect features
    for (i, p) in enumerate(prange)
        set_parameter!(mapper.integ, pidx, p)
        current_features = extract_features(mapper, ics; show_progress, N = spp)
        features[((i - 1)*spp + 1):i*spp] .= current_features
        next!(progress)
    end
    # TODO: Here we can have an additional step that adds the parameter value to features,
    # or configures different way to cluster weighted by parameter values

    # Construct basic distance matrix
    Dk = [ sum(abs.(x .- y)) for x in features, y in features]

    # use parameter distance weight (w is the weight for one parameter only)
    par_array = kron(prange, ones(spp))
    for k in 1:length(par_array)
        for j in 1:length(par_array)
            Dk[k,j] += w*abs(par_array[k] - par_array[j])
        end
    end

    # Cluster them
    db_res = dbscan(Dk, 5, round(Int, 5))
    cluster_labels = db_res.assignments
    # cluster_labels, = cluster_features(features, mapper.cluster_config)

    # And finally collect/group stuff into their dictionaries
    fractions_curves = Vector{Dict{Int, Float64}}(undef, n)
    dummy_info = info_extraction(feature)
    attractors_info = Vector{Dict{Int, typeof(dummy_info)}}(undef, n)
    for i in 1:n
        current_labels = view(cluster_labels, ((i - 1)*spp + 1):i*spp)
        current_ids = unique(current_labels)
        # getting fractions is easy; use predefined function
        fractions_curves[i] = basins_fractions(current_labels, current_ids)
        attractors_info[i] = Dict(id => info_extraction(
            view(current_labels, findall(isequal(id), current_labels)))
            for id in current_ids
        )
    end
    return fractions_curves, attractors_info
end
