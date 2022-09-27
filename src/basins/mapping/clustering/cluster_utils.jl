#####################################################################################
# Utilities
#####################################################################################
"""
Util function for `classify_features`. Returns the assignment vector, in which the i-th
component is the cluster index of the i-th feature.
"""
function cluster_assignment(clusters, data; include_boundary=true)
    assign = zeros(Int, size(data)[2])
    for (idx, cluster) in enumerate(clusters)
        assign[cluster.core_indices] .= idx
        if cluster.boundary_indices != []
            if include_boundary
                assign[cluster.boundary_indices] .= idx
            else
                assign[cluster.boundary_indices] .= -1
            end
        end
    end
    return assign
end
function cluster_assignment(dbscanresult::Clustering.DbscanResult)
    labels = dbscanresult.assignments
    return replace!(labels, 0=>-1)
end

"""
Calculate silhouettes. A bit slower than the implementation in `Clustering.jl` but seems
to be more robust. The latter seems to be incorrect in some cases.
"""
function silhouettes_new(dbscanresult::Clustering.DbscanResult, dists::AbstractMatrix)
    labels = dbscanresult.assignments
    clusters = [findall(x->x==i, labels) for i=1:maximum(labels)] #all clusters
    if length(clusters) == 1 return zeros(length(clusters[1])) end #all points in the same cluster -> sil = 0
    sils = zeros(length(labels))
    outsideclusters = findall(x->x==0, labels)
    for (idx_c, cluster) in enumerate(clusters)
        @inbounds for i in cluster
            a = sum(@view dists[i, cluster])/(length(cluster)-1) #dists should be organized s.t. dist[i, cluster] i= dist from i to idxs in cluster
            b = _calcb!(i, idx_c, dists, clusters, outsideclusters)
            sils[i] = (b-a)/(max(a,b))
        end
    end
    return sils
end

function _calcb!(i, idx_c_i, dists, clusters, outsideclusters)
    min_dist_to_clstr = typemax(eltype(dists))
    for (idx_c, cluster) in enumerate(clusters)
        idx_c == idx_c_i && continue
        dist_to_clstr = mean(@view dists[cluster,i]) #mean distance to other clusters
        if dist_to_clstr < min_dist_to_clstr min_dist_to_clstr = dist_to_clstr end
    end
    min_dist_to_pts = typemax(eltype(dists))
    for point in outsideclusters
        dist_to_pts = dists[point, i] # distance to points outside clusters
        if dist_to_pts < min_dist_to_pts
            min_dist_to_pts = dist_to_pts
        end
    end
    return min(min_dist_to_clstr, min_dist_to_pts)
end

#####################################################################################
# Optimal radius dbscan
#####################################################################################
function optimal_radius_dbscan(features, min_neighbors, metric, optimal_radius_method,
    num_attempts_radius, silhouette_statistic)
    if optimal_radius_method == "silhouettes"
        ϵ_optimal = optimal_radius_dbscan_silhouette(
            features, min_neighbors, metric, num_attempts_radius, silhouette_statistic
        )
    elseif optimal_radius_method == "silhouettes_optim"
        ϵ_optimal = optimal_radius_dbscan_silhouette_optim(
            features, min_neighbors, metric, num_attempts_radius, silhouette_statistic
        )
    elseif optimal_radius_method == "knee"
        ϵ_optimal = optimal_radius_dbscan_knee(features, min_neighbors, metric)
    else
        error("Unkown `optimal_radius_method`.")
    end
    return ϵ_optimal
end

"""
Find the optimal radius ε of a point neighborhood to use in DBSCAN, the unsupervised
clustering method for `AttractorsViaFeaturizing`. Iteratively search
for the radius that leads to the best clustering, as characterized by quantifiers known as
silhouettes. Does a linear (sequential) search.
"""
function optimal_radius_dbscan_silhouette(features, min_neighbors, metric,
        num_attempts_radius, silhouette_statistic
    )
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    ϵ_grid = range(
        minimum(feat_ranges)/num_attempts_radius, minimum(feat_ranges);
        length=num_attempts_radius
    )
    s_grid = zeros(size(ϵ_grid)) # silhouette statistic values (which we want to maximize)

    # vary ϵ to find the best one (which will maximize the mean sillhoute)
    for i in eachindex(ϵ_grid)
        dists = pairwise(metric, features)
        clusters = dbscan(dists, ϵ_grid[i], min_neighbors)
        sils = silhouettes_new(clusters, dists)
        s_grid[i] = silhouette_statistic(sils)
    end

    _, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]
    return ϵ_optimal
end

"""
Same as `optimal_radius_dbscan_silhouette`,
but find minimum via optimization with Optim.jl.
"""
function optimal_radius_dbscan_silhouette_optim(
        features, min_neighbors, metric, num_attempts_radius, silhouette_statistic
    )
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    # vary ϵ to find the best radius (which will maximize the mean sillhoute)
    dists = pairwise(metric, features)
    f = (ϵ) -> ChaosTools.silhouettes_from_distances(
        ϵ, dists, min_neighbors, silhouette_statistic
    )
    opt = Optim.optimize(
        f, minimum(feat_ranges)/100, minimum(feat_ranges); iterations=num_attempts_radius
    )
    ϵ_optimal = Optim.minimizer(opt)
    return ϵ_optimal
end

function silhouettes_from_distances(ϵ, dists, min_neighbors, silhouette_statistic=mean)
    clusters = dbscan(dists, ϵ, min_neighbors)
    sils = silhouettes_new(clusters, dists)
    # We return minus here because Optim finds minimum; we want maximum
    return -silhouette_statistic(sils)
end

"""
Find the optimal radius ϵ of a point neighborhood for use in DBSCAN through the elbow method
(knee method, highest derivative method).
"""
function optimal_radius_dbscan_knee(features, min_neighbors, metric)
    tree = searchstructure(KDTree, features, metric)
    # Get distances, excluding distance to self (hence the Theiler window)
    features_vec = [features[:,j] for j=1:size(features,2)]
    _, distances = bulksearch(tree, features_vec, NeighborNumber(min_neighbors), Theiler(0))
    meandistances = map(mean, distances)
    sort!(meandistances)
    maxdiff, idx = findmax(diff(meandistances))
    ϵ_optimal = meandistances[idx]
    return ϵ_optimal
end


# The following function is left here for reference. It is not used anywhere in the code.
# It is the original implementation we have written based on the bSTAB paper.
"""
Find the optimal radius ε of a point neighborhood to use in DBSCAN, the unsupervised clustering
method for `AttractorsViaFeaturizing`. The basic idea is to iteratively search for the radius that
leads to the best clustering, as characterized by quantifiers known as silhouettes.
"""
function optimal_radius_dbscan_silhouette_original(features, min_neighbors, metric; num_attempts_radius=200)
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    ϵ_grid = range(minimum(feat_ranges)/num_attempts_radius, minimum(feat_ranges), length=num_attempts_radius)
    s_grid = zeros(size(ϵ_grid)) # average silhouette values (which we want to maximize)

    # vary ϵ to find the best one (which will maximize the minimum sillhoute)
    for i in eachindex(ϵ_grid)
        clusters = dbscan(features, ϵ_grid[i]; min_neighbors)
        dists = pairwise(metric, features)
        class_labels = cluster_assignment(clusters, features)
        if length(clusters) ≠ 1 # silhouette undefined if only one cluster
            sils = silhouettes(class_labels, dists)
            s_grid[i] = minimum(sils)
        else
            s_grid[i] = 0; # considers single-cluster solution on the midpoint (following Wikipedia)
        end
    end

    max, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]
end
