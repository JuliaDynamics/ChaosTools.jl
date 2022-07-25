#####################################################################################
# Utilities
#####################################################################################
"""
Util function for `classify_features`. Returns the assignment vector, in which the i-th
component is the cluster index of the i-th feature
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

"""
Util function for `classify_features`. Returns the (`DbscanCluster`) sorted in decreasing order
of their size and their sizes.
"""
function sort_clusters_calc_size(clusters)
    sizes = [cluster.size for cluster in clusters]
    idxsort = sortperm(sizes; rev = true)
    return clusters[idxsort], sizes[idxsort]
end

"""
Find the optimal radius ε of a point neighborhood for use in DBSCAN, in the unsupervised
`classify_features`. It does so by finding the `ε` which maximizes the average silhouette
of the clusters. If only one cluster is found, assigned silhouette is 0.
"""
function optimal_radius_dbscan_silhouette(features, min_neighbors, metric)
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    ϵ_grid = range(minimum(feat_ranges)/200, minimum(feat_ranges), length=200)
    s_grid = zeros(size(ϵ_grid)) # average silhouette values (which we want to maximize)

    # vary ϵ to find the best one (which will maximize the minimum sillhoute)
    for i=1:length(ϵ_grid)
        clusters = dbscan(features, ϵ_grid[i]; min_neighbors)
        dists = pairwise(metric, features)
        class_labels = cluster_assignment(clusters, features)
        if length(clusters) ≠ 1 # silhouette undefined if only one cluster
            sils = silhouettes(class_labels, dists)
            s_grid[i] = mean(sils)
        else
            s_grid[i] = 0; # considers single-cluster solution on the midpoint (following Wikipedia)
        end
    end

    max, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]
end

"""
Find the optimal radius ϵ of a point neighborhood for use in DBSCAN through the elbow method
(knee method, highest derivative method). Works by calculating the distance of each point to its
k-nearest-neighbors (with `k=min_neighbors`) and finding the distance corresponding to the
highest derivative in the curve of the distances, sorted in ascending order. This distance
is chosen as the optimal radius. Described in the following references:
[^1]: Ester, Kriegel, Sander and Xu: A Density-Based Algorithm for Discovering Clusters in
Large Spatial Databases with Noise
[^2]: Schubert, Sander, Ester, Kriegel and Xu: DBSCAN Revisited, Revisited: Why and How You
Should (Still) Use DBSCAN
"""
function optimal_radius_dbscan_elbow(features, min_neighbors, metric)
    tree = searchstructure(KDTree, features, metric)
    neighbors, distances = bulksearch(tree, features, NeighborNumber(min_neighbors))
    meandistances = map(x->mean(x[2:end]), distances) #remove first element, which is dist to the element itself (:=0)
    sort!(meandistances)
    maxdiff, idx = findmax(diff(meandistances))
    ϵ_optimal =  meandistances[idx]
end

function optimal_radius_dbscan(features, min_neighbors, metric, optimal_radius_method)
    if optimal_radius_method == "silhouette" || optimal_radius_method == "silhouettes"
        ϵ_optimal = optimal_radius_dbscan_silhouette(features, min_neighbors, metric)
    elseif optimal_radius_method == "elbow" || optimal_radius_method == "knee"
        ϵ_optimal = optimal_radius_dbscan_elbow(features, min_neighbors, metric)
    else
        error("Unkown optimal_radius_method.")
    end
    return ϵ_optimal
end
