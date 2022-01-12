# ---- utilities for classify_solution

"""
Util function for classify_solution. It returns the size of all the DBSCAN clusters and the
assignment vector, in whch the i-th component is the cluster index of the i-th feature
"""
function cluster_props(clusters, data; include_boundary=true)
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
Util function for classify_solution. Calculates the clusters' (DbscanCluster) size and sorts
them according in decrescent order according to the size.
"""
function sort_clusters_calc_size(clusters)
    sizes = [cluster.size for cluster in clusters]
    idxsort = sortperm(sizes,rev=true)
    return clusters[idxsort], sizes[idxsort]
end

"""
Find the optimal radius ε of a point neighborhood for use in DBSCAN, in the unsupervised 
    `classify_solution`. It does so by finding the `ε` which maximizes the minimum silhouette
    of the cluster.
"""
function optimal_radius(features; min_neighbors)
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    ϵ_grid = range(minimum(feat_ranges)/200, minimum(feat_ranges), length=200)
    k_grid = zeros(size(ϵ_grid)) #number of clusters
    s_grid = zeros(size(ϵ_grid)) #min silhouette values (which we want to maximize)

    #vary ϵ to find the best one (which will maximize the minimum sillhoute)
    for i=1:length(ϵ_grid)
        clusters = dbscan(features, ϵ_grid[i], min_neighbors=min_neighbors)
        dists = pairwise(Euclidean(), features)
        class_labels = cluster_props(clusters, features)
        if length(clusters) ≠ 1 #silhouette undefined if only one cluster.
            sils = silhouettes(class_labels, dists) #values == 0 are due to boundary points
            s_grid[i] = minimum(sils[sils .!= 0.0]) #minimum silhouette value of core points
        else
            s_grid[i] = -2; #this would effecively ignore the single-cluster solution
        end
    end

    max, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]
end