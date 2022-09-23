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
Find the optimal radius ε of a point neighborhood to use in DBSCAN, the unsupervised clustering
method for `AttractorsViaFeaturizing`. The basic idea is to iteratively search for the radius that
leads to the best clustering, as characterized by quantifiers known as silhouettes.
"""
function optimal_radius_dbscan_silhouette_original(features, min_neighbors, metric; num_attempts_radius=200)
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    ϵ_grid = range(minimum(feat_ranges)/num_attempts_radius, minimum(feat_ranges), length=num_attempts_radius)
    s_grid = zeros(size(ϵ_grid)) # average silhouette values (which we want to maximize)

    # vary ϵ to find the best one (which will maximize the minimum sillhoute)
    for i=1:length(ϵ_grid)
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

"""
Find the optimal radius ε of a point neighborhood to use in DBSCAN, the unsupervised clustering
method for `AttractorsViaFeaturizing`. The basic idea is to iteratively search for the radius that
leads to the best clustering, as characterized by quantifiers known as silhouettes.
"""
function optimal_radius_dbscan_silhouette(features, min_neighbors, metric; num_attempts_radius=50)
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    ϵ_grid = range(minimum(feat_ranges)/num_attempts_radius, minimum(feat_ranges), length=num_attempts_radius)
    s_grid = zeros(size(ϵ_grid)) # average silhouette values (which we want to maximize)

    # vary ϵ to find the best one (which will maximize the mean sillhoute)
    for i=1:length(ϵ_grid)
        dists = pairwise(metric, features)
        clusters = dbscan(dists, ϵ_grid[i], min_neighbors)
        sils = silhouettes_new(clusters, dists)
        s_grid[i] = mean(sils)
    end

    max, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]
end

"""
Return cluster with radius that optimizes mean silhouette. The procedure in `silhouetes`
calculates the clusters to calculate the silhouettes. Therefore, it is more optimized to
already save that cluster as you iterate to find the optimal radius, which this function
does.
"""
function findcluster_optimal_radius_dbscan_silhouette(features, min_neighbors, metric; num_attempts_radius=50)
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    ϵ_grid = range(minimum(feat_ranges)/num_attempts_radius, minimum(feat_ranges), length=num_attempts_radius)

    # vary ϵ to find the best radius (which will maximize the mean sillhoute), and already save the clusters
    optimalcluster_labels = Vector{Int64}(undef, 1)
    max_meansil = typemin(Float64)
    @inbounds for i=1:length(ϵ_grid)
        dists = pairwise(metric, features)
        dbscanresult = dbscan(dists, ϵ_grid[i], min_neighbors)
        mean_sil = mean(silhouettes_new(dbscanresult, dists))
        if mean_sil > max_meansil max_meansil = mean_sil; optimalcluster_labels = cluster_assignment(dbscanresult) end
    end

    return optimalcluster_labels
end

function optimal_radius_dbscan_silhouette_optim(features, min_neighbors, metric; num_attempts_radius=50)
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];

    # vary ϵ to find the best radius (which will maximize the mean sillhoute), and already save the clusters
    dists = pairwise(metric, features)
    f = (ϵ) -> ChaosTools.silhouettes_from_distances(ϵ, dists; min_neighbors)
    opt = Optim.optimize(f, minimum(feat_ranges)/100, minimum(feat_ranges); iterations=num_attempts_radius)
    ϵ_optimal = Optim.minimizer(opt)
end

function silhouettes_from_distances(ϵ, dists; min_neighbors)
    clusters = dbscan(dists, ϵ, min_neighbors)
    sils = silhouettes_new(clusters, dists)
    return -mean(sils)
end

function cluster_assignment(dbscanresult)
    labels = dbscanresult.assignments 
    return replace!(labels, 0=>-1)
end

function silhouettes_new(dbscanresult::DbscanResult, dists::AbstractMatrix)
    labels = dbscanresult.assignments
    clusters = [findall(x->x==i, labels) for i=1:maximum(labels)] #all clusters
    if length(clusters) == 1 return zeros(length(clusters[1])) end #all points in the same cluster -> sil = 0
    sils = zeros(length(labels))
    outsideclusters = findall(x->x==0, labels)
    for (idx_c, cluster) in enumerate(clusters) 
        @inbounds for i in cluster
            a = sum(@view dists[i, cluster])/(length(cluster)-1) #dists should be organized s.t. dist[i, cluster] i= dist from i to idxs in cluster 
            b = _calcb!(i, idx_c, dists, clusters, labels, outsideclusters)
            sils[i] = (b-a)/(max(a,b))
        end
    end
    return sils 
end

function _calcb!(i, idx_c_i, dists, clusters, labels, outsideclusters)
    min_dist_to_clstr = typemax(eltype(dists))
    for (idx_c, cluster) in enumerate(clusters)
        if idx_c == idx_c_i continue end 
        dist_to_clstr = mean(@view dists[cluster,i]) #mean distance to other clusters
        if dist_to_clstr < min_dist_to_clstr min_dist_to_clstr = dist_to_clstr end
    end
    min_dist_to_pts = typemax(eltype(dists))
    for (idx_p, point) in enumerate(outsideclusters)
        dist_to_pts = dists[point, i] #distance to points outside clusters
        if dist_to_pts < min_dist_to_pts min_dist_to_pts = dist_to_pts  end
    end 
    b = min(min_dist_to_clstr, min_dist_to_pts)
end

"""
Find the optimal radius ϵ of a point neighborhood for use in DBSCAN through the elbow method
(knee method, highest derivative method).
"""
function optimal_radius_dbscan_elbow(features, min_neighbors, metric)
    tree = searchstructure(KDTree, features, metric)
    neighbors, distances = bulksearch(tree, features, NeighborNumber(min_neighbors))
    meandistances = map(x->mean(x[2:end]), distances) #remove first element, which is dist to the element itself (:=0)
    sort!(meandistances)
    maxdiff, idx = findmax(diff(meandistances))
    ϵ_optimal =  meandistances[idx]
end

function optimal_radius_dbscan(features, min_neighbors, metric, optimal_radius_method, num_attempts_radius)
    if optimal_radius_method == "silhouette" || optimal_radius_method == "silhouettes"
        ϵ_optimal = optimal_radius_dbscan_silhouette(features, min_neighbors, metric; num_attempts_radius)
    elseif optimal_radius_method == "silhouette_original" || optimal_radius_method == "silhouettes_original"
        ϵ_optimal = optimal_radius_dbscan_silhouette_original(features, min_neighbors, metric; num_attempts_radius)
    elseif optimal_radius_method == "elbow" || optimal_radius_method == "knee"
        ϵ_optimal = optimal_radius_dbscan_elbow(features, min_neighbors, metric)
    else
        error("Unkown optimal_radius_method.")
    end
    return ϵ_optimal
end
