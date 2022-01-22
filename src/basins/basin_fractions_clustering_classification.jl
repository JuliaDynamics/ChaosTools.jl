#####################################################################################
# Classification
#####################################################################################

"""
`classify_solution` is the supervised method for classifying the trajectories, identifying
to which attractor they belong. It receives the matrix `features` of features, containing in
each column the features from one particular initial condition, and also receives the matrix
`templates`, containing in each column one template of features. The classification is done
using a first-neighbor clustering method `clust_method`, identifying, for each vector of
features, the closest template.

If `clust_method = "kNN_thresholded`, the vectors of features whose classification error is
bigger than the threshold `clustering_threshold` receive label `-1` and are considered
unclassified. The default metric `clust_method_norm` used in the clustering is Euclidean,
but other options can also be used, such as seuclidean, cityblock, chebychev, minkowski,
mahalanobis, cosine, correlation, spearman, hamming, jaccard [^Stender2021]. Typically,
these are subtypes of `<:Metric` from Distances.jl.

Returns:
* `class_labels = Array{Int64,1}`, containing the labels for each vector of features (each
  initial condition). The label is the id of the closest template, which is itself that
  template's column index in `templates`. Unnclassified features, for which the error is
  bigger than threshold if `clust_method="kNN_thresholded"` receive label `-1`.
* `class_errors = Array{Float64,1}` with the errors for each vector of features. The error
  is the distance from the vector of features to its closest template (in the specified
  metric) `clust_method_norm`.
"""
function classify_solution(features, templates; clust_method="kNN",
    clust_method_norm=Euclidean(), clustering_threshold=0.0, kwargs...)

    if clust_method == "kNN" || clust_method == "kNN_thresholded"
        template_tree = searchstructure(KDTree, templates, clust_method_norm)
        class_labels, class_errors = Neighborhood.bulksearch(template_tree, features,
         Neighborhood.NeighborNumber(1))
        
        class_labels = vcat(class_labels...); class_errors = vcat(class_errors...); #convert
        # to simple 1d arrays

        if clust_method == "kNN_thresholded" #Make label -1 if error bigger than threshold
                class_labels[class_errors .>= clustering_threshold] .= -1
        end
    else
        @warn("clustering mode not available")
    end

    return class_labels, class_errors
end

"""
Unsupervised method 'classify_function' classifies the features in an unsupervised fashion.
It assumes that features belonging to the same attractor will be clustered in feature space,
and therefore groups clustered features in a same attractor. It identifies these clusters
using the DBSCAN method. The clusters are labeled according to their size, so that cluster
No. 1 is the biggest. Features not belonging to any cluster (attractor) are given id -1.
## Keyword arguments
* min_neighbors = 10

Returns: same as supervised method `classify_function`.
"""
function classify_solution(features; min_neighbors=10, kwargs...)
    ϵ_optimal = optimal_radius(features; min_neighbors)

    #Now recalculate the final clustering with the optimal ϵ
    clusters = dbscan(features, ϵ_optimal, min_neighbors=min_neighbors)
    clusters, sizes = sort_clusters_calc_size(clusters) 
    class_labels = cluster_props(clusters, features; include_boundary=false)
    k = length(sizes[sizes .> min_neighbors]) #number of real clusters (size above minimum
    # points); this is also the number of "templates"

    #create templates/labels, assign errors
    class_errors = zeros(size(features)[2])
    K = sizes
    num_clusters = length(clusters)
    for i=1:k
        idxs_cluster = class_labels .== i
        center = mean(features[:, class_labels .== i], dims=2)[:,1]
        # templ_features[i] = center
        dists = colwise(Euclidean(), center, features[:, idxs_cluster])
        class_errors[idxs_cluster] = dists
    end

    return class_labels, class_errors
end

#####################################################################################
# Utilities
#####################################################################################
"""
Util function for `classify_solution`. It returns the size of all the DBSCAN clusters and the
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
Util function for `classify_solution`. Calculates the clusters' (DbscanCluster) size and sorts
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