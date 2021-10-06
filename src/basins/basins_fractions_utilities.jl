

#------------------ GENERATING GRID OF INITIAL CONDITIONS AS IN bSTAB PAPER; MAY BE
#DEPRECATED AFTER IMPLEMENTING THE SAMPLER METHOD

"""
Function to generate `N` initial conditions, chosen in a region defined in a region of state
space with `min_vals minimum coordinates and `max_vals` maximum coordinates. The sampling
is done according to the `sampling_pdf` method. This method can be:
* 'uniform': random uniform distribution (default!)
* 'multGauss': multivariate, independent Gaussians
* 'grid': linearly spaced grid

The region can also have fewer dimensions than state space. The relevant dimensions are given
in `var_dims`.

##Keyword arguments
* seed : seed for the random number generator

Returns a matrix, containing in the j-th column the j-th initial condition.
"""
function generate_ic_grid(N, min_vals, max_vals, var_dims::Array{Bool, 1}, 
    sampling_pdf="uniform"; seed=1)
    if sampling_pdf == "uniform" 
        IC = generate_independent_uniform_distribution(N, min_vals, max_vals, var_dims; seed)
    end
    return IC
end

"""
 Generates a uniform distribution at random.
"""
function generate_independent_uniform_distribution(N, min_vals, max_vals, var_dims; seed=1)
    ndof =  length(min_vals); # degrees of freedom
    IC = zeros(Float64, (ndof, N)); # initialize
    rng = MersenneTwister(seed)
    for i=1:ndof
        if var_dims[i]
            dist = Uniform(min_vals[i], max_vals[i])
            IC[i, :] =  rand(rng, dist, N)
        else
            IC[i, :] = min_vals[i].*IC[i,:];
        end
    end
    return IC
end



function generate_uniformly_spaced_grid(N, min_vals, max_vals) #TODO: implement
end

function generate_independent_multivariate_gaussians() #TODO: implement
end


#---- UTILITIES FOR classify_solution


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
    of the clus
    #TODO: this seems a good method, but there may be better ones...
"""
function optimal_radius(features; min_neighbors)
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1];
    ϵ_grid = range(minimum(feat_ranges)/200, minimum(feat_ranges), length=200)
    #TODO: this hard-coded 200 is perhaps not ideal. Should we change it?
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
            #TODO: what should be done in this case?
            s_grid[i] = -2; #this would effecively ignore the single-cluster solution
        end
    end

    max, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]
end