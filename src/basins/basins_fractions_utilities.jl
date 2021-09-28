

#------------------ GENERATING GRID OF INITIAL CONDITIONS AS IN bSTAB PAPER; MAY BE DEPRECATED AFTER IMPLEMENTING THE SAMPLER METHOD

"""
Function to generate the grid of initial conditions necessary for basins_fractions.
So far, the only implmented method is "uniform", which is probably the most used one.

* `N = 10000` : number of samples in the initial condition grid, to evolve and extract features from
% - 'uniform': uniform distribution at random (default!)
% - 'multGauss': multivariate, independent Gaussians
% - 'grid': linearly spaced grid
% - 'custom': provide your own set of initial conditions per .samplingCustomPDF
"""
function generate_ic_grid(N, min_vals, max_vals, var_dims, samplingPDF; seed=1)
    IC = generate_independent_uniform_distribution(N, min_vals, max_vals, var_dims; seed)
    return IC
end

"""
 Generates a uniform distribution at random.
 - N: number samples
 - min_vals: minimum coordinate values. ROW vector
 - max_vals: maximum coordinate values. ROW vector
 - var_dims: boolean vector indicating which DOF to vary

 - IC: resulting vectors of initial conditions [n_dof x N] == j-th column contains j-th IC
"""
function generate_independent_uniform_distribution(N, min_vals, max_vals, var_dims; seed=1)
    println("initial condition sampling strategy: uniform random");
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
Util function for classify_solution. It returns the size of all the DBSCAN clusters and the assignment vector, in whch the i-th component is the cluster index of the i-th feature
"""
function cluster_props(clusters, data; include_boundary=true)
    assign = zeros(Int, size(data)[2])
    for (idx, cluster) in enumerate(clusters)
        assign[cluster.core_indices] .= idx
        if(cluster.boundary_indices != [])
            if(include_boundary)
                assign[cluster.boundary_indices] .= idx
            else
                assign[cluster.boundary_indices] .= -1
            end
        end
    end
    return assign
end
"""
Util function for classify_solution. Calculates the clusters' (DbscanCluster) size and sorts them according to it.
"""
function sort_clusters_calc_size(clusters)
    sizes = [cluster.size for cluster in clusters]
    idxsort = sortperm(sizes,rev=true)
    return clusters[idxsort], sizes[idxsort]
end
