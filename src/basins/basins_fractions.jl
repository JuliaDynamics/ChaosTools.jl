export basin_fractions
using Statistics: mean
using Neighborhood #for kNN
using Distances
using DataFrames
using Clustering
using Random
using Distributions

include("basins_fractions_utilities.jl")
"""
    basin_fractions(basins::Array) → fs::Dict
Calculate the fraction of the basins of attraction encoded in `basins`.
The elements of `basins` are integers, enumerating the attractor that the entry of `basins`
converges to. Return a dictionary that maps attractor IDs to their relative fractions.

In[^Menck2013] the authors use these fractions to quantify the stability of a basin of
attraction, and specifically how it changes when a parameter is changed.

[^Menck2013]: Menck, Heitzig, Marwan & Kurths. How basin stability complements the linear stability paradigm. [Nature Physics, 9(2), 89–92](https://doi.org/10.1038/nphys2516)
"""
function basin_fractions(basins::AbstractArray)
    fs = Dict{eltype(basins), Float64}()
    ids = unique(basins)
    N = length(basins)
    for ξ in ids
        B = count(isequal(ξ), basins)
        fs[ξ] = B/N
    end
    return fs
end

"""
    basin_fractions(ds::DynamicalSystem, sampler, featurizer; kwargs...)
Compute the fraction of basins of attraction of the given dynamical system.
`sampler` is a function without any arguments that generates random initial conditions
within a subset of the state space. You can use [`boxregion`](@ref) to create a `sampler`
on a state space box.
`featurizer` is a function that takes as an input an initial condition and outputs a
vector of "features". Various pre-defined featurizers are available, see below.



## Main arguments
* ds: DynamicalSystem; so far only Continuous implemented
* feature_extraction::Function : function that receives as input the trajectory, time vector, and optional parameters to return a matrix containing the features extracted from y, with the j-th feature in the j-th column of the matrix.
* ic_grid::AbstractMatrix : matrix containing the grid of initial conditions; Each column contains one IC; so it is of size dimensionality x N
* clust_mod = "unsupervised" : clustering method chosen, can be "supervised" or "unsupervised". If "supervised", user needs to input the templates features also


## Keywork arguments
* T = 100 : total time for evolving initial conditions (after transient)
* Ttr = 0 : transient time to evolve initial conditions
* Δt = 1 : Integration time step, Δt = 1/fs, fs being the sampling frequency used in the bSTAB paper
* y0 = []: initial conditions for the templates in the "supervised case"
* templates_labels : (for supervised) labels for the templates
* clust_params = NamedTuple() : other parameters for clustering method
* diffeq = NamedTuple() : other parameters for the solvers of DiffEqs
* extract_params = NamedTuplee() : other parameters for the feature_extraction function
* min_neighbors = 10 : (for unsupervised) minimum number of neighbors (similar features) each feature needs to have in order to be considered in a cluster (fewer than this, it is labeled as an outlier, id=-1). This is somewhat hard to define, as it directly interferes with how many attractors the clustering finds. The authors use it equal to 10 always.

## Available featurizers
* lyapunov spectrum
* statistical moments
* whatever else in the original paper, which should be trivial to implement here


## Description
Let ``F(A)`` be the fraction the basin of attraction of an attractor ``A`` has in the
chosen state space region ``\\mathcal{S}`` given by `sampler`. `basin_fractions` estimates ``F`` for all
attractors in ``\\mathcal{S}`` by randomly sampling `N` initial conditions and counting which
ones end up in which attractors. The error of this approach for each fraction is given by[^Menck2013]
``e = \\sqrt{F(A)(1-F(A))}``.

In `basin_fractions` we do not actually identify attractors, as e.g. done in [`basins_of_attraction`](@ref).
Instead we follow the approach of Stender & Hoffmann[^Stender2021] which transforms each
initial condition into a vector of features, and uses these to efficiently map initial
conditions to attractors using a clustering technique.


[^Menck2013]: Menck, Heitzig, Marwan & Kurths. How basin stability complements the linear stability paradigm. [Nature Physics, 9(2), 89–92](https://doi.org/10.1038/nphys2516)

[^Stender2021]: Stender & Hoffmann, [bSTAB: an open-source software for computing the basin stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
"""
function basin_fractions(ds::DynamicalSystem, feature_extraction::Function, ic_grid::AbstractMatrix, clust_mod="supervised"; kwargs...)
    #-- Integrate ICs to find features
    # println("Integrating ICs to find features.")
    feature_array = featurizer(ds, ic_grid,  feature_extraction; kwargs...)

    #-- Classify features
    # println("Classifying features in $(clust_mod) manner.")
    if(clust_mod == "supervised")
        y0 = kwargs[:y0]
        feature_templates = featurizer(ds, y0, feature_extraction; kwargs...) #TODO: could allow the user to directly pass the templates featuers also, instead of just their initial conditions
        class_labels, class_errors, template_labels_id = classify_solution(feature_array, feature_templates; kwargs...);
    else
        class_labels, class_errors, template_id = classify_solution(feature_array; kwargs...);
    end

    #-- Compute basin stability values
    S = basin_fractions(class_labels)
    return S, class_labels
end


#----- INTEGRATION AND FEATURE EXTRACTION
"""
`featurizer` receives the grid of ICs and returns their extracted features in a matrix.
* templates_features :  matrix containing the template features used in the supervised method, j-th column containing the j-th feature template
Returns Array{Float64, 2}, with the j-th column containing the features of the j-th trajectory
"""
function featurizer(ds, ic_grid::AbstractMatrix, feature_extraction::Function;  kwargs...)
    Nactual = size(ic_grid, 2) #number of actual ICs (may be a bit more than N due to rounding)
    feature_array = [Float64[] for i=1:Nactual]
    for i = 1:Nactual #TODO: implement parallelization, if necessary
        feature_array[i] = featurizer(ds, ic_grid[:,i], feature_extraction; kwargs...)
    end
    return hcat(feature_array...)
end


"""
`featurizer` is a function that takes as an input an initial condition and outputs a
vector of "features". Various pre-defined featurizers are available, see below.
## Available featurizers
* lyapunov spectrum
* statistical moments
* whatever else in the original paper, which should be trivial to implement here
The featurizer thus: 1. integrates the initial conditions; extracts its features via a (at-least-for-now) user defined feature extraction function
Return is the return type of feature_extraction
"""
function featurizer(ds, u0::AbstractVector, feature_extraction; T=100, Ttr=0, Δt=1, extract_params=NamedTuple(), diffeq=NamedTuple(), kwargs...)
        # perform the time integration
        y = trajectory(ds, T, u0; Ttr=Ttr, Δt=Δt, diffeq) #TODO: maybe starting an integrator and using re_init! is better # reinit!(integ, ic_grid[i,:])
        t = Ttr:Δt:T+Ttr
        # extract descriminative features from the time signals
        feature = feature_extraction(t, y, extract_params)
        return feature
end


#------ CLASSIFICATION OF FEATURES

"""
`classify_solution` receives the features, extracted from the trajectories evolved on the original grid,
and classifies these according to the templates received. This is therefore the supervised method.
The templates are also evolved from the initial conditions y0, and are a matrix with each template in a column.
The classification is done using a first-neighbor clustering method, identifying, for each feature,
the closest template.

label : for each feature, it is the id of the closest template == index of the template in y0
error : distance from the feature to its closest template (in the specified metric)
class_labels = Array{Int64,1} with the labels for each feature
class_errors = Array{Float64,1} with the errors for each feature
template_ids = Array{Int64, 1} with the labels for each template. Not really needed.

dist_norm = metric for calculating the distance in kNN. Possible examples are 'seuclidean', 'cityblock', 'chebychev', 'minkowski', 'mahalanobis', 'cosine', 'correlation', 'spearman', 'hamming', 'jaccard'.
Typically, these are subtypes of '<:Metric' from Distances.jl.
"""
function classify_solution(features, templates; clustMethod="kNN", clustMethodNorm=Euclidean(), templates_labels::Array{String, 1}=String[], clustering_threshold=0.0, kwargs...)
    # find the classification by the nearest neighbors
    if clustMethod == "kNN" || clustMethod == "kNN_thresholded" #k=1 nearest neighbor classification; find nearest neighbor through kNN (k=1)
        template_tree = searchstructure(KDTree, templates, clustMethodNorm)
        class_labels, class_errors = Neighborhood.bulksearch(template_tree, features, Neighborhood.NeighborNumber(1))
        class_labels = vcat(class_labels...); class_errors = vcat(class_errors...); #convert to simple 1d arrays
        if clustMethod == "kNN_thresholded"
                class_labels[class_errors .>= clustering_threshold] .= -1 #Make label -1 if error bigger than threshold
        end
    else
        @warn("clustering mode not available")
    end

    k = size(templates)[2] #number of templates
    template_ids = [1:k; -1]; #template labels

    return class_labels, class_errors, template_ids
    # class_result = DataFrame((label=class_label, error=class_errors))
    # return class_result, class_template_labels
end



"""
Unsupervised method
'classify_function' classifies the features in an unsupervised fashion. It assumes that features belonging to the
same attractor will be clustered in feature space, and therefore groups clustered features in a same attractor.
It identifies these clusters using the DBSCAN method. The clusters are labeled according to their size, so that cluster No. 1 is the biggest.
Features not belonging to any cluster (attractor) are given id -1.
## Keyword arguments
* min_neighbors = 10
"""
function classify_solution(features; min_neighbors=10, kwargs...)
    #--find optimal ϵ
    #initialize
    feat_ranges = maximum(features, dims=2)[:,1] .- minimum(features, dims=2)[:,1]; # (within each type of feature, calc the gap)
    ϵ_grid = range(minimum(feat_ranges)/200, minimum(feat_ranges), length=200) #This hard-coded 200 is perhaps not ideal. TODO: Should we change it?
    k_grid = zeros(size(ϵ_grid)) # number of clusters
    s_grid = zeros(size(ϵ_grid)) #min silhouette values (which we want to maximize)

    #vary ϵ to find the best one (which will maximize the minimum sillhoute)
    for i=1:length(ϵ_grid)
        clusters = dbscan(features, ϵ_grid[i], min_neighbors=min_neighbors)
        dists = pairwise(Euclidean(), features)
        class_labels = cluster_props(clusters, features)
        sils = silhouettes(class_labels, dists) #values == 0 are due to boundary points
        s_grid[i] = minimum(sils[sils .!= 0.0]) #minimum silhouette value of core points
    end

    #find max == optimal ϵ
    max, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]

    #Now recalculate the final clustering with the optimal ϵ
    clusters = dbscan(features, ϵ_optimal, min_neighbors=min_neighbors) #points: the d×n matrix of points. points[:, j] is a d-dimensional coordinates of j-th point
    clusters, sizes = sort_clusters_calc_size(clusters) #calcs their sizes and sorts them in descrecent order according to their sizes
    class_labels = cluster_props(clusters, features; include_boundary=false)
    k = length(sizes[sizes .> min_neighbors]) #number of real clusters (size above minimum points); this is also the number of "templates"

    #create templates/labels, assign errors
    class_errors = zeros(size(features)[2])
    K = sizes
    num_clusters = length(clusters)
    for i=1:k
        idxs_cluster = class_labels .== i
        center = mean(features[:, class_labels .== i], dims=2)[:,1]
        # templ_features[i] = center #TODO: could return this also
        dists = colwise(Euclidean(), center, features[:, idxs_cluster])
        class_errors[idxs_cluster] = dists
    end
    template_ids = [1:k; -1]; #template labels

    return class_labels, class_errors, template_ids
    # class_result = DataFrame((label=class_labels, error=class_error))
    # return class_result, class_cluster_labels
end




# TODO: Perhaps we can optimize the featurizers to instead of initializing an integrator # all the time by calling `trajectory`, to instead use `reinit!`
# TODO: implement usability for sampler method. For now, we are passing ic_grid directly to the function, as done in the bSTAB algorithm.
# TODO: maybe we should also return more info in basin_stability's output, like the classification error, and the properties of each attractor (such its center, in the unsupervised case). I fear maybe the user can confuse which attractor corresponds to which ID
