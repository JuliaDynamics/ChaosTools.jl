export basin_fractions
using Statistics: mean
using Neighborhood #for kNN
using Distances
using Clustering
using Random
using Distributions

include("basins_fractions_utilities.jl")
"""
    basin_fractions(basins::Array) → fs::Dict
Calculate the fraction of the basins of attraction encoded in `basins`. The elements of
`basins` are integers, enumerating the attractor that the entry of `basins` converges to.
Return a dictionary that maps attractor IDs to their relative fractions.

In[^Menck2013] the authors use these fractions to quantify the stability of a basin of
attraction, and specifically how it changes when a parameter is changed.

[^Menck2013] : Menck, Heitzig, Marwan & Kurths. How basin stability complements the linear
stability paradigm. [Nature Physics, 9(2), 89–92](https://doi.org/10.1038/nphys2516)
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

# `sampler` is a function without any arguments that generates random initial conditions
# within a subset of the state space. You can use [`boxregion`](@ref) to create a `sampler`
# on a state space box. #TODO: implement sampler as more efficient alternative to ic_grid.
"""
function basin_fractions(ds::DynamicalSystem, feature_extraction::Function,
    ic_grid::AbstractMatrix, ic_templates::AbstractMatrix=[]; kwargs...)

Compute the global stability of basins of attraction [^Menck2013] of the given dynamical
system `ds` by integrating trajectories starting in a grid of initial conditions `ic_grid`,
then classifying their respective steady-states by extracting features from the trajectories
using the `feature_extraction` function and identifying clusters of features as an attractor
[^Stender2021]. The function returns the stability of each basin, calculated as the fraction
of initial conditions ending at each attractor, along with the labels given to each initial
condition in the grid.

The dynamical system `ds` can be either `DiscreteDynamicalSystem` or
`ContinuousDynamicalSystem`. `ic_grid` contains the initial conditions in each
column. Its size is thus the dimensionality of the system by the number of initial
conditions provided. Different types of grids can be used, such as an evenly spaced grid, a
uniform randomly spaced grid, or a gaussian distributed grid. A uniform, randomly spaced
grid guarantees that the basin stability values calculated are proportional to the true
basins' volume as the number of initial conditions goes to infinity. For nonuniform grids,
the values indicate the probabilities for observing each specific steady-state.(see
[^Stender2021] for more discussions). The `feature_extraction` function receives as input a
trajectory `u`, time vector, and optional parameters, and returns a matrix containing the
features extracted from u, with the j-th feature in the j-th column of the matrix.
Finally, `ic_templates` contains in its columns the initial conditions for
the templates, to which the features are matched in the "supervised" clustering method.
If not provided, the "unsupervised" clustering method is used.

The output `S` is a dictionary whose keys are the labels given to each attractor, and the values
are their respective global stability. The label `-1` is given to any initial condition whose
attractor could not be identified. The `class_labels` output is an array of size `N`
containing the label of each initial condition given in `ic_grid`.


## Keyword arguments
### Integration
* `T = 100` : total time for evolving initial conditions (after transient)
* `Ttr = 0` : transient time to evolve initial conditions
* `Δt = 1` : Integration time step, Δt = 1/fs, fs being the sampling frequency used in the
  bSTAB paper
* `diffeq = NamedTuple()` : other parameters for the solvers of DiffEqs
### Feature extraction and classification
* `clust_params = NamedTuple()` : other parameters for clustering method
* `extract_params = NamedTuple()`  : other parameters for the feature_extraction function
* `clust_method_norm = "kNN"` : (supervised method only) which clusterization method to
    apply. If `"kNN"`, the first-neighbor clustering is used. If `"kNN_thresholded"`, a
    subsequent step is taken, which considers as unclassified (label `-1`) the features
    whose distance to the nearest template above the `clustering_threshold`.
* `clustering_threshold = 0.0` : ("supervised" method, with `kNN_thresholded` only).
* `min_neighbors = 10` : (unsupervised method only) minimum number of neighbors
    (i.e. of similar features) each feature needs to have in order to be considered in a
    cluster (fewer than this, it is labeled as an outlier,  id=-1). This is somewhat hard
     to define, as it directly interferes with how many attractors the clustering finds.
     The authors use it equal to 10 always.

## Available featurizers
* lyapunov spectrum #TODO: future implementation?
* statistical moments

## Description
Let ``F(A)`` be the fraction of initial conditions in a region of state space
``\\mathcal{S}``, given by `ic_grid`, which are in the basin of attraction of an attractor
``A``. `basin_fractions` estimates ``F`` for attractors in
``\\mathcal{S}`` by counting which initial conditions end up in which attractors. To do
this, it evolves each trajectory `u` for `T` times past an initial transient `Ttr`, values
which have to be large enough to guarantee the trajectory ends up in its steady state. This
steady state trajectory `u` is then transformed in a vector of features, extracted using the
user-defined `feature_extraction` function. Each feature is a number useful in
characterizing the trajectory and distinguishing it from trajectories in other attrators.
For instance, a useful feature distinguishing a stable node from a stable limit cycle is the
standard deviation of `u` (zero for the node, nonzero for the limit cycle). The vectors of
features are then used to identify to which attractor each trajectory belongs (i.e. in which
basin of attractor each initial condition is in). The algorithm presents two methods to do
this. In the supervised method, the attractors are known to the user, which provides one
initial condition for each attractor in the region. The algorithm then evolves these initial
conditions, extracts their features, and uses them as templates representing the attrators.
Each trajectory is considered to belong to the nearest template, which it finds using the
first-neighbor clustering algorithm.

If the attractors are not as well-known, the alternative, unsupervised method, can be
used. It maps the vectors of features to an attractor by analysing how the features are
clustered in the feature space. Using the DBSCAN algorithm, it identifies these clusters
of features, and considers each cluster to represent an attractor. Features whose attractor
is not identified are labeled as `-1`. Otherwise, they are labeled starting from `1` in
crescent order.

These labels are then returned by the algorithm, along with the fraction ``F(A)`` for each
label (attractor). The sampling error associated with this method is given by[^Stender2021] ``e = \\sqrt{F(A)(1-F(A))/N}``, with ``N`` denoting the number of initial conditions, if the uniform
random sampling is used in `ic_grid`.


[^Menck2013] : Menck, Heitzig, Marwan & Kurths. How basin stability complements the linear
stability paradigm. [Nature Physics, 9(2), 89–92](https://doi.org/10.1038/nphys2516)

[^Stender2021] : Stender & Hoffmann, [bSTAB: an open-source software for computing the basin
stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
"""
function basin_fractions(ds::DynamicalSystem, feature_extraction::Function,
    ic_grid::AbstractMatrix, ic_templates::AbstractArray=[]; kwargs...)

    feature_array = featurizer(ds, ic_grid,  feature_extraction; kwargs...)

    if ic_templates == [] #unsupervised, no templates
        class_labels, class_errors = classify_solution(feature_array; kwargs...)
    else #supervised
        feature_templates = featurizer(ds, ic_templates, feature_extraction; kwargs...)
        class_labels, class_errors = classify_solution(feature_array, feature_templates; 
        kwargs...);
    end

    S = basin_fractions(class_labels)
    return S, class_labels
end


#----- INTEGRATION AND FEATURE EXTRACTION
"""
`featurizer` receives the grid (matrix) of initial conditions and returns their extracted
 features in a matrix, with the j-th column containing the j-th feature. To do this, it
 calls the other `featurizer` method, made for just one array of ICs.
"""
function featurizer(ds, ic_grid::AbstractMatrix, feature_extraction::Function;  kwargs...)
    Nactual = size(ic_grid, 2) #number of actual ICs
    feature_array = [Float64[] for i=1:Nactual]
    for i = 1:Nactual #TODO: implement parallelization, if necessary
        feature_array[i] = featurizer(ds, ic_grid[:,i], feature_extraction; kwargs...)
    end
    return hcat(feature_array...)
end


"""
`featurizer` receives an initial condition and returns its extracted features in a vector.
It integrates the initial condition, applies the `feature_extraction` function and returns
its output. The type of the returned vector depends on `feature_extraction`'s output.
"""
function featurizer(ds, u0::AbstractVector, feature_extraction; T=100, Ttr=0, Δt=1,
    extract_params=NamedTuple(), diffeq=NamedTuple(), kwargs...)
    u = trajectory(ds, T, u0; Ttr=Ttr, Δt=Δt, diffeq) #TODO: maybe starting an integrator
                    # and using re_init! is better
    t = Ttr:Δt:T+Ttr
    feature = feature_extraction(t, u, extract_params)
    return feature
end


#------ CLASSIFICATION OF FEATURES

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



#TODO: Perhaps we can optimize the featurizers to use `reinit' instead of initializing an integrator
# all the time by calling `trajectory``
#TODO: implement usability for sampler method. For now, we are passing ic_grid directly to
# the function, as done in the bSTAB algorithm.
#TODO: maybe we should also return more info in basin_stability's output, like the
#classification error, and the properties of each attractor (such its center, in the
#unsupervised case). I fear maybe the user can confuse which attractor corresponds to
#which ID
#TODO: test clustering with another distance metric in the clusterings
#TODO: implement for discrete systems. Will have to find a system to test also, since the
#don't provide an example for them.
#TODO: could allow the user to directly pass the templates features also, instead of
#just their initial conditions