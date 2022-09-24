using ChaosTools
using ChaosTools.DelayEmbeddings
using Test

function featurizer(A, t)
    return [maximum(A[:,1]), maximum(A[:,2])]
end
function cluster_datasets(featurizer, t, datasets, clusterspecs)
    features = [featurizer(datasets[i], t) for i=1:length(datasets)]
    return cluster_features(features, clusterspecs)
end
attractor_pool = [[1 1], [20 20], [30 30]];
errors = [[0.0 0.0], [0.0 -0.01], [0.0 +0.01], [0.1 0.0], [0.1 0],
    [0.0 0.0], [0.0 0.0], [0.2 0]
]
correctlabels = [1,1,1,2,2,1,3,3]
a = attractor_pool[correctlabels] .+ errors
attractors = Dict(1:length(a) .=> Dataset.(a; warn = false));

# Unsupervised
clusterspecs = ClusteringConfig(; min_neighbors=1,  rescale_features=false)
clust_labels = cluster_datasets(featurizer, [], attractors, clusterspecs)
@test clust_labels == correctlabels

# Supervised
t = map(x->featurizer(x, []), attractor_pool)
template_labels = [i for i ∈ eachindex(attractor_pool)]
correctlabels = [1, 1, 1, -1, -1, 1, 3, -1]; # for threshold at 0.1
templates = Dict(template_labels.=> t)
clusterspecs = ClusteringConfig(;
    templates, min_neighbors=1, rescale_features=false, clustering_threshold=0.1
)
clust_labels = cluster_datasets(featurizer, 0, attractors, clusterspecs)
@test clust_labels == correctlabels