using ChaosTools
using ChaosTools.DelayEmbeddings
using Test

@testset "Artificial test for cluster_features" begin
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

    ## Unsupervised
    correcterrors = [0, 0.01, 0.01, 0, 0.0, 0, 0.1, 0.1] #error is dist to center of cluster (cloud of features)
    clusterspecs = ClusteringConfig(; min_neighbors=1,  rescale_features=false)
    clust_labels, clust_errors = cluster_datasets(featurizer, [], attractors, clusterspecs)
    @test clust_labels == correctlabels
    @test round.(clust_errors, digits=2) == correcterrors

    ## Supervised
    correcterrors = [0, 0.01, 0.01, 0.1, 0.1, 0, 0.0, 0.2] #now error is dist to template
    correctlabels = [1,1,1, -1, -1,1,3, -1]; #for threshold at 0.1
    t = map(x->featurizer(x, []), attractor_pool)
    templates = Dict(1:length(attractor_pool) .=> t)
    clusterspecs = ClusteringConfig(; templates, min_neighbors=1, rescale_features=false, clustering_threshold=0.1)
    clust_labels, clust_errors = cluster_datasets(featurizer, [], attractors, clusterspecs)
    @test clust_labels == correctlabels
    @test round.(clust_errors, digits=2) == correcterrors
end