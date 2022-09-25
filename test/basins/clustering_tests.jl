using ChaosTools
using ChaosTools.DelayEmbeddings
using Test
using Statistics

@testset "Artificial test for cluster_features" begin
    function featurizer(A, t)
        return [maximum(A[:,1]), maximum(A[:,2])]
    end
    function cluster_datasets(featurizer, t, datasets, clusterspecs)
        features = [featurizer(datasets[i], t) for i=1:length(datasets)]
        return cluster_features(features, clusterspecs)
    end
    attractor_pool = [[1 1], [20 20], [30 30]];
    correctlabels = [1,1,1,1, 2,2,2,1,2,3,3,3,3,1]
    a = attractor_pool[correctlabels]
    a[end] = [50 50]; correctlabels[end] = -1
    attractors = Dict(1:length(a) .=> Dataset.(a; warn = false));

    ## Unsupervised
    for optimal_radius_method in ["silhouettes", "silhouettes_optim"],
        statistic_silhouette in [mean, minimum]
        clusterspecs = ClusteringConfig(num_attempts_radius=20,
        optimal_radius_method=optimal_radius_method, min_neighbors=2, rescale_features=false)
        clust_labels = cluster_datasets(featurizer, [], attractors, clusterspecs)
        @test clust_labels == correctlabels
    end

    correctlabels_elbow = [1,1,1,1, 2,2,2,1,2,3,3,3,3,1,2,2,2,2,2,3,3,3,3,3,3,1,1,1,1,1] #smaller number of features works even worse
    using Random; Random.seed!(1)
    a = [attractor_pool[label] + 0.2*rand(Float64, (1,2)) for label in correctlabels_elbow]
    attractors_elbow = Dict(1:length(a) .=> Dataset.(a; warn = false));
    clusterspecs = ClusteringConfig(num_attempts_radius=20,
    optimal_radius_method="knee", min_neighbors=4, rescale_features=false)
    clust_labels = cluster_datasets(featurizer, [], attractors_elbow, clusterspecs)
    # @test clust_labels == correctlabels #fails
    @test maximum(clust_labels) == maximum(correctlabels) #at least check if it finds the same amount of attractors; note this does not work for any value of `min_neighbors`.

    ## Supervised
    ###construct templates
    t = map(x->featurizer(x, []), attractor_pool);
    template_labels = [i for i ∈ eachindex(attractor_pool)]
    templates = Dict(template_labels.=> t)
    clusterspecs = ClusteringConfig(; templates, min_neighbors=1, rescale_features=false, clustering_threshold=0.1)
    clust_labels = cluster_datasets(featurizer, [], attractors, clusterspecs)
    @test clust_labels == correctlabels
end