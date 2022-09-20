using Test, ChaosTools
using ChaosTools.DynamicalSystemsBase, ChaosTools.DelayEmbeddings
using Random

using OrdinaryDiffEq
F = 6.886; G = 1.347; a = 0.255; b = 4.0
ds = Systems.lorenz84(; F, G, a, b)
diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9, maxiters = 1e12)
M = 600; z = 3
xg = yg = zg = range(-z, z; length = M)
grid = (xg, yg, zg)

sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = minimum.(grid), max_bounds = maximum.(grid)
)

function featurizer(A, t)
    # This is the number of boxes needed to cover the set
    g = exp(genentropy(A, 0.1; q = 0))
    return [g, minimum(A[:,1])]
end

clusterspecs = ClusteringConfig()
mapper = AttractorsViaFeaturizing(ds, featurizer, clusterspecs; diffeq, Ttr = 500)

Grange = range(1.34, 1.37; length = 21)
Gidx = 2

continuation = ClusteringAcrossParametersContinuation(mapper)

fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, Grange, Gidx, sampler;
    show_progress = true, samples_per_parameter = 100
)

# Chaotic attractor and limit cycle are naturally grouped together!