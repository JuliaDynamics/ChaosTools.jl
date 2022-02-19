export correlationsum_fixedmass

using SpecialFunctions, Random

"""
    correlationsum_fixedmass(data, max_j; metric = Euclidean(), M = length(data)) → rs, ys
A fixed mass algorithm for the calculation of a fractal dimension ``\\Delta`` 
with `max_j` the maximum number of neighbours that
should be considered for the calculation. `M` defines the number of points
considered for the averaging of distances.

## Description
The calculated ``\\Delta`` approximates the information dimension.
The implementation here is due to to [^Grassberger1988], which defines
```math
\\Delta \\times \\overline{\\log r^{(j)}} \\sim Ψ(j) - \\log N
```
where `` \\Psi(j) = \\frac{\\text{d} \\log Γ(j)}{\\text{d} j}
`` is the digamma function, `rs` = ``\\overline{\\log r^{(j)}}`` and 
`ys` = ``\\Psi(j) - \\log N`` (``N`` is the length of the data).

``\\Delta`` can be computed by using `linear_region(rs, ys)`.

[^Grassberger1988]: 
    Peter Grassberger (1988) [Finite sample Corrections to Entropy and Dimension Estimates,
    Physics Letters A 128(6-7)](https://doi.org/10.1016/0375-9601(88)90193-4)
"""
function correlationsum_fixedmass(data, max_j; metric = Euclidean(), M = length(data), w = 0)
    N = length(data)
    @assert M ≤ N
    tree = searchstructure(KDTree, data, metric)
    searchdata = N == M ? data : view(data, view(randperm(N), 1:M))
    _, distances = bulksearch(tree, searchdata, NeighborNumber(max_j), Theiler(w))
    # The epsilons define the left side of the equation
    ys = [digamma(j) - log(N) for j in 1:max_j]
    # Holds the mean value of the logarithms of the distances.
    rs = zeros(max_j)
    for dists in distances
        for j in 1:max_j
            rs[j] += log(dists[j])
        end
    end
    return rs ./ M, ys
end
