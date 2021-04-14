export correlationsum_fixedmass

using SpecialFunctions, Random

"""
    correlationsum_fixedmass(data, max_j; metric = Euclidean(), M = length(data)) → rs, ys
A fixed mass algorithm for the calculation of the fractal dimension according
to [^Grassberger 1988] with `max_j` the maximum number of neighbours that
should be considered for the calculation, `M` defines the number of points
considered for the calculation, default is the whole data set.

Implements
```math
D_q \\overline{\\log r^{(j)}} \\sim Ψ(j) - \\log N
```
where `` \\Psi(j) = \\frac{\\text{d} \\log Γ(j)}{\\text{d} j}
``, `rs` = ``\\overline{\\log r^{(j)}}`` and `ys` = ``\\Psi(j) - \\log N``.

``D_q`` can be computed by using `linear_region(rs, ys)`.

[^Grassberger 1988]: Peter Grassberger (1988) [Finite sample Corrections to Entropy and Dimension Estimates, Physics Letters A 128(6-7)](https://doi.org/10.1016/0375-9601(88)90193-4)
"""
function correlationsum_fixedmass(data, max_j; metric = Euclidean(), M = length(data), w = 0)
    N = length(data)
    M > N && throw(ArgumentError("The number of points used for the calculation `M` should not exceed the number of points provided."))
    # Transform the data into a tree.
    tree = searchstructure(KDTree, data, metric)
    # Calculate all nearest neighbours up to max_j.
    _, distances =
        bulksearch(
            tree,
            N == M ? data.data : data[view(randperm(N), 1:M)],
            NeighborNumber(max_j),
            Theiler(w),
        )
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
