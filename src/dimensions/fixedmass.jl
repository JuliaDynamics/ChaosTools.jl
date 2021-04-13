using SpecialFunctions

"""
    fixed_mass_algorithm(data, max_j; metric = Euclidean(), M = length(data)) → rs, ϵs
A fixed mass algorithm for the calculation of the fractal dimension according
to [^Grassberger 1988] with `max_j` the maximum number of neighbours that
should be considered for the calculation, `M` defines the number of points
considered for the calculation, default is the whole data set.

Implements
```math
D_q \\overline{\\log r^{(j)}} \\sim Ψ(j) - \\log N
```
where `` Ψ(j) = \\frac{\\text{d} \\log Γ(j)}{\\text{d} j}
``, `rs` = ``\\overline{\\log r^{(j)}}`` and `ϵs` = ``Ψ(j) - \\log N``.

``D_q`` can be computed by using `linear_region(rs, ϵs)`.

[^Grassberger 1988]: Peter Grassberger (1988) [Finite sample Corrections to Entropy and Dimension Estimates, Physics Letters A 128(6-7)](https://doi.org/10.1016/0375-9601(88)90193-4)
"""
function fixed_mass_algorithm(data, max_j; metric = Euclidean(), M = length(data))
    # Define the length of the data set.
    N = length(data)
    # Transform the data into a tree.
    tree = KDTree(data, metric)
    # Calculate all nearest neighbours up to max_j.
    _, distances =
        knn(
            tree,
            # If M is given, only M random points of the set are considered.
            M == N ?
                [point for point in data] :
                [data[index] for index in randperm(N)[1:M]],
            max_j,
            true,
        )
    # The epsilons define the left side of the equation
    ϵs = [digamma(j) - log(N) for j in 2:max_j]
    # Holds the mean value of the logarithms of the distances.
    rs = zeros(max_j-1)
    for j in 2:max_j
        memory = 0.0
        for radii in distances
            memory += log(radii[j])
        end
        # After adding up all logarithms of the distances divide them by their number.
        rs[j-1] = memory / M
    end
    return rs, ϵs
end
