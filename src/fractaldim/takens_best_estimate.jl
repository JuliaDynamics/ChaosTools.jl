export takens_best_estimate

"""
    takens_best_estimate(X, εmax, metric = Chebyshev(),εmin = 0) → D_C, D_C_95u, D_C_95l
Use the so-called "Takens' best estimate" [^Takens1985][^Theiler1988]
method for estimating the correlation dimension
`D_C` and the upper (`D_C_95u`) and lower (`D_C_95l`) confidence limit for the given dataset `X`.

The original formula is
```math
D_C \\approx \\frac{C(\\epsilon_\\text{max})}{\\int_0^{\\epsilon_\\text{max}}(C(\\epsilon) / \\epsilon) \\, d\\epsilon}
```
where ``C`` is the [`correlationsum`](@ref) and ``\\epsilon_\\text{max}`` is an upper cutoff.
Here we use the later expression
```math
D_C \\approx - \\frac{1}{\\eta},\\quad \\eta = \\frac{1}{(N-1)^*}\\sum_{[i, j]^*}\\log(||X_i - X_j|| / \\epsilon_\\text{max})
```
where the sum happens for all ``i, j`` so that ``i < j`` and ``||X_i - X_j|| < \\epsilon_\\text{max}``.
In the above expression, the bias in the original paper has already been corrected, as suggested in [^Borovkova1999].

The confidence limits are estimated from the log-likelihood function by finding
the values of `D_C` where the function has fallen by 2 from its maximum, see e.g.
[^Barlow] chapter 5.3
Because the CLT does not apply (no independent measurements), the limits are not
neccesarily symmetric.

According to [^Borovkova1999], introducing a lower cutoff `εmin` can make the
algorithm more stable (no divergence), this option is given but defaults to zero.

If `X` comes from a delay coordinates embedding of a timseries `x`, a recommended value
for ``\\epsilon_\\text{max}`` is `std(x)/4`.

[^Takens1985]: Takens, On the numerical determination of the dimension of an attractor, in: B.H.W. Braaksma, B.L.J.F. Takens (Eds.), Dynamical Systems and Bifurcations, in: Lecture Notes in Mathematics, Springer, Berlin, 1985, pp. 99–106.
[^Theiler1988]: Theiler, [Lacunarity in a best estimator of fractal dimension. Physics Letters A, 133(4–5)](https://doi.org/10.1016/0375-9601(88)91016-X)
[^Borovkova1999]: Borovkova et al., [Consistency of the Takens estimator for the correlation dimension. The Annals of Applied Probability, 9, 05 1999.](https://doi.org/10.1214/aoap/1029962747)
[^Barlow]: Barlow, R., Statistics - A Guide to the Use of Statistical Methods in the Physical Sciences. Vol 29. John Wiley & Sons, 1993
"""
function takens_best_estimate(X, εmax, metric = Chebyshev(); εmin=0)
    n, η, N = 0, zero(eltype(X)), length(X)
    @inbounds for i in 1:N
        for j in i+1:N
            d = evaluate(metric, X[i], X[j])
            if εmin < d < εmax
                n += 1
                η += log(d/εmax)
            end
        end
    end
    # bias-corrected version (log-likelihood function shifted on x-axis)
    α = -(n-1)/η
    # biased version (maximum of original log-likelihood function)
    α_b = -n/η
    # value of maximum of original log-likelihood function
    mxl = n*log(α_b) + α_b * η
    # at the 95%-confidence interval, the log-l function has dropped by 2
    # -> log_l(x) - mxl + 2 = 0
    # this is a result of the invariance of the MLE, a really nice property
    # these limits are not going to be perfectly symmetric (CLT does not apply)
    mn, mx = fzeros(x-> n * log(x) + η * x - mxl +2 , 0,2*α)

    # Since the bias-correction is just a shift of the log-l function on the
    # x-axis, we can easily shift the confidence limits by the bias α-α_b
    α95u = α - α_b + mn
    α95l = α - α_b + mx

    return α, α95u, α95l
end
