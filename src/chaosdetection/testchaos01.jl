using Statistics

export testchaos01

"""
    testchaos01(x::Vector [, cs, N0]) -> chaotic?

Perform the so called "0-1" test for chaos introduced by Gottwald and
Melbourne[^Gottwald2016] on the timeseries `x`.
Return `true` if `x` is chaotic, `false` otherwise.

## Description

This method tests if the given timeseries is chaotic or not by transforming
it into a two-dimensional diffusive process like so:
```math
p_n = \\sum_{j=1}^{n}\\phi_j \\cos(j c),\\quad q_n = \\sum_{j=1}^{n}\\phi_j \\sin(j c)
```

If the timeseries is chaotic,
the mean square displacement of the process grows as `sqrt(length(x))`,
while it stays constant if the timeseries is regular.

The implementation here computes `K`, a coefficient measuring the growth of the mean square
displacement, and simply checks if `K > 0.5`.
`K` is the median of ``K_c`` over given `c`, see the reference.

If you want to access the various `Kc` you should call the method
`testchaos01(x, c::Real, N0)` which returns `Kc`. In fact, the high level method is
just `median(testchaos01(x, c, N0) for c in cs) > 0.5`.

`cs` defaults to `3π/5*rand(100) + π/4` and `N0`, the length
of the two-dimensional process, is `N0 = length(x)/10`.

For data sampled from continous dynamical systems, some
care must be taken regarding the values of `cs`.
Also note that this method performs rather poorly with even the slight amount
of noise, returning `true` for even small amounts of noise noisy timeseries.
Some possibilities to eliviate this exist, but are context specific on the application.
See [^Gottwald2016] for more info.

[^Gottwald2016]:
    Gottwald & Melbourne, “The 0-1 test for chaos: A review”
    [Lect. Notes Phys., vol. 915, pp. 221–247, 2016.](
    www.doi.org/10.1007/978-3-662-48410-4_7)
"""
function testchaos01(x::Vector, cs = 3π/5*rand(100) .+ π/4, N0 = Int(length(x)÷10))
    K = median(testchaos01(x, c, N0) for c in cs)
    return K > 0.5
end

function testchaos01(x::Vector, c::Real, N0 = Int(length(x)÷10))
    E = mean(x)
    @assert N0 ≤ length(x)/10
    pc, qc = trigonometric_decomposition(x, c)
    Dc = mmsd(E, pc, qc, N0, c)
    Kc = cor(Dc, 1:N0)
    return Kc
end

function trigonometric_decomposition(x, c)
    X = promote_type(eltype(x), typeof(c))
    N = length(x)
    pc, qc = zeros(X, N), zeros(X, N)
    @inbounds for n in 1:N-1
        si, co = sincos(c*n)
        pc[n+1] = pc[n] + x[n]*co
        qc[n+1] = qc[n] + x[n]*si
    end
    return pc, qc
end

"modified mean square displacement"
function mmsd(E, pc::Vector{T}, qc, N0, c) where T
    N, Dc = length(pc), zeros(T, N0)
    f = E^2/(1-cos(c)) # constant factor for Dc
    @inbounds for n in 1:N0
        Mc = sum((pc[n+j] - pc[j])^2 + (qc[n+j] - qc[j])^2 for j in 1:N-n)/(N-n)
        Dc[n] = Mc - f*(1 - cos(n*c))
    end
    return Dc
end
