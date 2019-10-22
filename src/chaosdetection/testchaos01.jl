using Statistics

export testchaos01

"""
    testchaos01(φ::Vector [, cs, N0]) -> chaotic?
Perform the so called "0-1" test for chaos introduced by Gottwald and
Melbourne [1] on the timeseries `φ`.
Return `true` if `φ` is chaotic, `false` otherwise.

## Description
This method tests if the given timeseries is chaotic or not by transforming
it into a two-dimensional diffusive process. If the timeseries is chaotic,
the mean square displacement of the process grows as `sqrt(length(φ))`,
while it stays constant if the timeseries is regular.
The implementation here computes `K`, the correlation coefficient (median
of `Kc for c ∈ cs`), and simply checks if `K > 0.5`.

If you want to access the various `Kc` you should call the method
`testchaos01(φ, c::Real, N0)` which returns `Kc`.

`cs` defaults to `3π/5*rand(10) + π/4` and `N0`, the length
of the two-dimensional process, is `N0 = length(φ)/10`.

Notice that for data sampled from continous dynamical systems, some
care must be taken regarding the values of `cs`, see [1].

## References

[1] : Gottwald & Melbourne, “The 0-1 test for chaos: A review”
[Lect. Notes Phys., vol. 915, pp. 221–247, 2016.](www.doi.org/10.1007/978-3-662-48410-4_7)
"""
function testchaos01(φ::Vector, cs = 3π/5*rand(10) .+ π/4, N0 = Int(length(φ)÷10))
    K = median(testchaos01(φ, c, N0) for c in cs)
    return K > 0.5
end

function testchaos01(φ::Vector, c::Real, N0 = Int(length(φ)÷10))
    N, E = length(φ), mean(φ)
    @assert N0 ≤ N/10
    pc, qc = trigonometric_decomposition(φ, c)
    Dc = mmsd(E, pc, qc, N0, c)
    Kc = cor(Dc, 1:N0)
    return Kc
end

function trigonometric_decomposition(φ, c)
    X = promote_type(eltype(φ), typeof(c))
    N = length(φ)
    pc, qc = zeros(X, N), zeros(X, N)
    @inbounds for n in 1:N-1
        si, co = sincos(c*n)
        pc[n+1] = pc[n] + φ[n]*co
        qc[n+1] = qc[n] + φ[n]*si
    end
    return pc, qc
end

"modified mean square displacement"
function mmsd(mf, pc::Vector{T}, qc, N0, c) where T
    N, Dc = length(pc), zeros(T, N0)
    f = mf^2/(1-cos(c)) # constant factor for Dc
    for n in 1:N0
        Mc = sum((pc[n+j] - pc[j])^2 + (qc[n+j] - qc[j])^2 for j in 1:N-n)/(N-n)
        Dc[n] = Mc - f*(1 - cos(n*c))
    end
    return Dc
end
