using NearestNeighbors
using Distances: chebyshev
using SpecialFunctions: digamma
using LsqFit: curve_fit
using StatsBase: autocor

export estimate_delay
export estimate_dimension
export mutinfo, mutinfo_delaycurve
export estimate_dimension, stochastic_indicator
# export mutual_info

#####################################################################################
#                                Mutual Information                                 #
#####################################################################################
"""
    mutinfo(k, X1, X2[, ..., Xm]) -> I

Calculate the mutual information `I` of the given vectors
`X1, ....`, using `k` nearest-neighbors.

The method follows
the second algorithm outlined by Kraskov [1^].

## References
[^1] : A. Kraskov *et al.*, [Phys. Rev. E **69**, pp 066138 (2004)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)

See also [`estimate_delay`](@ref).
"""
function mutinfo(k, Xm::Vararg{<:AbstractVector,M}) where M
    @assert M > 1
    @assert (size.(Xm,1) .== size(Xm[1],1)) |> prod
    k += 1
    N = size(Xm[1],1)
    invN = 1/N

    d = Dataset(Xm...)
    tree = KDTree(d.data, Chebyshev())

    n_x_m = zeros(M)

    Xm_sp = zeros(Int, N, M)
    Xm_revsp = zeros(Int, N, M)
    for m in 1:M
        Xm_sp[:,m] .= sortperm(Xm[m]; alg=QuickSort)
        Xm_revsp[:,m] .= sortperm(Xm_sp[:,m]; alg=QuickSort)
    end

    I = digamma(k) - (M-1)/k + (M-1)*digamma(N)

    nns = (x = knn(tree, d.data, k)[1]; [ind[1] for ind in x])

    I_itr = zeros(M)
    # Makes more sense computationally to loop over N rather than M
    for i in 1:N
        ϵ = abs.(d[nns[i]] - d[i])./2

        for m in 1:M # this loop takes 8% of time
            hb = lb = Xm_revsp[i,m]
            while abs(Xm[m][Xm_sp[hb,m]] - Xm[m][i]) <= ϵ[m] && hb < N
                hb += 1
            end
            while abs(Xm[m][Xm_sp[lb,m]] - Xm[m][i]) <= ϵ[m] && lb > 1
                lb -= 1
            end
            n_x_m[m] = hb - lb
        end

        I_itr .+= digamma.(n_x_m)
    end

    I_itr .*= invN

    I -= sum(I_itr)

    return max(0, I)
end

"""
    mutinfo_delaycurve(x; maxtau=100, k=1)

Return the [`mutinfo`](@ref) between `x` and itself for delays of `1:maxtau`.
"""
function mutinfo_delaycurve(X::AbstractVector; maxtau=100, k=1)
    I = zeros(maxtau)

    @views for τ in 1:maxtau
        I[τ] = mutinfo(k, X[1:end-τ],X[τ+1:end])
    end

    return I
end


#####################################################################################
#                               Estimate Delay Times                                #
#####################################################################################
"""
    localextrema(y) -> max_ind, min_ind
Find the local extrema of given array `y`, by scanning point-by-point. Return the
indices of the maxima (`max_ind`) and the indices of the minima (`min_ind`).
"""
function localextrema end
@inbounds function localextrema(y)
    l = length(y)
    i = 1
    maxargs = Int[]
    minargs = Int[]
    if y[1] > y[2]
        push!(maxargs, 1)
    elseif y[1] < y[2]
        push!(minargs, 1)
    end

    for i in 2:l-1
        left = i-1
        right = i+1
        if  y[left] < y[i] > y[right]
            push!(maxargs, i)
        elseif y[left] > y[i] < y[right]
            push!(minargs, i)
        end
    end

    if y[l] > y[l-1]
        push!(maxargs, l)
    elseif y[l] < y[l-1]
        push!(minargs, l)
    end
    return maxargs, minargs
end


function exponential_decay_extrema(c::AbstractVector)
    ac = abs.(c)
    ma, mi = localextrema(ac)
    # ma start from 1 but correlation is expected to start from x=0
    ydat = ac[ma]; xdat = ma .- 1
    # Do curve fit from LsqFit
    model(x, p) = @. exp(-x/p[1])
    decay = curve_fit(model, xdat, ydat, [1.0]).param[1]
    return decay
end

function exponential_decay(c::AbstractVector)
    # Do curve fit from LsqFit
    model(x, p) = @. exp(-x/p[1])
    decay = curve_fit(model, 0:length(c)-1, abs.(c), [1.0]).param[1]
    return decay
end


"""
    estimate_delay(s, method::String) -> τ

Estimate an optimal delay to be used in [`Reconstruction`](@ref). Returns the exponential
decay time `τ` rounded to an integer.

The `method` can be one of the following:

* `first_zero` : find first delay at which the auto-correlation function becomes 0.
* `first_min` : return delay of first minimum of the auto-correlation function.
* `exp_decay` : perform an exponential fit to the `abs.(c)` with `c` the auto-correlation function of `s`.
* `mutual_inf` : return the first minimum of the mutual information function (see [`mutinfo`](@ref). 
  this option also has the following keyword arguments:
    * `maxtau::Integer=100` : stops the delay calculations after the given `maxtau`. This may
      not be appropriate for all data (ie the optimal delay may be higher than the default `maxtau`)
    * `k::Integer=1` : the number of nearest-neighbors to include. As with `maxtau` the default
      value of 1 may not be appropriate for all data
"""
function estimate_delay(x::AbstractVector, method::String; maxtau=100, k=1)
    method ∈ ["first_zero", "first_min", "exp_decay", "mutual_inf"] ||
        throw(ArgumentError("Unknown method"))

    if method=="first_zero"
        c = autocor(x, 0:length(x)÷10; demean=true)
        i = 1
        # Find 0 crossing:
        while c[i] > 0
            i+= 1
            i == length(c) && break
        end
        return i

    elseif method=="first_min"
        c = autocor(x, 0:length(x)÷10, demean=true)
        i = 1
        # Find min crossing:
        while  c[i+1] < c[i]
            i+= 1
            i == length(c)-1 && break
        end
        return i
    elseif method=="exp_decay"
        c = autocor(x, 0:length(x)÷10, demean=true)
        # Find exponential fit:
        τ = exponential_decay(c)
        return round(Int,τ)
    elseif method=="mutual_inf"
        m = mutinfo(k, x,x)
        L = length(x)
        for i=1:maxtau
            n = mutinfo(k, view(x, 1:L-i), view(x, 1+i:L))
            n > m && return i
            m = n
        end
    end
end





#####################################################################################
#                                Estimate Dimension                                 #
#####################################################################################
function _average_a(s::AbstractVector{T},D,τ) where T
    #Sum over all a(i,d) of the Ddim Reconstructed space, equation (2)
    R1 = Reconstruction(s,D+1,τ)
    R2 = Reconstruction(s[1:end-τ],D,τ)
    tree2 = KDTree(R2)
    nind = (x = knn(tree2, R2.data, 2)[1]; [ind[1] for ind in x])
    e=0.
    for (i,j) in enumerate(nind)
        δ = norm(R2[i]-R2[j], Inf)
        #If R2[i] and R2[j] are still identical, choose the next nearest neighbor
        if δ == 0.
            j = knn(tree2, R2[i], 3, true)[1][end]
            δ = norm(R2[i]-R2[j], Inf)
        end
        e += norm(R1[i]-R1[j], Inf) / δ
    end
    return e / length(R1)
end

function dimension_indicator(s,D,τ) #this is E1, equation (3) of Cao
    return _average_a(s,D+1,τ)/_average_a(s,D,τ)
end


"""
    estimate_dimension(s::AbstractVector, τ:Int, Ds = 1:6) -> E1s

Compute a quantity that can estimate an optimal embedding
dimension to be used in [`Reconstruction`](@ref).

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E1` for each `D ∈ Ds`, according to Cao's Method (eq. 3 of [1]).

Return the vector of all computed `E1`s. To estimate a dimension from this,
find the dimension for which the value `E1` saturates, at some value around 1.

*Note: This method does not work for datasets with perfectly periodic signals.*

## References

[1] : Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)
"""
function estimate_dimension(s::AbstractVector{T}, τ::Int, Ds = 1:6) where {T}
    E1s = zeros(T, length(Ds))
    aafter = zero(T)
    aprev = _average_a(s, Ds[1], τ)
    for (i, D) ∈ enumerate(Ds)
        aafter = _average_a(s, D+1, τ)
        E1s[i] = aafter/aprev
        aprev = aafter
    end
    return E1s
end
# then use function `saturation_point(Ds, E1s)` from ChaosTools



"""
    stochastic_indicator(s::AbstractVector, τ:Int, Ds = 1:6) -> E2s

Compute an estimator for apparent randomness in a reconstruction of dimensions `Ds`.

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E2` for each `D ∈ Ds`, according to Cao's Method (eq. 5 of [1]).

Use this function to confirm that the
input signal is not random and validate the results of [`estimate_dimension`](@ref).
In the case of random signals, it should be `E2 ≈ 1 ∀ D`.

## References

[1] : Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)
"""
function stochastic_indicator(s::AbstractVector{T},τ, Ds=1:6) where T # E2, equation (5)
    #This function tries to tell the difference between deterministic
    #and stochastic signals
    #Calculate E* for Dimension D+1
    E2s = Float64[]
    for D ∈ Ds
        R1 = Reconstruction(s,D+1,τ)
        tree1 = KDTree(R1[1:end-1-τ])
        method = FixedMassNeighborhood(2)

        Es1 = 0.
        nind = (x = neighborhood(R1[1:end-τ], tree1, method); [ind[1] for ind in x])
        for  (i,j) ∈ enumerate(nind)
            Es1 += abs(R1[i+τ][end] - R1[j+τ][end]) / length(R1)
        end

        #Calculate E* for Dimension D
        R2 = Reconstruction(s,D,τ)
        tree2 = KDTree(R2[1:end-1-τ])
        Es2 = 0.
        nind = (x = neighborhood(R2[1:end-τ], tree2, method); [ind[1] for ind in x])
        for  (i,j) ∈ enumerate(nind)
            Es2 += abs(R2[i+τ][end] - R2[j+τ][end]) / length(R2)
        end
        push!(E2s, Es1/Es2)
    end
    return E2s
end
