#=
This file attempts to implement the "imporoved dimension estimation" by Judd:

Judd, K. (1992). An improved estimator of dimension and some comments on providing
confidence intervals. Physica D: Nonlinear Phenomena, 56(2–3), 216–228. https://doi.org/10.1016/0167-2789(92)90025-I

Judd, K. (1994). Estimating dimension from small samples. Physica D: Nonlinear Phenomena,
71(4), 421–429. https://doi.org/10.1016/0167-2789(94)90008-6

So far this attempt has failed.
=#
using Distances

function interpoint_distances(X, norm = Euclidean())
    ds = eltype(X)[]
    # changed N to be <10000, because otherwise I get an OutOfMemoryError()
    # because for long time series, N(N-1)/2 will get really large.
    # Problem arising from that: bins are not filled properly with default binning
    # for long time series/ systems with "large" maximum distances (Lorenz)
    N = minimum([length(X), 10000])
    # choosing N random points from X.
    set = X[rand(1:length(X),N)]
    sizehint!(ds, Int(N*(N-1)/2))
    @inbounds for i in 1:N
        @inbounds for j in i+1:N
            push!(ds, evaluate(norm, set[i], set[j]))
        end
    end
    return filter(x-> x!= 0.0,ds)
end

"""
    logspace_histogram(x::Vector, λ = exp(-w0), ϵ₀ = λ^2 * maximum(x)) -> bins, counts
Calculate a logarithmically spaced histogram of the values in `x` (must be positive
definite, i.e. distances) and return the bins and count of values in the bins.

The `i`-th bin is `[ϵ₀*λ^i, ϵ₀*λ^(i+1))`. Elements of `x` greater than `ϵ₀` are disregarded.
"""
function logspace_histogram(x::AbstractVector, λ = exp(-w0(x)), ϵ₀ = λ^2 * maximum(x))
    @assert λ < 1
    bins = [ϵ₀]
    λmin = λ*minimum(x)
    λmin == 0 && error("x cannot contain zero values!")
    for i in 1:length(x)
        push!(bins, bins[end]*λ)
        bins[end] < λmin && break
    end
    reverse!(bins)
    L = length(bins)
    counts = zeros(Int, L)
    # Fill bins with values
    for v in x
        i = searchsortedfirst(bins, v)
        i == L + 1 && continue
        counts[i] += 1
    end
    return reverse!(bins), reverse!(counts)
end
# TODO: test the algorithm with including the bin [∞, ε0)
w0(x) = log(maximum(x)/minimum(x))/√length(x)

# The two papers are really complicated. The way to really do this that I will
# follow is given by the 1992 Judd paper, page 223, second column.
# p is a function ε^d * polynomial of degree `t`, with `t` " = 2 being a good choice."
# Then the formula for L (no equation number, because why write good papers) is
# "minimized" by some routine.
# Notice that pᵢ is just the function of the ϵᵢ , the bin edges.

# To estimate the dimension we have to do an optimization solve of a problem...? wtf!
using JuMP
using Ipopt
# %% TODO: Add option to choose degree of polynomial (seems not really feasible
# for the estimation with JuMP, check if there's another way)
# I found a Paper (doi.org/10.1063/1.166489) where it is stated that
# "degree 1 is usually sufficient".
# Personally, I find this idea of choosing the degree sort of "by feeling" a little
# weird (and, tbh, unscientific).

function judd_estimator(bins, count; guess = 1e-9)
    # initiate model with nonlinear optimizer
    model = Model(Ipopt.Optimizer)

    # initiate variables. If I don't restrict them to be larger than zero, nothing works
    @variable(model,d >= guess)
    @variable(model,a0  >= 1e-9)
    @variable(model,a1  >= 1e-9)
    @variable(model,a2  >= 1e-9)

    # Function to minimize = minus log-likelihood function
    # JuMP doesn't like pre-defined functions, so you have to write the function in here explicitly
    @NLobjective(model,Min, -sum(count[i] * log( bins[i]^d * (a0 + a1*bins[i] + a2*bins[i]^2)) for i in 1:length(bins)))

    # constraints. If I go for equality in the second one (sum(pᵢ == 1)), nothing
    # converges. Since we're not actually considering the last bin, this should be
    # fine I guess.
    # Also, the constraints don't allow for >/ <, so that's why I'm using >= ep 
    ep = eps(Float64)
    @NLconstraint(model, con[i = 1:length(bins)], bins[i]^d*( a0 + a1*bins[i] + a2*bins[i]^2) >= ep)
    @NLconstraint(model, sum( bins[i]^d *( a0 + a1*bins[i] + a2*bins[i]^2) for i in 1:length(bins)) <= 1.)

    # Alternative definition of the objective and constraints according to the
    # 1994 paper (in the 1992 one he just argues that you can approximate the
    # difference by the larger bin). This doesn't actually make anything better
    # and only increases computation time.
    # For this to work, we would also have to pass ϵ₀ as an argument.
    # @NLobjective(model,Min, -sum(counts[i] * log( (bins[i]/ϵ₀)^d * (a0 + a1*bins[i]/ϵ₀ + a2*(bins[i]/ϵ₀)^2) - (bins[i+1]/ϵ₀)^d * (a0 + a1*bins[i+1]/ϵ₀ + a2*(bins[i+1]/ϵ₀)^2)) for i in 2:length(bins)-1))
    #
    # @NLconstraint(model, con[i = 2:length(bins)-1],(bins[i]/ϵ₀)^d * (a0 + a1*bins[i]/ϵ₀ + a2*(bins[i]/ϵ₀)^2) - (bins[i+1]/ϵ₀)^d * (a0 + a1*bins[i+1]/ϵ₀ + a2*(bins[i+1]/ϵ₀)^2) >= 1e-9)

    optimize!(model)

    # optimal value of variable. This is currently... not very realistic.
    # Also, it "works" more or less for Henon, towel etc. (fast convergence, unrealistic result),
    # while it only converges incredibly slowly or not at all for Lorenz with
    # default binning (because bins are not filled properly, at least not with the
    # amount of points that my poor little 8GB RAM machine can process).
    # It "works" (= converges) for Lorenz by setting λ manually to a smaller value
    value(d)

end
