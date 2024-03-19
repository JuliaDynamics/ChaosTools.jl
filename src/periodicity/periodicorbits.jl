using Combinatorics: permutations, multiset_permutations
using Random: randperm

export lambdamatrix, lambdaperms, periodicorbits

##########################################################################################
# Docstring and core function
##########################################################################################
"""
    periodicorbits(ds::DeterministicIteratedMap,
                   o, ics [, λs, indss, singss]; kwargs...) -> FP

Find fixed points `FP` of order `o` for the map `ds`
using the algorithm due to Schmelcher & Diakonos[^Schmelcher1997].
`ics` is a collection of initial conditions (container of vectors) to be evolved.

## Optional arguments

The optional arguments `λs, indss, singss` *must be containers* of appropriate
values, besides `λs` which can also be a number. The elements of those containers
are passed to: [`lambdamatrix(λ, inds, sings)`](@ref), which creates the appropriate
``\\mathbf{\\Lambda}_k`` matrix. If these arguments are not given,
a random permutation will be chosen for them, with `λ=0.001`.

## Keyword arguments

* `maxiters::Int = 100000`: Maximum amount of iterations an i.c. will be iterated
   before claiming it has not converged.
* `disttol = 1e-10`: Distance tolerance. If the 2-norm of a previous state with
   the next one is `≤ disttol` then it has converged to a fixed point.
* `inftol = 10.0`: If a state reaches `norm(state) ≥ inftol` it is assumed that
   it has escaped to infinity (and is thus abandoned).
* `abstol = 1e-8`: A detected fixed point isn't stored if it is in `abstol` 
   neighborhood of some previously detected point. Distance is measured by 
   euclidian norm. If you are getting duplicate fixed points, decrease this value.

## Description

The algorithm used can detect periodic orbits
by turning fixed points of the original
map `ds` to stable ones, through the transformation
```math
\\mathbf{x}_{n+1} = \\mathbf{x}_n +
\\mathbf{\\Lambda}_k\\left(f^{(o)}(\\mathbf{x}_n) - \\mathbf{x}_n\\right)
```
The index ``k`` counts the various
possible ``\\mathbf{\\Lambda}_k``.

## Performance notes

*All* initial conditions are
evolved for *all* ``\\mathbf{\\Lambda}_k`` which can very quickly lead to
long computation times.

[^Schmelcher1997]:
    P. Schmelcher & F. K. Diakonos, Phys. Rev. Lett. **78**, pp 4733 (1997)
"""
function periodicorbits(
        ds::DeterministicIteratedMap,
        o::Int,
        ics,
        λs,
        indss,
        singss;
        maxiters::Int = 100000,
        disttol::Real = 1e-10,
        inftol::Real = 10.0,
        roundtol = nothing,
        abstol::Real = 1e-8,
    )
    if !isnothing(roundtol)
        warn("`roundtol` keyword has been removed in favor of `abstol`")
    end

    type = typeof(current_state(ds))
    FP = fpcollection(type)
    for λ in λs, inds in indss, sings in singss
        Λ = lambdamatrix(λ, inds, sings)
        _periodicorbits!(FP, ds, o, ics, Λ, maxiters, disttol, inftol, abstol)
    end
    return Dataset(tovector(FP))
end

function periodicorbits(
        ds::DeterministicIteratedMap,
        o::Int,
        ics;
        maxiters::Int = 100000,
        disttol::Real = 1e-10,
        inftol::Real = 10.0,
        roundtol = nothing,
        abstol::Real = 1e-8,
    )
    if !isnothing(roundtol)
        warn("`roundtol` keyword has been removed in favor of `abstol`")
    end

    type = typeof(current_state(ds))
    FP = fpcollection(type)
    Λ = lambdamatrix(0.001, dimension(ds))
    _periodicorbits!(FP, ds, o, ics, Λ, maxiters, disttol, inftol, abstol)
    return Dataset(tovector(FP))
end

function _periodicorbits!(FP, ds, o, ics, Λ, maxiter, disttol, inftol, abstol)
    for st in ics
        reinit!(ds, st)
        prevst = st
        for _ in 1:maxiter
            prevst, st = Sk(ds, prevst, o, Λ)
            norm(st) > inftol && break

            if norm(prevst - st) < disttol
                storefp!(FP, st, abstol)
                break
            end
            prevst = st
        end
    end
end

function Sk(ds::DeterministicIteratedMap, prevst, o::Int, Λ)
    reinit!(ds, prevst)
    step!(ds, o)
    # TODO: For IIP systems optimizations can be done here to not re-allocate vectors...
    return prevst, prevst + Λ*(current_state(ds) .- prevst)
end