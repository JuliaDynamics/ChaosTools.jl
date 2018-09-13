using StaticArrays
using Combinatorics: permutations, multiset_permutations
using Random: randperm

export lambdamatrix, lambdaperms, periodicorbits


"""
    lambdamatrix(λ, inds::Vector{Int}, sings) -> Λk
Return the matrix ``\\mathbf{\\Lambda}_k`` used to create a new
dynamical system with some unstable fixed points turned to stable in the function
[`periodicorbits`](@ref).

## Arguments

1. `λ<:Real` : the multiplier of the ``C_k`` matrix, with `0<λ<1`.
2. `inds::Vector{Int}` :
   The `i`th entry of this vector gives the *row* of the nonzero element of the `i`th
   column of ``C_k``.
3. `sings::Vector{<:Real}` : The element of the `i`th column of ``C_k`` is +1
   if `signs[i] > 0` and -1 otherwise (`sings` can also be `Bool` vector).

Calling `lambdamatrix(λ, D::Int)`
creates a random ``\\mathbf{\\Lambda}_k`` by randomly generating
an `inds` and a `signs` from all possible combinations. The *collections*
of all these combinations can be obtained from the function [`lambdaperms`](@ref).

## Description
Each element
of `inds` *must be unique* such that the resulting matrix is orthogonal
and represents the group of special reflections and permutations.

Deciding the appropriate values for `λ, inds, sings` is not trivial. However, in
ref. [2] there is a lot of information that can help with that decision. Also,
by appropriately choosing various values for `λ`, one can sort periodic
orbits from e.g. least unstable to most unstable, see [3] for details.

## References

[2] : D. Pingel *et al.*, Phys. Rev. E **62**, pp 2119 (2000)

[3] : F. K. Diakonos *et al.*, Phys. Rev. Lett. **81**, pp 4349 (1998)
"""
function lambdamatrix(λ::Real, inds::AbstractVector{<:Integer},
    sings::AbstractVector{<:Real})
    # this function seems to be super inefficient

    D = length(inds)
    D != length(sings) && throw(ArgumentError("inds and sings must have equal size."))
    0 < λ < 1 || throw(ArgumentError("λ must be in (0,1)"))
    unique(inds)!=inds && throw(ArgumentError("All elements of inds must be unique."))
    # This has to be improved to not create intermediate arrays!!!
    a = zeros(typeof(λ), (D,D))
    for i in 1:D
        a[(i-1)*D + inds[i]] = λ*(sings[i] > 0 ? +1 : -1)
    end
    return SMatrix{D,D}(a)
end

function lambdamatrix(λ::T, D::Integer) where {T<:Real}
    positions = randperm(D)
    signs = rand(Bool, D)
    lambdamatrix(λ, positions, signs)
end

"""
    lambdaperms(D) -> indperms, singperms
Return two collections that each contain all possible combinations of indices (total of
``D!``) and signs (total of ``2^D``) for dimension `D` (see [`lambdamatrix`](@ref)).
"""
function lambdaperms(D::Integer)
    indperms = collect(permutations([1:D;], D))
    p = trues(D)
    singperms = [p[:]] #need p[:] because p is mutated afterwards
    for i = 1:D
        p[i] = false; append!(singperms, multiset_permutations(p, D))
    end
    return indperms, singperms
end


"""
    periodicorbits(ds::DiscreteDynamicalSystem,
                   o, ics [, λs, indss, singss]; kwargs...) -> FP
Find fixed points `FP` of order `o` for the map `ds`
using the algorithm due to Schmelcher & Diakonos [1].
`ics` is a collection of initial conditions (container of vectors) to be evolved.

## Optional Arguments
The optional arguments `λs, indss, singss` *must be containers* of appropriate
values, besides `λs` which can also be a number. The elements of those containers
are passed to: [`lambdamatrix(λ, inds, sings)`](@ref), which creates the appropriate
``\\mathbf{\\Lambda}_k`` matrix. If these arguments are not given,
a random permutation will be chosen for them, with `λ=0.001`.

## Keyword Arguments
* `maxiters::Int = 100000` : Maximum amount of iterations an i.c. will be iterated
   before claiming it has not converged.
* `disttol = 1e-10` : Distance tolerance. If the 2-norm of a previous state with
   the next one is `≤ disttol` then it has converged to a fixed point.
* `inftol = 10.0` : If a state reaches `norm(state) ≥ inftol` it is assumed that
   it has escaped to infinity (and is thus abandoned).
* `roundtol::Int = 4` : The found fixed points are rounded
   to `roundtol` digits before pushed into the list of returned fixed points `FP`,
   *if* they are not already contained in `FP`.
   This is done so that `FP` doesn't contain duplicate fixed points (notice
   that this has nothing to do with `disttol`). Turn this to `typemax(Int)`
   to get the full precision of the algorithm.

## Description
The algorithm used can detect periodic orbits
by turning fixed points of the original
map `ds` to stable ones, through the transformation
```math
\\mathbf{x}_{n+1} = \\mathbf{x}_n +
\\mathbf{\\Lambda}_k\\left(f^{(o)}(\\mathbf{x}_n) - \\mathbf{x}_n\\right)
```
with ``f`` = `eom`. The index ``k`` counts the various
possible ``\\mathbf{\\Lambda}_k``.

## Performance Notes
*All* initial conditions are
evolved for *all* ``\\mathbf{\\Lambda}_k`` which can very quickly lead to
long computation times.

## References

[1] : P. Schmelcher & F. K. Diakonos, Phys. Rev. Lett. **78**, pp 4733 (1997)
"""
function periodicorbits(ds::DDS{false, S},
                        o::Int,
                        ics,
                        λs,
                        indss,
                        singss;
                        maxiters::Int = 100000,
                        disttol::Real = 1e-10,
                        inftol::Real = 10.0,
                        roundtol::Int = 4) where {S}

    FP = S[]
    integ = integrator(ds)
    for λ in λs
        for inds in indss
            for sings in singss
                Λ = lambdamatrix(λ, inds, sings)
                _periodicorbits!(FP, integ, o, ics, Λ, maxiters, disttol, inftol, roundtol)
            end
        end
    end
    return Dataset(FP)
end
function periodicorbits(ds::DDS{false, S},
                        o::Int,
                        ics;
                        maxiters::Int = 100000,
                        disttol::Real = 1e-10,
                        inftol::Real = 10.0,
                        roundtol::Int = 4) where {S}

    integ = integrator(ds)
    FP = S[]
    Λ = lambdamatrix(0.001, dimension(ds))
    _periodicorbits!(FP, integ, o, ics, Λ, maxiters, disttol, inftol, roundtol)
    return Dataset(FP)
end
periodicorbits(::DDS{true}, args...; kwargs...) = error(
"Currently `periodicorbits` is implemented only for out-of-place maps. Sorry!")

function _periodicorbits!(
    FP, integ, o, ics, Λ, maxiter, disttol, inftol, roundtol)

    for st in ics
        reinit!(integ, st)
        prevst = st
        for i in 1:maxiter
            prevst, st = Sk(integ, prevst, o, Λ)
            norm(st) > inftol && break

            if norm(prevst - st) < disttol
                unist = round.(st, digits = roundtol)
                unist ∉ FP && push!(FP, unist)
                break
            end
            prevst = st
        end
    end
end

function Sk(integ::MDI{false}, prevst, o, Λ)
    integ.u = prevst
    step!(integ, o)
    return prevst, prevst + Λ*(integ.u - prevst)
end
