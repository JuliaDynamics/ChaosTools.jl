##########################################################################################
# Lambda matrix stuff
##########################################################################################
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
ref.[^Pingel2000] there is a lot of information that can help with that decision. Also,
by appropriately choosing various values for `λ`, one can sort periodic
orbits from e.g. least unstable to most unstable, see[^Diakonos1998] for details.

[^Pingel2000]: D. Pingel *et al.*, Phys. Rev. E **62**, pp 2119 (2000)

[^Diakonos1998]: F. K. Diakonos *et al.*, Phys. Rev. Lett. **81**, pp 4349 (1998)
"""
function lambdamatrix(λ::Real, inds::AbstractVector{<:Integer},
    sings::AbstractVector{<:Real})
    0 < λ < 1 || throw(ArgumentError("λ must be in (0,1)"))
    return cmatrix(λ, inds, sings)
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
# TO-DO
"""
function cmatrix(constant::Real, inds::AbstractVector{<:Integer},
    sings::AbstractVector{<:Real})
    # this function seems to be super inefficient
    D = length(inds)
    D != length(sings) && throw(ArgumentError("inds and sings must have equal size."))
    unique(inds)!=inds && throw(ArgumentError("All elements of inds must be unique."))
    # This has to be improved to not create intermediate arrays!!!
    a = zeros(typeof(constant), (D,D))
    for i in 1:D
        a[(i-1)*D + inds[i]] = constant*(sings[i] > 0 ? +1 : -1)
    end
    return SMatrix{D,D}(a)
end

"""
TO-DO
"""
cmatrix(inds::AbstractVector{<:Integer}, signs::AbstractVector{<:Real}) = cmatrix(1.0, inds, signs)
