export gali
using LinearAlgebra

"""
    gali(ds::DynamicalSystem, T, k::Int; kwargs...) -> GALI_k, t

Compute ``\\text{GALI}_k``[^Skokos2007] for a given `k` up to time `T`.
Return ``\\text{GALI}_k(t)`` and time vector ``t``.

The third argument sets the order of `gali`.
`gali` function simply initializes a [`TangentDynamicalSystem`](@ref) with `k`
deviation vectors and calls the method below.
This means that the automatic Jacobian is used by default.
Initialize manually a [`TangentDynamicalSystem`](@ref) if you have a hand-coded Jacobian.

## Keyword arguments
* `threshold = 1e-12`: If `GALI_k` falls below the `threshold`
  iteration is terminated.
* `Δt = 1`: Time-step between deviation vector normalizations. For continuous
  systems this is approximate.
* `u0`: Initial state for the system. Defaults to `current_state(ds)`.

## Description

The Generalized Alignment Index,
``\\text{GALI}_k``, is an efficient (and very fast) indicator of chaotic or regular
behavior type in ``D``-dimensional Hamiltonian systems
(``D`` is number of variables). The *asymptotic* behavior of
``\\text{GALI}_k(t)`` depends critically on
the type of orbit resulting
from the initial condition. If it is a chaotic orbit, then
```math
\\text{GALI}_k(t) \\sim
\\exp\\left[\\sum_{j=1}^k (\\lambda_1 - \\lambda_j)t \\right]
```
with ``\\lambda_j`` being the `j`-th Lyapunov exponent
(see [`lyapunov`](@ref), [`lyapunovspectrum`](@ref)).
If on the other hand the orbit is regular, corresponding
to movement in ``d``-dimensional torus with `` 1 \\le d \\le D/2``
then it holds
```math
\\text{GALI}_k(t) \\sim
    \\begin{cases}
      \\text{const.}, & \\text{if} \\;\\; 2 \\le k \\le d  \\; \\; \\text{and}
      \\; \\;d > 1 \\\\
      t^{-(k - d)}, & \\text{if} \\;\\;  d < k \\le D - d \\\\
      t^{-(2k - D)}, & \\text{if} \\;\\;  D - d < k \\le D
    \\end{cases}
```

Traditionally, if ``\\text{GALI}_k(t)`` does not become less than
the `threshold` until `T`
the given orbit is said to be chaotic, otherwise it is regular.

Our implementation is not based on the original paper, but rather in
the method described in[^Skokos2016b], which uses the product of the singular values of ``A``,
a matrix that has as *columns* the deviation vectors.

[^Skokos2007]: Skokos, C. H. *et al.*, Physica D **231**, pp 30–54 (2007)

[^Skokos2016b]:
    Skokos, C. H. *et al.*, *Chaos Detection and Predictability* - Chapter 5
    (section 5.3.1 and ref. [85] therein), Lecture Notes in Physics **915**, Springer (2016)
"""
function gali(ds::DynamicalSystem, T::Real, k::Int;
        u0 = current_state(ds), kwargs...
    )
    tands = TangentDynamicalSystem(ds; k, u0)
    return gali(tands, T; kwargs...)
end

"""
    gali(tands::TangentDynamicalSystem, T; threshold = 1e-12, Δt = 1)

The low-level method that is called by `gali(ds::DynamicalSystem, ...)`.
Use this method for looping over different initial conditions or parameters by
calling [`reinit!`](@ref) to `tands`.

The order of ``\\text{GALI}_k`` computed is the amount of deviation vectors in `tands`.

Also use this method if you have a hand-coded Jacobian to pass when creating `tands`.
"""
function gali(tands::TangentDynamicalSystem, T; threshold = 1e-12, Δt = 1)
    rett = [current_time(tands)]
    gali_k = [one(eltype(current_state(tands)))]
    t0 = current_time(tands)

    while current_time(tands) < T + t0
        step!(tands, Δt)
        normalize_deviations!(tands)
        zs = LinearAlgebra.svd(current_deviations(tands)).S
        push!(gali_k, prod(zs))
        push!(rett, current_time(tands))

        if gali_k[end] < threshold
            break
        end
    end
    return gali_k, rett
end

function normalize_deviations!(tands::TangentDynamicalSystem)
    Y = current_deviations(tands)
    if Y isa SMatrix # out of place
        Ynorm = normalize_static_deviations(Y)
        set_deviations!(tands, Ynorm)
    else # inplace
        normalize_inplace!(Y)
        set_deviations!(tands, Y)
    end
    return
end

# Metaprogramming ontributed by @saschatimme.
# It is a way to create a normalized version of each column of the SMatrix
@generated function normalize_static_deviations(A::SMatrix)
    normalize_metaprogramming(A)
end
function normalize_metaprogramming(::Type{SMatrix{D, K, T, DK}}) where {D, K, T, DK}
    exprs = []
    for j = 1:K
        c_j = Symbol("c", j)
        push!(exprs, :($c_j = normalize(A[:, $j])))
    end

    ops = Expr[]
    for j=1:K, i=1:D
        c_j = Symbol("c", j)
        push!(ops, :($c_j[$i]))
    end

    Expr(:block,
        exprs...,
        Expr(:call, SMatrix{D, K, T, D*K}, ops...)
        )
end

function normalize_inplace!(A)
    for i in 1:size(A)[2]
        LinearAlgebra.normalize!(view(A, :, i))
    end
end




# # I AM SURE THE FOLLOWING CAN BE SHORTENED USING UNIONS!!!

# # OOP version (either cont or disc)
# """
#     normalize_deviations!(tands)
# Normalize (in-place) the deviation vectors of the tangent integrator.
# """
# function normalize_deviations!(
#     tands::AbstractODEIntegrator{Alg, IIP, S}) where {Alg, IIP, S<:SMatrix}
#     norms = normalize_devs(current_deviations(tands))
#     set_deviations!(tands, norms)
#     return
# end
# function normalize_deviations!(
#     tands::TDI{false})
#     norms = normalize_devs(current_deviations(tands))
#     set_deviations!(tands, norms)
#     return
# end

# # IIP
# function normalize_inplace!(A)
#     for i in 1:size(A)[2]
#         LinearAlgebra.normalize!(view(A, :, i))
#     end
# end
# function normalize_deviations!(tands::Union{
#         AbstractODEIntegrator{Alg, IIP, S},
#         MDI{Alg, S}}) where {Alg, IIP, S<:Matrix}
#     A = current_deviations(tands)
#     normalize_inplace!(A)
#     # no reason to call set_deviations, because
#     # current_deviations always returns views for IIP
#     u_modified!(tands, true)
#     return
# end
# function normalize_deviations!(tands::TDI{true})
#     A = current_deviations(tands)
#     normalize_inplace!(A)
#     return
# end
