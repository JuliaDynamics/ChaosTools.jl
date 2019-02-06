export gali
using LinearAlgebra
#####################################################################################
#                               Continuous GALI                                     #
#####################################################################################
"""
    gali(ds::DynamicalSystem, tmax, k::Int | Q0; kwargs...) -> GALI_k, t
Compute ``\\text{GALI}_k`` [1] for a given `k` up to time `tmax`.
Return ``\\text{GALI}_k(t)`` and time vector ``t``.

The third argument, which sets the order of `gali`, can be an integer `k`, or
a matrix with its columns being the deviation vectors (then
`k = size(Q0)[2]`). In the first case random orthonormal vectors are chosen.

## Keyword Arguments
* `threshold = 1e-12` : If `GALI_k` falls below the `threshold`
  iteration is terminated.
* `dt = 1` : Time-step between deviation vector normalizations. For continuous
  systems this is approximate.
* `u0` : Initial state for the system. Defaults to `get_state(ds)`.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.

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
(see [`lyapunov`](@ref), [`lyapunovs`](@ref)).
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
the `threshold` until `tmax`
the given orbit is said to be chaotic, otherwise it is regular.

Our implementation is not based on the original paper, but rather in
the method described in [2], which uses the product of the singular values of ``A``,
a matrix that has as *columns* the deviation vectors.

## Performance Notes
This function uses a [`tangent_integrator`](@ref).
For loops over initial conditions and/or
parameter values one should use the low level method that accepts
an integrator, and `reinit!` it to new initial conditions.
See the "advanced documentation" for info on the integrator object.
The low level method is
```
ChaosTools.gali(tinteg, tmax, dt, threshold)
```

## References

[1] : Skokos, C. H. *et al.*, Physica D **231**, pp 30–54 (2007)

[2] : Skokos, C. H. *et al.*, *Chaos Detection and Predictability* - Chapter 5
(section 5.3.1 and ref. [85] therein), Lecture Notes in Physics **915**,
Springer (2016)
"""
gali(ds::DS, tmax::Real, k::Int; kwargs...) =
    gali(ds, tmax, orthonormal(dimension(ds), k); kwargs...)

function gali(ds::DS{IIP, S, D}, tmax::Real, Q0::AbstractMatrix;
    threshold = 1e-12, dt = 1, u0 = get_state(ds),
    diffeq...) where {IIP, S, D}

    size(Q0)[1] != D && throw(ArgumentError(
    "Deviation vectors must have first dimension equal to the dimension of the "*
    "system, $D."))
    2 ≤ size(Q0)[2] ≤ D || throw(ArgumentError(
    "The order of GALI_k must be 2 ≤ k ≤ $D."))

    # Create tangent integrator:
    if typeof(ds) <: DDS
        tinteg = tangent_integrator(ds, Q0; u0 = u0)
    else
        tinteg = tangent_integrator(ds, Q0; u0 = u0, diffeq...)
    end

    ST = stateeltype(ds)
    TT = typeof(ds.t0)
    gal::Vector{ST}, tvec::Vector{TT} = gali(tinteg, tmax, dt, threshold)
    return gal, tvec
end

function gali(tinteg, tmax, dt, threshold)

    rett = [tinteg.t]
    gali_k = [one(eltype(tinteg.u))]
    t0 = tinteg.t

    while tinteg.t < tmax + t0
        step!(tinteg, dt)
        # Normalize deviation vectors
        normalize_deviations!(tinteg)
        # Calculate singular values:
        zs = LinearAlgebra.svd(get_deviations(tinteg)).S
        push!(gali_k, prod(zs))
        push!(rett, tinteg.t)

        if gali_k[end] < threshold
            break
        end
    end
    return gali_k, rett
end

#####################################################################################
#                            Normalize Deviation Vectors                            #
#####################################################################################
# Metaprogramming ontributed by @saschatimme
function normalize_impl(::Type{SMatrix{D, K, T, DK}}) where {D, K, T, DK}
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
@generated function normalize_devs(A::SMatrix)
    normalize_impl(A)
end

# I AM SURE THE FOLLOWING CAN BE SHORTENED USING UNIONS!!!

# OOP version (either cont or disc)
"""
    normalize_deviations!(tinteg)
Normalize (in-place) the deviation vectors of the tangent integrator.
"""
function normalize_deviations!(
    tinteg::AbstractODEIntegrator{Alg, IIP, S}) where {Alg, IIP, S<:SMatrix}
    norms = normalize_devs(get_deviations(tinteg))
    set_deviations!(tinteg, norms)
    return
end
function normalize_deviations!(
    tinteg::TDI{false})
    norms = normalize_devs(get_deviations(tinteg))
    set_deviations!(tinteg, norms)
    return
end

# IIP
function normalize_inplace!(A)
    for i in 1:size(A)[2]
        LinearAlgebra.normalize!(view(A, :, i))
    end
end
function normalize_deviations!(tinteg::Union{
        AbstractODEIntegrator{Alg, IIP, S},
        MDI{Alg, S}}) where {Alg, IIP, S<:Matrix}
    A = get_deviations(tinteg)
    normalize_inplace!(A)
    # no reason to call set_deviations, because
    # get_deviations always returns views for IIP
    u_modified!(tinteg, true)
    return
end
function normalize_deviations!(tinteg::TDI{true})
    A = get_deviations(tinteg)
    normalize_inplace!(A)
    return
end
