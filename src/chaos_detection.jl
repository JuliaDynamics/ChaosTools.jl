export gali
using OrdinaryDiffEq
#####################################################################################
#                               Continuous GALI                                     #
#####################################################################################
"""
    gali(ds::DynamicalSystem, tmax, k::Int; kwargs...) -> GALI_k, t
Compute ``\\text{GALI}_k`` [1] for a given `k` up to time `tmax`.
Return ``\\text{GALI}_k(t)`` and time vector ``t``.

## Keyword Arguments
* `threshold = 1e-12` : If `GALI_k` falls below the `threshold` iteration is terminated.
* `dt = 1` : Time-step between variational vector normalizations. Must be integer
  for discrete systems.
* `diff_eq_kwargs` : See [`trajectory`](@ref).
* `u0` : Initial state for the system. Defaults to `state(ds)`.
* `w0` : Initial orthonormal vectors (in matrix form).
  Defaults to `orthonormal(dimension(ds), k)`, i.e. `k` random orthonormal vectors.

## Description
The Generalized Alignment Index,
``\\text{GALI}_k``, is an efficient (and very fast) indicator of chaotic or regular
behavior type in ``D``-dimensional Hamiltonian systems
(``D`` is number of variables). The *asymptotic* behavior of
``\\text{GALI}_k(t)`` depends critically of
the type of orbit resulting
from the initial condition `state(ds)`. If it is a chaotic orbit, then
```math
\\text{GALI}_k(t) \\sim
\\exp\\left[\\sum_{j=1}^k (\\lambda_1 - \\lambda_j)t \\right]
```
with ``\\lambda_1`` being the maximum [`lyapunov`](@ref) exponent.
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
This function uses a [`tangent_integrator`](@ref). For loops over initial conditions and/or
parameter values one should use the lower level methods that accept
an integrator, and `reinit!` it to new initial conditions.

See the "advanced documentation" for info on the integrator object
and use `@which ...` to go to the source code for the low-level
call signature.

## References

[1] : Skokos, C. H. *et al.*, Physica D **231**, pp 30–54 (2007)

[2] : Skokos, C. H. *et al.*, *Chaos Detection and Predictability* - Chapter 5
(section 5.3.1 and ref. [85] therein), Lecture Notes in Physics **915**,
Springer (2016)
"""
function gali(ds::DS{IIP, S, D}, k::Int, tmax::Real;
    w0 = orthonormal(dimension(ds), k),
    threshold = 1e-12, dt = 1, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    u0 = state(ds)) where {IIP, S, D}

    # Create tangent integrator:
    if typeof(ds) <: DDS
        tinteg = tangent_integrator(ds, w0; u0 = u0)
    else
        tinteg = tangent_integrator(ds, w0; diff_eq_kwargs = diff_eq_kwargs, u0 = u0)
    end
    k = size(w0)[2]
    @assert k > 1

    return _gali(tinteg, tmax, dt, threshold)
end

function _gali(tinteg, tmax, dt, threshold)

    rett = [tinteg.t]
    gali_k = [one(eltype(tinteg.u))]
    k = size(tinteg.u)[2] - 1
    ws_index = SVector{k, Int}(2:(k+1)...)
    t0 = tinteg.t

    while tinteg.t < tmax + t0
        step!(tinteg, dt)
        # Normalize deviation vectors
        normalize_deviations!(tinteg, ws_index)
        # Calculate singular values:
        zs = singular_values(tinteg.u, ws_index)
        push!(gali_k, prod(zs))
        push!(rett, tinteg.t)

        if gali_k[end] < threshold
            break
        end
    end
    return gali_k, rett
end

#####################################################################################
#                 Helpers (normalize and singular values)                           #
#####################################################################################
# Super convienient dispatch that allows a single function
using DynamicalSystemsBase: MDI
# Contributed by @saschatimme
function normalize_impl(::Type{SMatrix{D, K, T, DK}}) where {D, K, T, DK}
    exprs = []
    for j = 2:K
        c_j = Symbol("c", j)
        push!(exprs, :($c_j = normalize(A[:, $j])))
    end

    ops = Expr[]
    for j=2:K, i=1:D
        c_j = Symbol("c", j)
        push!(ops, :($c_j[$i]))
    end

    Expr(:block,
        exprs...,
        Expr(:call, SMatrix{D, K-1, T, D*(K-1)}, ops...)
        )
end
@generated function normalize_devs(A::SMatrix)
    normalize_impl(A)
end
# ws_index is just the SVector(2:(k+1)...) which is also of type SVector{k}
# OOP Versions:
function normalize_deviations!(tinteg::ODEIntegrator{Alg, S}, ws_index) where{Alg, S<:SMatrix}
    tinteg.u = hcat(tinteg.u[:, 1], normalize_devs(tinteg.u))
    u_modified!(tinteg, true)
    return
end
function normalize_deviations!(tinteg::MDI{false}, ws_index)
    tinteg.u = hcat(tinteg.u[:, 1], normalize_devs(tinteg.u))
    u_modified!(tinteg, true)
    return
end
# IIP Versions:
function normalize_deviations!(tinteg::ODEIntegrator{Alg, S}, ws_index) where{Alg, S<:Matrix}
    for i in ws_index
        normalize!(view(tinteg.u, :, i))
    end
    u_modified!(tinteg, true)
    return
end
function normalize_deviations!(tinteg::MDI{true}, ws_index)
    for i in ws_index
        normalize!(view(tinteg.u, :, i))
    end
    u_modified!(tinteg, true)
    return
end

# SVD methods:
singular_values(u::Matrix, ws_index::SVector{k, Int}) where {k} =
svdfact(view(u, :, 2:(k+1)))[:S]
singular_values(u::SMatrix, ws_index::SVector{k, Int}) where {k} =
svdfact(u[:, ws_index]).S

# function test(
#     integ::Union{A, B}) where
#     {A<:ODEIntegrator{Alg, S} where {Alg, S<:SMatrix}, B<:MDI{false}}
#   1
# end
#
# function test(
#     integ::Union{A, B}) where
#     {A<:ODEIntegrator{Alg, S} where {Alg, S<:Matrix}, B<:MDI{true}}
#   1
# end

# Version 1
# test(integ::ODEIntegrator{Alg, S}) where{Alg, S<:SMatrix} = 1
# test(integ::MDI{false}) = 1
#
# # Version 2
# test(integ::ODEIntegrator{Alg, S}, ws_index) where{Alg, S<:Matrix} = 2
# test(integ::MDI{true}) = 2

#=
#####################################################################################
#                                 Discrete GALI                                     #
#####################################################################################
@inbounds function gali(ds::DiscreteDS{D, S, F, JJ}, k::Int,
    tmax, Ws = qr(rand(dimension(ds), dimension(ds)))[1][:, 1:k];
    threshold = 1e-12) where {D,S,F,JJ}

    ws = DynamicalSystemsBase.to_vectorSvector(Ws)

    f = (x) -> ds.eom(x, ds.p)
    J = (x) -> ds.jacob(x, ds.p)
    x = state(ds)

    rett = 0:Int(tmax)
    gali_k = ones(S, length(rett))

    ti=1

    for ti in 2:length(rett)
        # evolve state:
        x = f(x)
        # evolve all deviation vectors:
        jac = J(x)
        for i in 1:k
            ws[i] = normalize(jac*ws[i]) #gotta normalize bro!!!
        end
        # Calculate singular values:
        At = cat(2, ws...) # transpose of "A" in the paper, ref [2].
        zs = svdfact(At)[:S]
        gali_k[ti] =  prod(zs)
        if gali_k[ti] < threshold
            break
        end
    end

    return gali_k[1:ti], rett[1:ti]
end



@inbounds function gali(ds::BigDiscreteDS, k::Int,
    tmax, Ws = qr(rand(dimension(ds), dimension(ds)))[1][:, 1:k];
    threshold = 1e-12)

    ws = DynamicalSystemsBase.to_matrix(Ws)

    f! = (dx, x) -> ds.eom!(dx, x, ds.p)
    jacob! = (J, x) -> ds.jacob!(J, x, ds.p)
    J = ds.J
    x = copy(state(ds))
    xprev = copy(x)
    wsdummy = copy(ws)

    rett = 0:Int(tmax)
    gali_k = ones(eltype(x), length(rett))

    ti=1
    zs = zeros(k)

    for ti in 2:length(rett)
        # evolve state:
        xprev .= x
        f!(x, xprev)
        jacob!(J, x)
        # evolve all deviation vectors:
        for i in 1:k
            A_mul_B!(view(ws, :, i), J, view(wsdummy, :, i))
            normalize!(@view ws[:, i]) #gotta normalize bro!!!
        end
        wsdummy .= ws
        # SVD fact for gali:
        zs .= svdfact(ws)[:S]
        # println("zs = $zs")

        gali_k[ti] =  prod(zs)
        if gali_k[ti] < threshold
            break
        end
    end

    return gali_k[1:ti], rett[1:ti]
end
=#
