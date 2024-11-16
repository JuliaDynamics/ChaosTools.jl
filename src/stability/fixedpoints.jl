import IntervalRootFinding, LinearAlgebra
using IntervalRootFinding: (..), (×), interval
export fixedpoints, .., ×, interval

"""
    fixedpoints(ds::CoreDynamicalSystem, box, J = nothing; kwargs...) → fp, eigs, stable

Return all fixed points `fp` of the given out-of-place `ds`
(either `DeterministicIteratedMap` or `CoupledODEs`)
that exist within the state space subset `box` for parameter configuration `p`.
Fixed points are returned as a [`StateSpaceSet`](@ref).
For convenience, a vector of the Jacobian eigenvalues of each fixed point, and whether
the fixed points are stable or not, are also returned.

`box` is an appropriate interval box from IntervalRootFinding.jl (a vector of intervals).
E.g. for a 3D system it would be something like
```julia
v, z = interval(-5, 5), interval(-2, 2)  # 1D intervals
box = [v, v, z]
```

`J` is the Jacobian of the dynamic rule of `ds`.
It is like in [`TangentDynamicalSystem`](@ref), however in this case automatic Jacobian
estimation does not work, hence a hand-coded version must be given.

Internally IntervalRootFinding.jl is used and as a result we are guaranteed to find all
fixed points that exist in `box`, regardless of stability. Since IntervalRootFinding.jl
returns an interval containing a unique fixed point, we return the midpoint of the
interval as the actual fixed point.
Naturally, limitations inherent to IntervalRootFinding.jl apply here.

The output of `fixedpoints` can be used in the [BifurcationKit.jl](https://github.com/rveltz/BifurcationKit.jl)
as a start of a continuation process. See also [`periodicorbits`](@ref).

## Keyword arguments
- `method = IntervalRootFinding.Krawczyk` configures the root finding method,
  see the docs of IntervalRootFinding.jl for all possibilities.
- `tol = 1e-15` is the root-finding tolerance.
- `warn = true` throw a warning if no fixed points are found.
- `order = nothing` search for fixed points of the n-th iterate of
  [`DeterministicIteratedMap`](@ref). Must be a positive integer or `nothing`.
  Select `nothing` or 1 to search for the fixed points of the original map.

## Performance notes

Setting `order` to a value greater than 5 can be very slow. Consider using
more suitable algorithms for periodic orbit detection, such as
[`periodicorbits`](@ref).

"""
function fixedpoints(ds::DynamicalSystem, box, J = nothing;
        method = IntervalRootFinding.Krawczyk, tol = 1e-15, warn = true,
        order = nothing,
    )
    if isinplace(ds)
        error("`fixedpoints` currently works only for out-of-place dynamical systems.")
    end

    if !(isnothing(order) || (isa(order, Int) && order > 0))
        error("`order` must be a positive integer or `nothing`.")
    end

    p = current_parameters(ds)
    # Find roots via IntervalRootFinding.jl
    fun = to_root_f(ds, p, order)
    jac = to_root_J(J, ds, p, order)
    r = IntervalRootFinding.roots(fun, box; abstol = tol, derivative = jac, contractor = method)
    D = dimension(ds)
    fp = roots_to_dataset(r, D, warn)
    # extract eigenvalues and stability
    eigs = Vector{Vector{Complex{Float64}}}(undef, length(fp))
    if isnothing(J)
        f = dynamic_rule(ds)
        Jf(u, p, t = 0) = DynamicalSystemsBase.ForwardDiff.jacobian(x -> f(x, p, t), u)
    else
        Jf = J
    end
    Jm = zeros(dimension(ds), dimension(ds)) # `eigvals` doesn't work with `SMatrix`
    for (i, u) in enumerate(fp)
        Jm .= Jf(u, p, 0) # notice that we use the "pure" jacobian, no -u!
        eigs[i] = LinearAlgebra.eigvals(Array(Jm))
    end
    stable = Bool[isstable(ds, e) for e in eigs]
    return fp, eigs, stable
end

to_root_J(Jf::Nothing, ::DynamicalSystem, args...) = nothing

to_root_f(ds::CoupledODEs, p, ::Nothing) = u -> dynamic_rule(ds)(u, p, 0.0)
to_root_J(Jf::Function, ::CoupledODEs, p, ::Nothing) = u -> Jf(u, p, 0)

to_root_f(ds::DeterministicIteratedMap, p, ::Nothing) = u -> dynamic_rule(ds)(u, p, 0) - u
function to_root_J(Jf::Function, ds::DeterministicIteratedMap, p, ::Nothing)
    c = Diagonal(ones(typeof(current_state(ds))))
    return u -> Jf(u, p, 0) - c
end

# Discrete with periodic order
function to_root_f(ds::DeterministicIteratedMap, p, o::Int)
    f = dynamic_rule(ds)
    u -> begin
        v = u
        for _ in 1:o
            v = f(v, p, 0)
        end
        return v - u
    end
end

# this can be derived from chain rule : J(f(g(x))) = Jf(g(x))*Jg(x)
function to_root_J(Jf, ds::DeterministicIteratedMap, p, o::Int)
    c = Diagonal(ones(typeof(current_state(ds))))
    d = dimension(ds)
    u -> begin
        J = Jf(u, p, 0.0)
        x = ds.f(u, p, 0.0)
        for i in 2:o
            J = Jf(x, p, 0.0) * J
            x = ds.f(x, p, 0.0)
        end
        return J .- c
    end
end


function roots_to_dataset(r, D, warn)
    if isempty(r) && warn
        @warn "No fixed points found!"
        return StateSpaceSet{D, Float64}()
    end
    if any(root.status != :unique for root in r) && warn
        @warn "Non-unique fixed points found!"
    end
    F = zeros(length(r), D)
    for (j, root) in enumerate(r)
        F[j, :] .= map(i -> (i.bareinterval.hi + i.bareinterval.lo)/2, root.region)
    end
    return StateSpaceSet(F; warn = false)
end

isstable(::CoupledODEs, e) = maximum(real(x) for x in e) < 0
isstable(::DeterministicIteratedMap, e) = maximum(abs(x) for x in e) < 1