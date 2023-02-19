import IntervalRootFinding, LinearAlgebra
using IntervalRootFinding: (..), (×), IntervalBox, interval
export fixedpoints, .., ×, IntervalBox, interval

"""
    fixedpoints(ds::CoreDynamicalSystem, box, J = nothing; kwargs...) → fp, eigs, stable

Return all fixed points `fp` of the given out-of-place `ds`
(either `DeterministicIteratedMap` or `CoupledODEs`)
that exist within the state space subset `box` for parameter configuration `p`.
Fixed points are returned as a [`Dataset`](@ref).
For convenience, a vector of the Jacobian eigenvalues of each fixed point, and whether
the fixed points are stable or not, are also returned.

`box` is an appropriate `IntervalBox` from IntervalRootFinding.jl.
E.g. for a 3D system it would be something like
```julia
v, z = -5..5, -2..2   # 1D intervals, can use `interval(-5, 5)` instead
box = v × v × z       # `\\times = ×`, or use `IntervalBox(v, v, z)` instead
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
  see the docs of IntervalRootFinding.jl for all posibilities.
- `tol = 1e-15` is the root-finding tolerance.
- `warn = true` throw a warning if no fixed points are found.
"""
function fixedpoints(ds::DynamicalSystem, box, J;
        method = IntervalRootFinding.Krawczyk, tol = 1e-15, warn = true,
        o = nothing, # the keyword `o` will be the period in a future version...
    )
    if isinplace(ds)
        error("`fixedpoints` works only for out-of-place dynamical systems.")
    end
    # Jacobian: copy code from `DynamicalSystemsBase`
    f = dynamic_rule(ds)
    if isnothing(J)
        error("At the moment automatic differentiation doesn't work...")
        Jf = (u, p, t) -> DynamicalSystemsBase.ForwardDiff.jacobian((x) -> f(x, p, t), u)
    else
        Jf = J
    end
    p = current_parameters(ds)
    # Find roots via IntervalRootFinding.jl
    f = to_root_f(ds, p, o)
    jac = to_root_J(Jf, ds, p, o)
    r = IntervalRootFinding.roots(f, jac, box, method, tol)
    D = dimension(ds)
    fp::Dataset{D, Float64} = roots_to_dataset(r, D, warn)
    # Find eigenvalues and stability
    eigs = Vector{Vector{Complex{Float64}}}(undef, length(fp))
    J = zeros(dimension(ds), dimension(ds)) # `eigvals` doesn't work with `SMatrix`
    for (i, u) in enumerate(fp)
        J .= Jf(u, p, 0) # notice that we use the "pure" jacobian, no -u!
        eigs[i] = LinearAlgebra.eigvals(Array(J))
    end
    stable = Bool[isstable(ds, e) for e in eigs]
    return fp, eigs, stable
end

to_root_f(ds::CoupledODEs, p, ::Nothing) = u -> dynamic_rule(ds)(u, p, 0.0)
to_root_J(Jf, ::CoupledODEs, p, ::Nothing) = u -> Jf(u, p, 0.0)

to_root_f(ds::DeterministicIteratedMap, p, ::Nothing) = u -> dynamic_rule(ds)(u, p, 0) - u
function to_root_J(Jf, ds::DeterministicIteratedMap, p, ::Nothing)
    c = Diagonal(ones(typeof(current_state(ds))))
    return u -> Jf(u, p, 0) - c
end

# Discrete with periodic order
function to_root_f(ds::DeterministicIteratedMap, p, o::Int)
    f = dynamic_rule(ds)
    u -> begin
        v = copy(u) # copy is free for `SVector`
        for _ in 1:o
            v = f(v, p, 0)
        end
        return v - u
    end
end
# TODO: Estimate periodic orbits of period `o` for discrete systems.
# What's left is to create the Jacobian for higher iterates.


function roots_to_dataset(r, D, warn)
    if isempty(r) && warn
        @warn "No fixed points found!"
        return Dataset{D, Float64}()
    end
    if any(root.status != :unique for root in r) && warn
        @warn "Non-unique fixed points found!"
    end
    F = zeros(length(r), D)
    for (j, root) in enumerate(r)
        F[j, :] .= map(i -> (i.hi + i.lo)/2, root.interval)
    end
    return Dataset(F; warn = false)
end

isstable(::CoupledODEs, e) = maximum(real(x) for x in e) < 0
isstable(::DeterministicIteratedMap, e) = maximum(abs(x) for x in e) < 1