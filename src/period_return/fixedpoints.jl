import IntervalRootFinding, LinearAlgebra
using IntervalRootFinding: (..), (×)
export fixedpoints, .., ×

"""
    fixedpoints(ds::DynamicalSystem, box, p = ds.p; kwargs...) → fp, eigs, stable
Return all fixed points `fp` of the given `ds`
that exist within the state space subset `box` for parameter configuration `p`.
Fixed points are returned as a [`Dataset`](@ref).
For convenience, a vector of the Jacobian eigenvalues of each fixed point, and whether 
the fixed points are stable or not, are also returned.
`fixedpoints` is valid for both discrete and continuous systems, but only for out of place
format (see [`DynamicalSystem`](@ref)).

Internally IntervalRootFinding.jl is used and as a result we are guaranteed to find all
fixed points that exist in `box`, regardless of stability. Since IntervalRootFinding.jl
returns an interval containing a unique fixed point, we return the midpoint of the
interval as a fixed point.
`box` is an appropriate `IntervalBox` from IntervalRootFinding.jl. 
E.g. for a 3D system it would be something like 
```julia
v, z = -5..5, -2..2   # 1D intervals
box = v × v × z       # `\\times = ×`, or use `IntervalBox`
```

## Keywords
* `method = IntervalRootFinding.Krawczyk` configures the root finding method, 
  see the docs of IntervalRootFinding.jl for all posibilities.
* `tol = 1e-15` is the root-finding tolerance.
* `o = nothing` if given, must be an integer. It finds `o`-th order fixed points
  (i.e., periodic orbits of length `o`). It is only valid for discrete dynamical systems.
"""
function fixedpoints(ds::DynamicalSystem, box, p = ds.p;
        method = IntervalRootFinding.Krawczyk, o = nothing, tol = 1e-15
    )
    DynamicalSystemsBase.isinplace(ds) && error("`fixedpoints` works only for out-of-place dynamical systems.")
    # Find roots via IntervalRootFinding.jl
    f = to_root_f(ds, p, o)
    jac = to_root_J(ds, p, o)
    r = IntervalRootFinding.roots(f, jac, box, method, tol)
    D = dimension(ds)
    fp::Dataset{D, Float64} = roots_to_dataset(r, D)
    # Find eigenvalues and stability
    eigs = Vector{Vector{Complex{Float64}}}(undef, length(fp))
    J = Array(jacobian(ds)) # `eigvals` doesn't work with StaticArrays.jl
    for (i, u) in enumerate(fp)
        J .= jacobian(ds, u, p, 0.0) # notice that we use the "pure" jacobian, no -u!
        eigs[i] = LinearAlgebra.eigvals(Array(J))
    end
    stable = Bool[isstable(ds, e) for e in eigs]
    return fp, eigs, stable
end

to_root_f(ds::CDS, p, ::Nothing) = u -> ds.f(u, p, 0.0)
to_root_J(ds::CDS, p, ::Nothing) = u -> ds.jacobian(u, p, 0.0)
to_root_f(ds::DDS, p, ::Nothing) = u -> ds.f(u, p, 0.0) - u
function to_root_J(ds::DDS{IIP, S}, p, ::Nothing) where {IIP, S}
    c = Diagonal(ones(S))
    return u -> ds.jacobian(u, p, 0.0) - c
end

# Discrete with periodic order
function to_root_f(ds::DDS, p, o::Int) 
    u -> begin
        v = copy(u) # copy is free for StaticArrays
        for _ in 1:o
            v = ds.f(v, p, 0.0)
        end
        return v - u
    end
end


function roots_to_dataset(r, D)
    if isempty(r)
        @warn "No fixed points found!"
        return Dataset{D, Float64}()
    end
    if any(root.status != :unique for root in r)
        @warn "Non-unique fixed points found!"
    end
    F = zeros(length(r), D)
    for (j, root) in enumerate(r)
        F[j, :] .= map(i -> (i.hi + i.lo)/2, root.interval)
    end
    return Dataset(F; warn = false)
end

isstable(::CDS, e) = maximum(real(x) for x in e) < 0
isstable(::DDS, e) = maximum(abs(x) for x in e) < 1