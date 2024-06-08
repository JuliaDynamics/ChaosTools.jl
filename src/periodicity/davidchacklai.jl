export davidchacklai

"""
    davidchacklai(ds::DeterministicIteratedMap, n::Int, ics, m::Int; kwargs...) -> fps

Find periodic orbits `fps` of orders `1` to `n` for the map `ds`
using the algorithm propesed by Davidchack & Lai[Davidchack1999](@cite).
`ics` is a collection of initial conditions (container of vectors) to be evolved.
`ics` will be used to detect periodic orbits of orders `1` to `m`. These `m` 
periodic orbits will be used to detect periodic orbits of order `m+1` to `n`.
`fps` is a vector with `n` elements. `i`-th element is a periodic orbit of order `i`.

## Keyword arguments

* `β = nothing`: If it is nothing, then `β(n) = 10*1.2^n`. Otherwise can be a 
   function that takes period `n` and return a number. It is a parameter mentioned
   in the paper[Davidchack1999](@cite).
* `maxiters = nothing`: If it is nothing, then initial condition will be iterated
  `max(100, 4*β(p))` times (where `p` is the order of the periodic orbit)
   before claiming it has not converged. If an integer, then it is the maximum 
   amount of iterations an initial condition will be iterated before claiming 
   it has not converged.
* `disttol = 1e-10`: Distance tolerance. If `norm(f^{n}(x)-x) < disttol` 
   where `f^{n}` is the `n`-th iterate of the dynamic rule `f`, then `x` 
   is an `n`-periodic point.
* `abstol = 1e-8`: A detected periodic point isn't stored if it is in `abstol` 
   neighborhood of some previously detected point. Distance is measured by 
   euclidian norm. If you are getting duplicate periodic points, decrease this value.

## Description

The algorithm is an extension of Schmelcher & Diakonos[Schmelcher1997](@cite)
implemented in [`periodicorbits`](@ref).

The algorithm can detect periodic orbits
by turning fixed points of the original
map `ds` to stable ones, through the transformation
```math
\\mathbf{x}_{n+1} = \\mathbf{x}_{n} + 
[\\beta |g(\\mathbf{x}_{n}| C^{T} - J(\\mathbf{x}_{n})]^{-1} g(\\mathbf{x}_{n})
```
where
```math
g(\\mathbf{x}_{n}) = f^{n}(\\mathbf{x}_{n}) - \\mathbf{x}_{n}
```
and
```math
J(\\mathbf{x}_{n}) = \\frac{\\partial g(\\mathbf{x}_{n})}{\\partial \\mathbf{x}_{n}}
````

The main difference between Schmelcher & Diakonos[Schmelcher1997](@cite) and 
Davidchack & Lai[Davidchack1999](@cite) is that the latter uses periodic points of
previous period as seeds to detect periodic points of the next period.
Additionally, [`periodicorbits`](@ref) only detects periodic points of a given order, 
while `davidchacklai` detects periodic points of all orders up to `n`.


## Important note

For low periods `n` circa less than 6, you should select `m = n` otherwise the algorithm 
detect periodic orbits correctly. For higher periods, you can select `m` as 6. 
You can use initial grid of points for `ics`. Increase `m` in case the orbits are 
not being detected correctly.

"""
function davidchacklai(
        ds::DeterministicIteratedMap,
        n::Int,
        ics,
        m::Int = 1;
        kwargs...
    )
    if isinplace(ds)
        throw(ArgumentError("`davidchacklai` currently supports only out of place systems."))
    end

    if (n < 1)
        throw(ArgumentError("`n` must be a positive integer."))
    end

    if !(1 <= m <= n)
        throw(ArgumentError("`m` must be in [1, `n`=$(n)]"))
    end

    type = typeof(current_state(ds))
    fps = storage(type, n)
    detection!(fps, ds, n, ics, m; kwargs...)
    return output(fps, type, n)
end

function detection!(fps, ds, n, ics, m; β=nothing, kwargs...)
    betagen=betagenerator(β)
    indss, signss = lambdaperms(dimension(ds))
    C_matrices = [cmatrix(inds,signs) for inds in indss, signs in signss]

    initial_detection!(fps, ds, ics, m, betagen, C_matrices; kwargs...)
    main_detection!(fps, ds, n, betagen, C_matrices; kwargs...)
end

function betagenerator(β)
    if isnothing(β)
        return n-> 10*1.2^(n)
    else
        return β
    end
end

function initial_detection!(fps, ds, ics, m, betagenerator, C_matrices; kwargs...)
    for i in 1:m
        detect_orbits(fps[i], ds, i, ics, betagenerator(i), C_matrices; kwargs...)
    end
end

function main_detection!(fps, ds, n, betagenerator, C_matrices; kwargs...)
    for period = 2:n
        β = betagenerator(period)
        previousfps = fps[period-1]
        currentfps = fps[period]
        nextfps = fps[period+1]
        for (container, seed, order) in [
            (currentfps, previousfps, period), 
            (nextfps, currentfps, period+1), 
            (currentfps, nextfps, period)
            ]
            detect_orbits(container, ds, order, seed, β, C_matrices; kwargs...)
        end
    end
end

function _detect_orbits!(fps, ds, n, seed, C, β; disttol::Float64=1e-10, abstol::Float64=1e-8, maxiters=nothing)
    for x in seed
        for _ in 1:(isnothing(maxiters) ? max(100, 4*β) : maxiters)
            xn = DL_rule(x, β, C, ds, n)
            if converged(ds, xn, n, disttol)
                if previously_detected(fps, xn, abstol) == false
                    completeorbit!(fps, ds, xn, n, disttol, abstol)
                end
                break
            end
            x = xn
        end
    end
end

function completeorbit!(fps, ds, xn, n, disttol, abstol)
    traj = trajectory(ds, n, xn)[1]
    for t in traj
        if converged(ds, t, n, disttol)
            storefp!(fps, t, abstol)
        end
    end
end

function converged(ds, xn, n, disttol)
    return norm(g(ds, xn, n)) < disttol
end


function detect_orbits(
        fps::Set{T},
        ds::DeterministicIteratedMap,
        n::Integer,
        seed::AbstractVector{D},
        β,
        C_matrices;
        kwargs...
    ) where {T, D}
    for C in C_matrices
        _detect_orbits!(fps, ds, n, seed, C, β; kwargs...)
    end
end

function detect_orbits(
        fps::Set{T},
        ds::DeterministicIteratedMap,
        n::Integer,
        seed::Set{T},
        β,
        C_matrices;
        kwargs...
    ) where {T}
    for C in C_matrices
        _detect_orbits!(fps, ds, n, collect(seed), C, β; kwargs...)
    end
end

function DL_rule(x, β, C, ds, n)
    Jx = DynamicalSystemsBase.ForwardDiff.jacobian(x0 -> g(ds, x0, n), x)
    gx = g(ds, x, n)
    xn = x + inv(β*norm(gx)*C' - Jx) * gx
    return xn
end

function g(ds, state, n)
    p = current_parameters(ds)
    newst = state
    # not using step!(ds, n) to allow automatic jacobian
    for _ = 1:n
        newst = ds.f(newst, p, 1.0)
    end
    return newst - state
end

function output(fps, type, n)
    output = Vector{Vector{type}}(undef, n)
    for i in 1:n # not including periodic orbit n+1 because it may be incomplete
        output[i] = collect(fps[i])
    end
    return output
end

function storage(type, n)
    storage = Vector{Set{type}}(undef, n+1)
    for i in 1:n+1
        storage[i] = Set{type}()
    end
    return storage
end