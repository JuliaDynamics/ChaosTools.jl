export basin_fractions

"""
    basin_fractions(basins::Array) → fs::Dict
Calculate the fraction of the basins of attraction encoded in `basins`.
The elements of `basins` are integers, enumerating the attractor that the entry of `basins`
converges to. Return a dictionary that maps attractor IDs to their relative fractions.

In[^Menck2013] the authors use these fractions to quantify the stability of a basin of
attraction, and specifically how it changes when a parameter is changed.

[^Menck2013]: Menck, Heitzig, Marwan & Kurths. How basin stability complements the linear stability paradigm. [Nature Physics, 9(2), 89–92](https://doi.org/10.1038/nphys2516)
"""
function basin_fractions(basins::AbstractArray)
    fs = Dict{eltype(basins), Float64}()
    ids = unique(basins)
    N = length(basins)
    for ξ in ids
        B = count(isequal(ξ), basins)
        fs[ξ] = B/N
    end
    return fs
end

"""
    basin_fractions(ds::DynamicalSystem, sampler, featurizer; kwargs...)
Compute the fraction of basins of attraction of the given dynamical system.
`sampler` is a function without any arguments that generates random initial conditions
within a subset of the state space. You can use [`boxregion`](@ref) to create a `sampler`
on a state space box.
`featurizer` is a function that takes as an input an initial condition and outputs a 
vector of "features". Various pre-defined featurizers are available, see below.

## Keyword Arguments
* `N = 10000`.
TBD.

## Description
Let ``F(A)`` be the fraction the basin of attraction of an attractor ``A`` has in the
chosen state space region ``\\mathcal{S}`` given by `sampler`. `basin_fractions` estimates ``F`` for all
attractors in ``\\mathcal{S}`` by randomly sampling `N` initial conditions and counting which
ones end up in which attractors. The error of this approach for each fraction is given by[^Menck2013]
``e = \\sqrt{F(A)(1-F(A))}``.

In `basin_fractions` we do not actually identify attractors, as e.g. done in [`basins_of_attraction`](@ref).
Instead we follow the approach of Stender & Hoffmann[^Stender2021] which transforms each
initial condition into a vector of features, and uses these to efficiently map initial
conditions to attractors using a clustering technique.

## Available featurizers
* lyapunov spectrum
* statistical moments
* whatever else in the original paper, which should be trivial to implement here

[^Menck2013]: Menck, Heitzig, Marwan & Kurths. How basin stability complements the linear stability paradigm. [Nature Physics, 9(2), 89–92](https://doi.org/10.1038/nphys2516)

[^Stender2021]: Stender & Hoffmann, [bSTAB: an open-source software for computing the basin stability of multi-stable dynamical systems](https://doi.org/10.1007/s11071-021-06786-5)
"""
function basin_fractions(ds::DynamicalSystem, args...; kwargs...)
end

# TODO: Perhaps we can optimize the featurizers to instead of initializing an integrator
# all the time by calling `trajectory`, to instead use `reinit!`