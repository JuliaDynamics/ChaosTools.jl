export statespace_sampler
using Distributions, LinearAlgebra, Random

"""
    statespace_sampler(rng = Random.GLOBAL_RNG; kwargs...) → sampler, isinside
Convenience function that creates two functions. `sampler` is a 0-argument function
that generates random points inside a state space region defined by the keywords.
`isinside` is a 1-argument function that decides returns `true` if the given 
state space point is inside that region.

The regions can be:
* **Rectangular box**, with edges `min_bounds` and `max_bounds`.
  The sampling of the points inside the box is decided by the keyword `method` which can
  be either `"uniform"` or `"multgauss"`.
* **Sphere**, of `spheredims` dimensions, radius `radius` and centered on `center`.
"""
function statespace_sampler(rng = Random.GLOBAL_RNG; 
        min_bounds=[], max_bounds=[], method="uniform", 
        radius::Number=-1,
        spheredims::Int=0, center=zeros(spheredims),
    )

    if min_bounds ≠ [] && max_bounds != []
        if method == "uniform" gen, _ = boxregion(min_bounds, max_bounds, rng)
        elseif method == "multgauss" gen, _ = boxregion_multgauss(min_bounds, max_bounds, rng)
        else @error("Unsupported boxregion sampling method")
        end
    elseif radius ≥ 0 && spheredims ≥ 1
        gen, _ = sphereregion(radius, spheredims, center, rng)
    else
        @error("Incorrect keyword specification.")
    end
    return gen 
end


function boxregion_multgauss(as, bs, rng)
    @assert length(as) == length(bs) > 0
    center = mean(hcat(as,bs), dims=2)
    gen() = [rand(rng, truncated(Normal(center[i]), as[i], bs[i])) for i=1:length(as)]
    isinside(x) = all(as .< x .< bs)
    return gen, isinside
end

# this has a docstring only because it was part of the expansionentropy api.
# It won't be exported in future versions
"""
    boxregion(as, bs, rng = Random.GLOBAL_RNG) -> sampler, isinside

Define a box in ``\\mathbb{R}^d`` with edges the `as` and `bs` and then
return two functions: `sampler`, which generates a random initial condition in that box
and `isinside` that returns `true` if a given state is in the box.
"""
function boxregion(as, bs, rng = Random.GLOBAL_RNG)
    @assert length(as) == length(bs) > 0
    gen() = [rand(rng)*(bs[i]-as[i]) + as[i] for i in 1:length(as)]
    isinside(x) = all(as .< x .< bs)
    return gen, isinside
end

# Specialized 1-d version
function boxregion(a::Real, b::Real, rng = Random.GLOBAL_RNG)
    a, b = extrema((a, b))
    gen() = rand(rng)*(b-a) + a
    isinside = x -> a < x < b
    return gen, isinside
end

#Algorithm is taken from https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere.
#It follows from the fact that a multivariate normal distribution is spherically symmetric.
function sphereregion(r, dim, center, rng)
    @assert r ≥ 0 
    gen() = normalize([( 2*randn(rng) - 1 ) for j=1:dim]) .* r .+ center
    isinside(x) = norm(x .- center) < r
    return gen, isinside
end
