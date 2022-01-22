export sampler
using Distributions, LinearAlgebra, Random
"""
    boxregion_multgauss(as, bs) -> gen
Define a box in ``\\mathbb{R}^d`` with edges `as` and `bs` and then
return a generator of initial conditions inside that box.
"""
function boxregion_multgauss(as, bs)
    @assert length(as) == length(bs) > 0
    center = mean(hcat(as,bs), dims=2)
    gen() = [rand(truncated(Normal(center[i]), as[i], bs[i])) for i=1:length(as)]
    return gen
end

"""
    sphereregion(r, dim, center=zeros(length(dim)) -> gen
Define a sphere in ``\\mathbb{R}^dim`` with radius `r` and center `center and return a 
generator of initial conditions inside the sphere. 
Algorithm is taken from https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere.
It follows from the fact that a multivariate normal distribution is spherically symmetric.
"""
function sphereregion(r, dim, center=zeros(length(dim)))
    @assert r ≥ 0 
    gen() = normalize([( 2*randn() - 1 ) for j=1:dim]) .* r .+ center
    return gen
end

"""
    statespace_sampler(rng = Random.default_rng(); kwargs...) → sampler, restraining
Convenience function that creates two functions. `sampler` is a 0-argument function
that generates random points inside a state space region defined by the keywords.
`restraining` is a 1-argument function that decides returns `true` if the given 
state space point is inside that region.

The regions can be:
* **Rectangular box**, with edges `min_bounds` and `max_bounds`.
  The sampling of the points inside the box is decided by the keyword `method` which can
  be either `"uniform"` (see also [`expansionentropy`](@ref)) or `"multgauss"`.
* **Sphere**, of `spheredims` dimensions, radius `radius` and centered on `center`.
"""
function statespace_sampler(rng = Random.default_rng(); 
        min_bounds=[], max_bounds=[], method="uniform", 
        radius::Number=-1,
        spheredims::Int=0, center=zeros(spheredims),
    )

    if min_bounds ≠ [] && max_bounds != []
        if method == "uniform" gen, _ = boxregion(min_bounds, max_bounds)
        elseif method == "multgauss" gen = boxregion_multgauss(min_bounds, max_bounds)
        else @error("Unsupported boxregion sampling method")
        end
    elseif radius ≥ 0 && spheredims ≥ 1
        gen = sphereregion(radius, spheredims, center)
    else
        @error("Incorrect keyword specification.")
    end
    return gen 
end
