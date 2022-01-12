export sampler
using Distributions, LinearAlgebra
"""
    boxregion_multgauss(as, bs) -> sampler, restraining
Define a box in ``\\mathbb{R}^d`` with edges the `as` and `bs` and then
return two functions: `sampler`, which generates a random initial condition in that box
and `restraining` that returns `true` if a given state is in the box.
"""
function boxregion_multgauss(as, bs)
    @assert length(as) == length(bs) > 0
    center = mean(hcat(as,bs), dims=2)
    gen() = [rand(truncated(Normal(center[i]), as[i], bs[i])) for i=1:length(as)]
    return gen
end

"""
    sphereregion(r, dim, center=zeros(length(dim)) -> sampler
Define a sphere in ``\\mathbb{R}^dim`` with radius `r` and center `center. 
Return: `sampler`, which generates a random initial condition in that box.
Algorithm is taken from https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere.
It follows from the fact that a multivariate normal distribution is spherically symmetric.
"""
function sphereregion(r, dim, center=zeros(length(dim)))
    @assert r ≥ 0 
    gen() = normalize([( 2*randn() - 1 ) for j=1:dim]) .* r .+ center
    return gen
end


"""
Convenience sampler function for generating random points inside a region. Returns a
     generator for the points. The region can be:
    * a box, with edges `min_bounds` and `max_bounds`. The sampling of the points inside this
    box is according to `method`, which can be either `"uniform"` (implemented in 
    `expansionentropy.jl`) or `"multgauss"`.
    * a sphare, of `spheredims` dimensions, radius `radius` and centered on `center`.
    
"""
function sampler(; min_bounds::Vector=[], max_bounds::Vector=[], method="uniform", radius::Number=-1, spheredims::Int=0, center=zeros(spheredims) )
    if min_bounds ≠ [] && max_bounds != []
        if method == "uniform" gen, _ = boxregion(min_bounds, max_bounds)
        elseif method == "multgauss" gen = boxregion_multgauss(min_bounds, max_bounds)
        else @error("Unsupported boxregion sampling method")
        end
    elseif radius ≥ 0 && spheredims ≥ 1
        gen = sphereregion(radius, spheredims, center)
    else
        @error("No method found.")
    end
    return gen 
end