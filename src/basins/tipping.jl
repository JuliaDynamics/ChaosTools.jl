"""
    tipping_probabilities(basins_before, basins_after) → P
Return the tipping probabilities of the computed basins before and after a change
in the system parameters (or time forcing), according to the definition of ^..

The input `basins` are integer-valued arrays, where the integers enumerate the attractor.
They can be of any dimensionality provided that `size(basins_before) == size(basins_after).
Typically they are 2D, as the output of [`basins_map2D`](@ref) or [`basins_general`](@ref)

## Description
Let ``\\mathcal{B}_i(p)`` denote the basin of attraction of attractor ``A_i`` at
parameter(s) ``p``. Kaszás et al[^Kaszás2019] define the tipping probability
from ``A_i`` to ``A_j``, given a parameter change in the system of ``p_- \\to p_+``, as

```math
P(A_i \\to A_j | p_- \\to p_+) =
\\frac{|\\mathcal{B}_j(p_+) \\cap \\mathcal{B}_i(p_-)|}{|\\mathcal{B}_i(p_-)|}
```
where ``|\cdot|`` is simply the volume of the enclosed set.
The value of `` P(A_i \\to A_j | p_- \\to p_+)`` is `P[i, j]`.
The equation describes something quite simple:
what is the overlap of the basin of attraction of ``A_i`` at ``p_-`` with that of the
attractor ``A_j`` at ``p_+``.
If `basins_before, basins_after` contain values of `-1`, corresponding to trajectories
that diverge, these are considered as the last attractor of the system in `P`.

[^Kaszás2019]: Kaszás, Feudel, & Tél. Tipping phenomena in typical dynamical systems
subjected to parameter drift. [Scientific Reports, 9(1)](https://doi.org/10.1038/s41598-019-44863-3)
"""
function tipping_probabilities(basins_before, basins_after)
    @assert size(basins_before) == size(basins_after)

    bid, aid = unique.((basins_before, basins_after))
    P = zeros(length(bid), length(aid))
    N = length(basins_before)
    # Make -1 last entry in bid, aid, if it exists
    put_minus_1_at_end!(bid); put_minus_1_at_end!(aid)

    # Notice: the following loops could be optimized with smarter boolean operations,
    # however they are so fast that everything should be done within milliseconds even
    # on a potato
    for (i, ι) in enumerate(bid)
        B_i = findall(isequal(ι), basins_before)
        μ_B_i = length(B_i)/N # μ = measure
        for (j, ξ) in enumerate(aid)
            B_j = findall(isequal(ξ), basins_after)
            μ_overlap = length(B_i ∩ B_j)/N
            P[i, j] = μ_overlap/μ_B_i
        end
    end
    return P
end

function put_minus_1_at_end!(bid)
    if -1 ∈ bid
        sort!(bid)
        popfirst!(bid)
        push!(bid, -1)
    end
end

# test
ds = Systems.magnetic_pendulum(γ=1, d=0.2, α=0.2, ω=0.8, N=3)
xg=range(-4, 4, length=150)
yg=range(-4, 4, length=150)
@time basins_before, attractors = basins_general(xg, yg, ds; idxs = 1:2, reltol = 1e-9)
attractors

ds = Systems.magnetic_pendulum(γ=1, d=0.2, α=0.2, ω=0.8, N=4)
xg=range(-4, 4, length=150)
yg=range(-4, 4, length=150)
@time basins_after, attractors = basins_general(xg, yg, ds; idxs = 1:2, reltol = 1e-9)
attractors

P = tipping_probabilities(basins_before, basins_after)
