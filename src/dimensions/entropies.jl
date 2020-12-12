#=
The code that used to be on this document has been moved to Entropies.jl
and improved in its entirety. This page now only holds convenience functions.
=#
using Entropies
export non0hist, binhist, genentropy, permentropy, probabilities

using Combinatorics: permutations

"""
    permentropy(x, m = 3; τ = 1, base = Base.MathConstants.e)

Compute the permutation entropy[^Brandt2002] of given order `m`
from the `x` timeseries.

This method is equivalent with
```julia
genentropy(x, SymbolicPermutation(; m, τ); base)
```

[^Bandt2002]: C. Bandt, & B. Pompe, [Phys. Rev. Lett. **88** (17), pp 174102 (2002)](http://doi.org/10.1103/PhysRevLett.88.174102)
"""
function permentropy(x, m = 3; τ = 1, base = Base.MathConstants.e)
    Entropies.genentropy(x, SymbolicPermutation(; τ = 1, m = 3))
end
