################################################################################
# Molteno histogram based dimension by boxing values
################################################################################
"""
    molteno_dim(α, data::Dataset, k0 = 10; base = Base.MathConstants.e)
Calculate the generalized dimension using the algorithm for box division defined
by Molteno[^Molteno1993].

## Description
Divide the data into boxes with each new box having half the side length of the
former box using [`molteno_boxing`](@ref). Break if the number of points over
the number of filled boxes falls below `k0`. Then the generalized dimension can
be calculated by using [`genentropy`](@ref) to calculate the sum over the
logarithm also considering possible approximations and fitting this to the
logarithm of one over the boxsize using [`linear_region`](@ref).

This algorithm is faster than the traditional approach of using [`non0hist`](@ref),
but it is only suited for low dimensional data since it divides each
box into `2^D` new boxes if `D` is the dimension. For large `D` this leads to low numbers of
box divisions before the threshold is passed and the divison stops and as a result
to a low number of data points to fit the dimension to and thereby a poor
estimate.

[^Molteno1993]: Molteno, T. C. A., [Fast O(N) box-counting algorithm for estimating dimensions. Phys. Rev. E 48, R3263(R) (1993)](https://doi.org/10.1103/PhysRevE.48.R3263)
"""
function molteno_dim(α, data, k0 = 10; base = ℯ)
    boxes, εs = molteno_boxing(data, k0)
    dd = genentropy.(boxes; α = α, base = base)
    return linear_region(-log.(base, εs), dd)[2]
end

"""
    molteno_boxing(data::Dataset, k0 = 10) → (boxes, εs)
Distribute the `data` into boxes whose size is halved in each step. Stop if the
average number of points per filled box falls below the threshold `k0`.

Return `boxes`, a vector of `Propabilities` for different box sizes and the
corresponding box sizes `εs`.

## Description
Project the `data` onto the whole interval of numbers that is covered by
`UInt64`. This projected data is distributed into boxes whose size
decreases by factor 2 in each step. For each box that contains more than one
point `2^D` new boxes are created where `D` is the dimension of the data.

The new boxes are stored in a vector. The data points are distributed into
these boxes by bit shifting and an `&`-comparison to check whether the `i`th
bit of the value is one or zero. For more than one dimension the values of the
comparison are multiplied with `2^j` if `j` iterates through `0:dim-1` and
added up afterwards.

The process of dividing the data into new boxes stops when the number of points
over the number of filled boxes falls below `k0`. The box sizes `εs` are
calculated and returned together with the `boxes`.
"""
function molteno_boxing(data::Dataset, k0 = 10)
    integers, ε0 = float_to_int(data)
    boxes = _molteno_boxing(integers, k0)
    εs = ε0 ./ 2 .^ (1:length(boxes))
    return boxes, εs
end

"""
    float_to_int(data::Dataset{D,T}) where {D, T}
Calculate maximum and minimum value of `data` to then project the values onto
``[0 + \\epsilon, 1 + \\epsilon] \\cdot M`` where ``\\epsilon`` is the
precision of the used Type and ``M`` is the maximum value of the UInt64 type.
"""
function float_to_int(data::Dataset{D,T}) where {D,T}
    N = length(data)
    mins, maxs = minmaxima(data)
    sizes = maxs .- mins
    ε0 = maximum(sizes)
    # Let f:[min,max] -> [0+eps(T),1-eps(T)]*typemax(UInt64), then f(x) = m*x + b.
    m = (1-2eps(T)) ./ ε0 .* typemax(UInt64)
    b = eps(T) * typemax(UInt64) .- mins .* m

    res = Vector{SVector{D,UInt64}}()
    sizehint!(res, N)
    for x in data
        int_val = floor.(UInt64, m .* x .+ b)
        push!(res, int_val)
    end
    Dataset(res), ε0
end

function _molteno_boxing(data::Dataset{D,T}, k0 = 10) where {D,T}
    N = length(data)
    box_probs = Vector{Float64}[]
    iteration = 1
    boxes = [[1:N;]]
    while N / length(boxes) > k0
        l = length(boxes)
        for t in 1:l
            # Take the first box.
            box = popfirst!(boxes)
            # Check if only one element is contained.
            if length(box) == 1
                push!(boxes, box)
                continue
            end
            # Append a new partitioned box.
            append!(boxes, molteno_subboxes(box, data, iteration))
        end
        # Calculate probabilities by dividing the number of elements in a box by N.
        push!(box_probs, length.(boxes) ./ N)
        iteration += 1
    end
    Probabilities.(box_probs)
end

"""
    molteno_subboxes(box, data::AbstractVector{S}, iteration) where {D,S<:SVector{D,UInt64}}
Divide a `box` containing indices into `data` to `2^D` smaller boxes and sort
the points contained in the former box into the new boxes. Implemented according to
Molteno[^Molteno]. Sorts the elements of the former box into the smaller boxes
using cheap bit shifting and `&` operations on the value of `data` at each box
element. `iteration` determines which bit of the array should be shifted to the
last position.
"""
function molteno_subboxes(box, data::Dataset{D,UInt64}, iteration) where {D}
    new_boxes = [UInt64[] for i in 1:2^D]
    index_multipliers = [2^i for i in 0:D-1]
    sorting_number = 64-iteration
    for elem in box
        index = one(UInt64)
        for (i, multi) in enumerate(index_multipliers)
            # index shifting magic
            index += ((data[elem][i] >> sorting_number) & 1) * multi
        end
        push!(new_boxes[index], elem)
    end
    filter!(!isempty, new_boxes)
end
