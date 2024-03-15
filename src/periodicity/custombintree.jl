using DataStructures: RBTree
import Base: <, ==

struct DummyStructure{T}
    value::T
    eps::Float64
end

function <(a::DummyStructure, b::DummyStructure)
    abs_diff = norm(a.value - b.value)
    return abs_diff > a.eps && a.value < b.value
end

function ==(a::DummyStructure, b::DummyStructure)
    return norm(a.value - b.value) <= a.eps
end

function tovector(tree::RBTree{DummyStructure{T}}) where T
    len = length(tree)
    type = typeof(tree).parameters[1].parameters[1]
    vect = Vector{type}(undef, len)
    index = 1
    _tovector!(tree.root, vect, index)
    return vect
end

function _tovector!(node, vect::Vector{T}, index::Int64) where T
    if !isnothing(node.data)
        index = _tovector!(node.leftChild, vect, index)
        vect[index] = node.data.value
        index += 1
        index = _tovector!(node.rightChild, vect, index)
    end
    return index
end