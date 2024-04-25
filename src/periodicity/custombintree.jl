using DataStructures: RBTree, search_node
import Base: <, ==

# This structure is used to overload the behavior of < and == for the use in a binary tree.
# This way binary tree will store inserted values only if they are at least `eps` away from each other.
# For example if the root is 0.0 and `eps` is 1.0 then 1.1 will be stored as a child on the right.
# However, 0.9 won't be stored because it is in the `eps` neighborhood of 0.0.
struct VectorWithEpsRadius{T}
    value::T
    eps::Float64
end

function <(a::VectorWithEpsRadius, b::VectorWithEpsRadius)
    abs_diff = norm(a.value - b.value)
    return abs_diff > a.eps && a.value < b.value
end

function ==(a::VectorWithEpsRadius, b::VectorWithEpsRadius)
    return norm(a.value - b.value) <= a.eps
end

function tovector(tree::RBTree{VectorWithEpsRadius{T}}) where T
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

# gererates datastructure for storing fixed points
function fpcollection(type)
    # Binary tree is used for fast insertion and searching of fixed points.
    # VectorWithEpsRadius wrapper ensures that each point in the binary
    # tree is at least `abstol` away from each other. This eliminated duplicate
    # fixed points in the output.
    collection = RBTree{VectorWithEpsRadius{type}}()
    return collection
end

function storefp!(collection, fp, abstol)
    push!(collection, VectorWithEpsRadius{typeof(fp)}(fp, abstol))
end

function previously_detected(tree, fp, abstol)
    return !isnothing(search_node(tree, VectorWithEpsRadius{typeof(fp)}(fp, abstol)).data)
end