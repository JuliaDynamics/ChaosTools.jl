# This file contains parts of the source from SparseArrayKit.jl, which in all honestly
# is very similar to how we compute histograms in Entropies.jl,
# but nevertheless here is the original license:

# MIT License

# Copyright (c) 2020 Jutho Haegeman <jutho.haegeman@ugent.be> and Maarten Van Damme <Maarten.VanDamme@UGent.be>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#########################################################################################
# Basic struct
#########################################################################################
struct SparseArray{T,N} <: AbstractArray{T,N}
    data::Dict{CartesianIndex{N},T}
    dims::NTuple{N,Int64}
    function SparseArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
        return new{T,N}(Dict{CartesianIndex{N},T}(), dims)
    end
end
SparseArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} =
    SparseArray{T,N}(undef, dims)
SparseArray{T}(::UndefInitializer, dims...) where {T} = SparseArray{T}(undef, dims)

Base.empty!(x::SparseArray) = empty!(x.data)
Base.size(a::SparseArray) = a.dims

#########################################################################################
# Indexing
#########################################################################################
@inline function Base.getindex(a::SparseArray{T,N}, I::CartesianIndex{N}) where {T,N}
    @boundscheck checkbounds(a, I)
    return get(a.data, I, zero(T))
end
Base.@propagate_inbounds Base.getindex(a::SparseArray{T,N}, I::Vararg{Int,N}) where {T,N} =
                                        getindex(a, CartesianIndex(I))

@inline function Base.setindex!(a::SparseArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
    @boundscheck checkbounds(a, I)
    if v != zero(v)
        a.data[I] = v
    else
        delete!(a.data, I) # does not do anything if there was no key corresponding to I
    end
    return v
end
Base.@propagate_inbounds function Base.setindex!(
    a::SparseArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
    return setindex!(a, v, CartesianIndex(I))
end