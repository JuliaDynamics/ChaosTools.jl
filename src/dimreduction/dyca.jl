export dyca
using LinearAlgebra

"""
    dyca(data, eig_thresold) -> eigenvalues, proj_mat, projected_data
Compute the Dynamical Component analysis (DyCA) of the given `data` [^Uhl2018]
used for dimensionality reduction.

Return the eigenvalues, projection matrix, and reduced-dimension data
(which are just `data*proj_mat`).
## Keyword Arguments
* norm_eigenvectors=false : if true, normalize the eigenvectors

## Description
Dynamical Component Analysis (DyCA) is a method to detect projection vectors to reduce
the dimensionality of multi-variate, high-dimensional deterministic datasets. Unlike
methods like PCA or ICA that make a stochasticity assumption, DyCA relies on a determinacy
assumption on the time-series and is based on the solution of a generalized eigenvalue
problem. After choosing an appropriate eigenvalue threshold and solving the eigenvalue
problem, the obtained eigenvectors are used to project the high-dimensional dataset onto
a lower dimension. The obtained eigenvalues measure the quality of the assumption of linear
determinism for the investigated data. Furthermore, the number of the generalized eigenvalues
with a value of approximately 1.0 are a measure of the number of linear equations contained
in the dataset. This property is useful in detecting regions with highly deterministic parts
in the time-series and also as a preprocessing step for reservoir computing of high-dimensional
spatio-temporal data.

The generalised eigenvalue problem we solve is:

```math
C_1 C_0^{-1} C_1^{\\top} \\bar{u} = \\lambda C_2 \\bar{u}

```
where ``C_0`` is the correlation matrix of the data with itself, ``C_1`` the correlation
matrix of the data with its derivative, and ``C_2`` the correlation matrix of the
derivative of the data with itself. The eigenvectors ``\\bar{u}`` with eigenvalues
approximately 1 and their ``C_1^{-1} C_2 u`` counterpart, form the space where the data
is projected onto.

[^Uhl2018]:
    B Seifert, K Korn, S Hartmann, C Uhl, *Dynamical Component Analysis (DYCA):
    Dimensionality Reduction for High-Dimensional Deterministic Time-Series*,
    10.1109/mlsp.2018.8517024, 2018 IEEE 28th International Workshop on Machine Learning
    for Signal Processing (MLSP)
"""
dyca(A::Dataset, e) = dyca(Matrix(A), e)
function dyca(data, eig_thresold::AbstractFloat; norm_eigenvectors::Bool=false)
    derivative_data = matrix_gradient(data)  #get the derivative of the data
    time_length = size(data,1) #for time averaging

    #construct the correlation matrices
    C0 = Array{Float64, 2}(undef, size(data,2), size(data,2))
    C1, C2, C3 = copy(C0), copy(C0), copy(C0)
    mul!(C0, transpose(data), data/ time_length)
    mul!(C1, transpose(derivative_data), data/ time_length)
    mul!(C2, transpose(derivative_data), derivative_data/ time_length)

    #solve the generalized eigenproblem
    eigenvalues, eigenvectors = eigen(C1*inv(C0)*transpose(C1),C2)
    norm_eigenvectors && normalize_eigenvectors!(eigenvectors)
    eigenvectors = eigenvectors[:, vec(eig_thresold .< abs.(eigenvalues) .<= 1.0)]
    if size(eigenvectors, 2) > 0
        mul!(C3, inv(C1), C2)
        proj_mat = hcat(eigenvectors,mapslices(x -> C3*x,eigenvectors,dims=[1]))
    else
        error("No generalized eigenvalue fulfills threshold!")
    end
     return eigenvalues, proj_mat, data*proj_mat
end

function normalize_eigenvectors!(eigenvectors)
    for i=1:size(eigenvectors,2)
        eigenvectors[:,i] = normalize(eigenvectors[:,i])
    end
end


"""
    matrix_gradient(matrix)
Compute the gradient of a matrix along 1st axis.

## Description
Compute the gradient using second order accurate central differences in the interior points
and first order accurate one-sides differences at the boundaries. We find the standard
second order approximation by using:
```math
\\hat{f}_i^{(1)} = \\frac{f(x_{i+1}-f(x_{i-1})}{2h} + O(h^2)
```
The returned gradient matrix hence has the same shape as the input array. Here we compute
the gradient along axis=1 (row-wise), so to compute gradient along axis=2 (column-wise),
the tranpose of the input matrix must be given.
"""
function matrix_gradient(matrix::Matrix)
    gradient = copy(matrix)
    gradient[1,:] = (matrix[2,:] .- matrix[1,:])
    gradient[end,:] = (matrix[end,:] .- matrix[end-1,:])
    gradient[2:end-1,:] = (matrix[3:end,:] .- matrix[1:end-2,:]) .*0.5
    return gradient
end
