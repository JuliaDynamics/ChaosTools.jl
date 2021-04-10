using LinearAlgebra

"""
    matrix_fdm_gradient(matrix::Array,axis::Number)
Compute the gradient of 2-dimensional array using second order accurate central differences in the interior points and either first order accurate one-sides (forward or backwards) differences at the boundaries. The returned gradient hence has the same shape as the input array. 
We find the standard second order approximation by using: 
```math
\\hat{f}_i^{(1)} = \\frac{f(x_{i+1}-f(x_{i-1})}{2h} + O(h^2)
```


## Arguments
- `matrix::Array{float64}` : The input matrix with two dimensions
- `axis::Number` : Axis to compute the gradient over (1 or 2) 

## Example: 

```julia
random_array = rand(2,22;8,8);
matrix_fdm_gradient(random_array,1)
```


"""
function matrix_gradient(matrix)
    gradient = Array{Float64}(undef, size(matrix));
    gradient[:,1] = (matrix[:,2] .- matrix[:,1]) ;
    gradient[:,end] = (matrix[:,end] .- matrix[:,end-1]);
    gradient[:,2:end-1] = (matrix[:,3:end] - matrix[:,1:end-2]) .*0.5 ;
    end
    return gradient
end

"""
    dyca(data::Array,eig_thresold::float64)
Computes the Dynamical Component analysis matrix [^Uhl2018] used for dimensionality reduction. Here, we solve the main eigenvalue equation: 
```math
C_1 C_0^{-1} C_1^{\\top} \\bar{u} = \\lambda C_2 \\bar{u}

```
where ``C_0`` is the correlation matrix of the signal with itself, ``C_1`` the correlation matrix of the signal with its derivative, and ``C_2`` the correlation matrix of the derivative of the data with itself. The eigenvectors ``\\bar{u}`` to eigenvalues approximately 1 and their ``C_1^{-1} C_2 u`` counterpart form the space where to project onto. 

## Arguments
- `data::Array{float64}` : The input matrix with size= (2,2)
- `eig_thresold::float64` : the eigenvalue thresold for DyCA


## Example: 

```julia
random_array = rand(2,22;100,100);
eigen_thresold = 0.8 ;
DyCA(random_array,eigen_thresold)
```

[^Uhl2018]: B Seifert, K Korn, S Hartmann, C Uhl, *Dynamical Component Analysis (DYCA): Dimensionality Reduction for High-Dimensional Deterministic Time-Series*, 10.1109/mlsp.2018.8517024, 2018 IEEE 28th International Workshop on Machine Learning for Signal Processing (MLSP)


"""
function DyCA(data,eig_thresold=0.98)

    derivative_data = matrix_gradient(data) ;
    time_length = size(data,1) ;#for time averaging
    
     #construct the correlation matrices
    C0 = (transpose(data) * data )/ time_length ;
    C1 = (transpose(derivative_data) * data )/ time_length ;
    C2 = (transpose(derivative_data) * derivative_data )/ time_length ;
    
     #solve the generalized eigenproblem
    eigenvalues,eigenvectors = eigvals(((C1*inv(C0))*transpose(C1)),C2) ;
    eigenvectors = eigenvectors[:,vec(eig_thresold .< eigenvalues .<= 1.0)] ;
    if size(eigenvectors,2) > 0
        C3 = inv(C1) * C2 ;
        proj_mat = hcat(eigenvectors,mapslices(x -> C3*x,eigenvectors,dims=[1]))
    else
        throw(DomainError("No generalized eigenvalue fulfills threshold!"))
    end    
        
    
     return proj_mat, data * proj_mat   
end