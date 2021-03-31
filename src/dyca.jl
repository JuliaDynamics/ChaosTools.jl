using LinearAlgebra

"""
    matrix_fdm_gradient(matrix::Array,axis::Number)
Compute the gradient of 2-dimensional array using second order accurate central differences in the interior points and either first order accurate one-sides (forward or backwards) differences at the boundaries. The returned gradient hence has the same shape as the input array. 

## Keyword Arguments
- `matrix::Array{float64}` : The input matrix with two dimensions
- `axis::Number` : Axis to compute the gradient over (1 or 2) 

## Example: 

```julia
random_array = rand(2,22;8,8);
matrix_fdm_gradient(random_array,1)
```


"""
function matrix_fdm_gradient(matrix,axis::Number)
    gradient = Array{Float64}(undef, size(matrix));
    if axis == 1
        gradient[:,1] = (matrix[:,2] .- matrix[:,1]) ;
        gradient[:,end] = (matrix[:,end] .- matrix[:,end-1]);
        gradient[:,2:end-1] = (matrix[:,3:end] - matrix[:,1:end-2]) .*0.5 ;
    elseif axis == 2
        gradient[1,:] = (matrix[2,:] .- matrix[1,:]) ;
        gradient[end,:] = (matrix[end,:] .- matrix[end-1,:]);
        gradient[2:end-1,:] = (matrix[3:end,:] .- matrix[1:end-2,:]) .*0.5 ;
    end
    return gradient
end

"""
    DyCA(data::Array,eig_thresold::float64)
Computes the Dynamical Component analysis matrix used for Dimensionality reduction. Here, we solve the main eigenvalue equation: 
$$C_1 C_0^{-1} C_1^\top u = \lambda C_2 u$$
where $C_0$ is the correlation matrix of the signal with itself, $C_1$ the correlation matrix of the signal with its derivative, and $C_2$ the correlation matrix of the derivative of the data with itself. The eigenvectors $u$ to eigenvalues approximately $1$ and their $C_1^{-1} C_2 u$ counterpart form the space where to project onto. 

## Keyword Arguments
- `data::Array{float64}` : The input matrix with dimensions (2,2)
- `eig_thresold::float64` : the eigenvalue thresold for DyCA


## Example: 

```julia
random_array = rand(2,22;100,100);
eigen_thresold = 0.8 ;
DyCA(random_array,eigen_thresold)
```

Source : https://arxiv.org/abs/1807.10629 


"""
function DyCA(data,eig_thresold=0.98)

    derivative_data = matrix_fdm_gradient(data,1) ;
    time_length = size(data,1) ;#for time averaging
    
     #construct the correlation matrices
    C0 = (transpose(data) * data )/ time_length ;
    C1 = (transpose(derivative_data) * data )/ time_length ;
    C2 = (transpose(derivative_data) * derivative_data )/ time_length ;
    
     #solve the generalized eigenproblem
    eigenvalues, eigenvectors = eigen(((C1*inv(C0))*transpose(C1))*C2) ;
    eigenvectors = eigenvectors[:,vec(eigenvalues .> eig_thresold) .& vec(eigenvalues .< 1.0)] ;
    if size(eigenvectors,2) > 0
        C3 = inv(C1) * C2 ;
        proj_mat = hcat(eigenvectors,mapslices(x -> C3*x,eigenvectors,dims=[1]))
    else
        throw(DomainError("No generalized eigenvalue fulfills threshold!"))
    end    
        
    
     return proj_mat, data * proj_mat   
end