using LinearAlgebra

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

function DyCA(data,eig_thresold=0.98)

    derivative_data = matrix_fdm_gradient(data,2) ;
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