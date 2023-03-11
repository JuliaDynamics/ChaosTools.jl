using ChaosTools
using Test
using DynamicalSystemsBase #to get Systems
using DelayEmbeddings #to get columns() and embed()

@testset "DyCA Tests" begin
    println("Running dyca() tests...")

    @testset "Eigenvalues of Roessler system" begin
        # Initial conditions from `dyca()` paper
        eigenthreshold = collect(0.10:0.01:0.995)
        ds = Systems.roessler(ones(3); a = 0.15, b = 0.2, c = 10);
        ts = trajectory(ds, 1000.0; Δt = 0.05);
        for threshold in eigenthreshold
            eigenvalues,_ = dyca(Matrix(ts),threshold)
            @test sum([round(i) for i in eigenvalues]) == 2.0
        end
    end

    @testset "Eigenvalues of Embeded Roessler system" begin
        # Initial conditions from `dyca()` paper
        eigenthreshold = collect(0.90:0.1:0.95)
        ds = Systems.roessler(ones(3); a = 0.15, b = 0.2, c = 10);
        ts = trajectory(ds, 1000.0; Δt = 0.05);
        cols = columns(ts);
        for threshold in eigenthreshold
            for i in 1:3
                Embedded_system = Matrix(embed(cols[i], 25, 1));
                eigenvals,_ = dyca(Embedded_system, threshold)
                # eigenvalues satisfy `dyca` condition and have imaginary part = 0.0
                wanted_eigenvalues = eigenvals[vec(0.999 .< abs.(eigenvals) .< 1.0) .& vec(imag(eigenvals) .== 0)]
                # due to floating point errors more than 2 eigenvalues might satisfy the above condition
                # might be better to use `length(psum) in [1,2,3]`
                @test length(wanted_eigenvalues) in 0:8
            end
        end
    end
end

#Test using EEG seizure data, based on Datseris's python notebook available at https://gist.github.com/Datseris/8121e5019071fe9e3cb9b1a0811ac26a
# using PyPlot, PyCall
# pyimport("mpl_toolkits.mplot3d").Axes3D
# using DataFrames, CSV
# eeg_data = DataFrame(CSV.File("input/seizure.csv")) #data available in https://github.com/JuliaDynamics/ChaosTools.jl/files/3065748/dyca.zip

# figure()
# for i=2:size(eeg_data,2)
# 	plot( eeg_data[:,i], "-")
# end
# savefig("eeg_data.png")

# eigenvalues, proj_mat, proj_eeg = dyca(Matrix(eeg_data[:, 2:end]), 0.9; norm_eigenvectors=true)
# #the eigendecomposition function used in the python version normalizes the eigenvectors. To compare the figures more easily, I also normalized them here.

# figure()
# plot(proj_eeg)
# savefig("projected_eeg_data_time_series.png")

# fig = figure(figsize=(15,14))
# ax = fig.add_subplot(2, 2, 1, projection="3d")
# ax.plot(proj_eeg[:,1], proj_eeg[:,2], proj_eeg[:,3])
# ax.view_init(elev=51, azim=86)
# ax.set_xlabel(L"x_1"); ax.set_ylabel(L"x_2"); ax.set_zlabel(L"x_3");
# ax = fig.add_subplot(2, 2, 2, projection="3d")
# ax.plot(proj_eeg[:,1], proj_eeg[:,2], proj_eeg[:,4])
# ax.view_init(elev=37, azim=173)
# ax.set_xlabel(L"x_1"); ax.set_ylabel(L"x_2"); ax.set_zlabel(L"x_3");
# ax = fig.add_subplot(2, 2, 3, projection="3d")
# ax.plot(proj_eeg[:,2], proj_eeg[:,3], proj_eeg[:,4])
# ax.view_init(elev=17, azim=36)
# ax.set_xlabel(L"x_1"); ax.set_ylabel(L"x_2"); ax.set_zlabel(L"x_3");
# ax = fig.add_subplot(2, 2, 4, projection="3d")
# ax.plot(proj_eeg[:,1], proj_eeg[:,3], proj_eeg[:,4])
# ax.view_init(elev=43, azim=148)
# ax.set_xlabel(L"x_1"); ax.set_ylabel(L"x_2"); ax.set_zlabel(L"x_3");
# savefig("projected_eeg_data_phasespace.png")
