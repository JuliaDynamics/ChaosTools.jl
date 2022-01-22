using Statistics, BenchmarkTools
using ChaosTools, DelayEmbeddings, DynamicalSystemsBase

#Magnetic pendulum (3 attractors)
function feature_extraction(t, y)
        return y[end][1:2] #final position 
    end

ds = Systems.magnetic_pendulum()

#parameters
T = 205
Ttr = 200 
Δt = 1.0


#region of interest 
N = 1000; 
min_limits = [-1,-1,-1,-1]; 
max_limits = [1,1,1,1]; 
sampling_method = "uniform"; 

s = ChaosTools.sampler(min_bounds=min_limits, max_bounds=max_limits, method=sampling_method)
ics = Dataset([s() for i=1:N])


fs, class_labels = @benchmark basin_fractions_clustering(ds, feature_extraction, ics; T=T, Ttr=Ttr, Δt=Δt)


#--- Lorenz 84 (3 attractors)
function feature_extraction(t, y)
        return [std(y[:,i]) for i=1:3] #not sure this is a great feature
end

ds = Systems.lorenz84(F = 6.886, G = 1.347, a = 0.255, b = 4.0)

#parameters
T = 2000
Ttr = 500 
Δt = 1.0


#region of interest 
N = 100; 
min_limits = [0,0,0]; 
max_limits = [1,1,1]; 
sampling_method = "uniform"; 

s = ChaosTools.sampler(min_bounds=min_limits, max_bounds=max_limits, method=sampling_method)
ics = Dataset([s() for i=1:N])
fs, class_labels = basin_fractions_clustering(ds, feature_extraction, ics; T=T, Ttr=Ttr, Δt=Δt)


