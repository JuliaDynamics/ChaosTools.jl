export ensemble_averaged_pairwise_distance,lyapunov_instant

"""
    lyapunov_instant(ρ,times;interval=1:length(times)) -> λ(t)

Convenience function that calculates the instantaneous Lyapunov exponent by taking the slope of 
the ensemble-averaged pairwise distance function `ρ` wrt. to the saved time points `times` in `interval`.
"""
function lyapunov_instant(ρ,times;interval=1:length(times))
    _,s = linreg(times[interval], ρ[interval]) #return estimated slope  
    return s
end

"""
    ensemble_averaged_pairwise_distance(ds,init_states::StateSpaceSet,T;kwargs...) -> ρ,t

Calculate the ensemble-averaged pairwise distance function `ρ` for non-autonomous dynamical systems 
with a time-dependent parameter. Time-dependence is assumed to be linear (sliding). To every member 
of the ensemble `init_states`, a perturbed initial condition is assigned. `ρ(t)` is the natural log 
of phase space distance between the original and perturbed states averaged over all pairs, calculated 
for all time steps up to `T`. 

This function implements the method described in 
https://doi.org/10.1016/j.physrep.2024.09.003.

## Keyword arguments

* `sliding_param_rate_index = 0`: index of the parameter that gives the rate of change of the sliding parameter
* `initial_params = deepcopy(current_parameters(ds))`: initial parameters 
* `Ttr = 0`: transient time used to evolve initial states to reach 
    initial autonomous attractor (without sliding)
* `perturbation = perturbation_uniform`: if given, it should be a function `perturbation(ds,ϵ)`,
   which outputs perturbed state vector of `ds` (preferrably `SVector`). If not given, a normally distributed 
   random perturbation with norm `ϵ` is added.
*  `Δt = 1`: step size 
*  `ϵ = sqrt(dimension(ds))*1e-10`: initial distance between pairs of original and perturbed initial conditions 
"""
function ensemble_averaged_pairwise_distance(ds,init_states::StateSpaceSet,T;sliding_param_rate_index=0,
    initial_params = deepcopy(current_parameters(ds)),Ttr=0,perturbation=perturbation_uniform,Δt = 1,ϵ=sqrt(dimension(ds))*1e-10)

	set_parameters!(ds,initial_params)
    N = length(init_states)
    d = dimension(ds)
    dimension(ds) != d && throw(AssertionError("Dimension of `ds` doesn't match dimension of states in init_states!"))
    
    nt = length(0:Δt:T) #number of time steps
    ρ = zeros(nt) #store ρ(t)
    times = zeros(nt) #store t
    
    #duplicate every state
    #(add test particle to every ensemble member)
    init_states_plus_copies = StateSpaceSet(vcat(init_states,init_states))

    #create a pds for the ensemble
    #pds is a ParallelDynamicalSystem
    pds = ParallelDynamicalSystem(ds,init_states_plus_copies)

	#set to non-drifting for initial ensemble
    sliding_param_rate_index != 0 && set_parameter!(pds,sliding_param_rate_index,0.0) 

    #step system pds to reach attractor(non-drifting)
    #system starts to drift at t0=0.0
    for _ in 0:Δt:Ttr
        step!(pds,Δt,true)
    end
    
    #rescale test states 
    #add perturbation to test states
    for i in 1:N 
        state_i = current_state(pds,i)
        perturbed_state_i = state_i .+ perturbation(ds,ϵ)
        #set_state!(pds.systems[N+i],perturbed_state_i)
        set_state!(pds,perturbed_state_i,N+i)
    end

	#set to drifting for initial ensemble
    set_parameters!(pds,initial_params) 

    #set back time to t0 = 0
    reinit!(pds,current_states(pds))

    #calculate EAPD for each time step
    ensemble_averaged_pairwise_distance!(ρ,times,pds,T,Δt)
    return ρ,times

end

#calc distance for every time step until T
function ensemble_averaged_pairwise_distance!(ρ,times,pds,T,Δt)
    for (i,t) in enumerate(0:Δt:T)
        ρ[i] = ensemble_averaged_pairwise_distance(pds)
        times[i] = current_time(pds)
        step!(pds,Δt,true)
    end
end

#calc distance for current states of pds
function ensemble_averaged_pairwise_distance(pds)

    states = current_states(pds)
    N = Int(length(states)/2)

    #calculate distance averages
    ρ = 0.0
    for i in 1:N
        ρ += log.(norm(states[i] - states[N+i]))
    end
    return ρ/N 

end

function perturbation_uniform(ds,ϵ)
    D, T = dimension(ds), eltype(ds)
    p0 = randn(SVector{D, T})
    p0 = ϵ * p0 / norm(p0)  
end
