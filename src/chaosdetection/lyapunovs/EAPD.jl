export ensemble_averaged_pairwise_distance,lyapunov_instant

"""
    ensemble_averaged_pairwise_distance(ds, init_states::StateSpaceSet, T, pidx;kwargs...) -> ρ,t

Calculate the ensemble-averaged pairwise distance function `ρ` for non-autonomous dynamical systems 
with a time-dependent parameter, using the metod described by [^Jánosi,Tél2024]. Time-dependence is assumed to be a linear drift. The rate of change
of the parameter needs to be stored in the parameter container of the system `p = current_parameters(ds)`,
at the index `pidx`. In case of autonomous systems (with no drift), `pidx` can be set to any index as a dummy.
To every member of the ensemble `init_states`, a perturbed initial condition is assigned.
`ρ(t)` is the natural log of state space distance between the original and perturbed states averaged 
over all pairs, calculated for all time steps up to `T`. 


## Keyword arguments

* `initial_params = deepcopy(current_parameters(ds))`: initial parameters 
* `Ttr = 0`: transient time used to evolve initial states to reach 
    initial autonomous attractor (without drift)
* `perturbation = perturbation_normal`: if given, it should be a function `perturbation(ds,ϵ)`,
   which outputs perturbed state vector of `ds` (preferrably `SVector`). If not given, a normally distributed 
   random perturbation with norm `ϵ` is added.
*  `Δt = 1`: step size 
*  `ϵ = sqrt(dimension(ds))*1e-10`: initial distance between pairs of original and perturbed initial conditions 


## Description
In non-autonomous systems with parameter drift, long time averages are less useful to assess chaoticity.
Thus, quantities using time averages are rather calculated using ensemble averages. Here, a new 
quantity called the Ensemble-averaged pairwise distance (EAPD) is used to measure chaoticity of 
the snapshot attractor/ state space object traced out by the ensemble [^Jánosi, Tél].

To any member of the original ensemble (`init_states`) a close neighbour (test) is added at an initial distance `ϵ`. Quantity `d(t)` is the 
state space distance between a test particle and an ensemble member at time t .
If `init_states` are randomly initialized (far from the attractor at the initial parameter), and there's no transient, 
the first few time steps cannot be used to calculate any reliable averages.
The function of the EAPD `ρ(t)` is defined as the average logarithmic distance between original and 
perturbed initial conditions at every time step: `ρ(t) = ⟨ln d(t)⟩`

An analog of the classical largest Lyapunov exponent can be extracted from the 
EAPD function `ρ`: the local slope can be considered an instantaneous Lyapunov exponent.

## Example
Example of a rate-dependent (linearly drifting parameter) system

```julia
#r parameter is replaced by r(n) = r0 + R*n drift term
function drifting_logistic(x,p,n)
    r0,R = p
    return SVector( (r0 + R*n)*x[1]*(1-x[1]) )
end

r0 = 3.8 #inital parameter
R = 0.001 #rate parameter
p = [r0,R] # pidx = 2

init_states = StateSpaceSet(rand(1000)) #initial states of the ensemble 
ds = DeterministicIteratedMap(drifting_logistic, [0.1], p)
ρ,times = ensemble_averaged_pairwise_distance(ds,init_states,100,2;Ttr=1000)
```

[^Jánosi,Tél2024]: Dániel Jánosi, Tamás Tél, Phys. Rep. **1092**, pp 1-64 (2024)

"""
function ensemble_averaged_pairwise_distance(ds,init_states::StateSpaceSet,T,pidx;
    initial_params = deepcopy(current_parameters(ds)),Ttr=0,perturbation=perturbation_normal,Δt = 1,ϵ=sqrt(dimension(ds))*1e-10)

	set_parameters!(ds,initial_params)
    original_rate = current_parameter(ds, pidx)
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
    set_parameter!(pds,pidx,0.0) 

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
    set_parameter!(pds,pidx,original_rate) 

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

function perturbation_normal(ds,ϵ)
    D, T = dimension(ds), eltype(ds)
    p0 = randn(SVector{D, T})
    p0 = ϵ * p0 / norm(p0)  
end

"""
    lyapunov_instant(ρ,times;interval=1:length(times)) -> λ(t)

Calculates the instantaneous Lyapunov exponent by taking the slope of 
the ensemble-averaged pairwise distance function `ρ` wrt. to the saved time points `times` in `interval`.

"""
function lyapunov_instant(ρ,times;interval=eachindex(times))
    _,s = linreg(times[interval], ρ[interval]) #return estimated slope  
    return s
end