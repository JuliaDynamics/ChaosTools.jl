---
title: 'ChaosTools.jl: Tools for the exploration of chaos and nonlinear dynamics'
tags:
  - chaos
  - physics
  - nonlinear
  - lyapunov
  - entropy
  - dimension
authors:
 - name: George Datseris
   orcid: 0000-0002-6427-2385
   affiliation: 1
affiliations:
 - name: Max Planck Institute for Dynamics and Self-Organization
   index: 1
date: 1 January 2018
bibliography: paper.bib
---

# Introduction

Chaotic systems are everywhere [@Strogatz]; from celestial mechanics to biology to electron transport. Not only they cover many scales, but the phenomena that fall under the scope of "nonlinear dynamics" are multi-faceted [@Faust2015].
This vast extend of chaotic systems requires the use of methods from nonlinear dynamics and chaos theory in many diverse areas of science.

On the other hand, chaotic systems are not analytically solvable, which requires computer hardware & software for their study. The best case scenario then would be the existence of a software package that has implementations for as many algorithms as possible from those required for chaotic systems, have extensive unit tests, performant implementations and be easy-to-use. Unexpectedly however, in striking contrast with the spread of chaos and nonlinear dynamics, there is no such panacea.

# Enter ChaosTools.jl
ChaosTools.jl was created to fill this role. It is a Julia [@Bezanson2017] package that offers functions useful in the study of chaos, nonlinear dynamics
and timeseries analysis.
It is part of [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/latest/) software ecosystem, a bundle of packages aimed at the study of chaotic systems. DynamicalSystems.jl itself is also a part of the [JuliaDynamics](https://github.com/JuliaDynamics) organization, similarly with the package DynamicalBilliards.jl [@Datseris2017].

Our official documentation is hosted [here](https://juliadynamics.github.io/DynamicalSystems.jl/latest/).

## ChaosTools.jl Goals
Our goals with this package can be summarized in the following three:

1. Be concise, intuitive, and general. All functions we offer work just as well with any system, whether it is a simple continuous chaotic system, like the Lorenz attractor [@Lorenz1963], or a high dimensional discrete map like coupled standard maps [@Kantz1988].
2. Be accurate, reliable and performant.
3. Be transparent with respect to what is happening "under the hood", i.e. be clear about exactly what each function call does. We take care of this aspect in many ways; by being well-documented, giving references to scientific papers and having clear source code.

## Features

* Intuitive and general system definition.
* Dedicated interface for datasets, including IO.
* Delay coordinates embedding interface.

The above features are not directly part of the source code of ChaosTools.jl.
Instead they come from [DynamicalSystemsBase.jl](https://github.com/JuliaDynamics/DynamicalSystemsBase.jl)  (which belongs to the same ecosystem) and are re-exported by ChaosTools.jl because
they are crucial dependencies for it.

The following however are indeed part of the source code, as of version v0.2.0:

* Poincaré surface of sections.
* Orbit diagrams (also called bifurcation diagrams).
* Automated production of orbit diagrams for continuous systems.
* Maximum Lyapunov exponent.
* Spectrum of Lyapunov exponents.
* Generalized entropies.
* Generalized dimensions and automated procedure of deducing them.
* Neighborhood estimation of points in a dataset.
* Numerical (maximum) Lyapunov exponent of a timeseries.
* Finding fixed points of any map of any order.
* Detecting and distinguishing chaos using the GALI method.

We advise the reader to visit the latest [documentation page](https://juliadynamics.github.io/DynamicalSystems.jl/latest/) because new methods are constantly enriching ChaosTools.jl.

## Similar existing software
We would now like to mention three other software packages that offer similar functionality to ours. We are only considering open-sourced packages in this section.

The first, [TSTOOL](http://www.physik3.gwdg.de/tstool/index.html), is aimed at
nonlinear time series analysis and is implemented in MATLAB (which unfortunately is paid-license language) with a partial backend of C++. Features of TSTOOLS that are not currently offered by ChaosTools.jl are surrogate timeseries and estimating dimensions for delay coordinates embedding. TSTOOL operates on data, and thus any data can easily be loaded through the provided interface, but there is no definition of equations of motion. This has the result that all methods contained are purely numerical and cannot take advantage of known equations of motion.

The second, E&F chaos [@Diks2008], is implemented in LUA with a partial C/Pascal backend
and is aimed at nonlinear dynamics in economics and finance. Features that E&F chaos has that we do not offer are basin boundary plots, cobwebs and parameter basins.
E&F chaos is the only software mentioned here that allows definition of new systems through equations of motion, although only through external text files and not interactively.

LP-VIcode [@Carpintero2014] is a suite devoted solely for computing variational indicators of chaos and is written in FORTRAN77. ChaosTools.jl offers only the latest  indicator from all the ones available in LP-VIcode, namely GALI [@Skokos2007]. In addition, LP-VIcode places the severe constrain that all systems must not only be Hamiltonian, but must also have parabolic kinetic energy term. This leads to having to completely re-write FORTRAN77 source code almost from scratch to add any new system to the package.

## ChaosTools.jl advantages vs other packages
* It is written in purely in Julia [@Bezanson2017].
  * Julia is (currently) the only open sourced & dynamic language that has performance equal to C/Fortran, allowing interactivity without adding computational costs.
* Offers the widest range of methods and does not place limitations on them.
* Transparent and small source code.
* It is concise, intuitive and general: all functions work just as well
  with any defined dynamical system.
* Extendable; adding completely new systems or algorithms requires small effort.
* Well-documented.
* Actively maintained and constantly growing.
* Hosted on GitHub, making interaction of users and developers easy and straightforward.

# Examples
In the following examples we want to demonstrate how easy and general it is to use ChaosTools.jl. In the first example will show how one can find the maximum Lyapunov exponent and GALI for a continuous system, while the second will show how to use delay coordinates embedding to calculate the attractor dimension of a timeseries.
Both examples are benchmarked with a laptop with Intel Core i7-4710MQ CPU @ 2.50GHz, 16GB RAM, and 64-bit Windows 10 operating system.
## Lyapunov & GALI of a continuous system
```julia
# Pkg.add("ChaosTools")
using ChaosTools
# Define Hénon-Heiles system
function eom_henon(du::EomVector, u::EomVector)
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -u[1] - 2*u[1]*u[2]
    du[4] = -u[2] - u[1]^2 + u[2]^2
    return nothing
end
# A function for the Jacobian is useful but not necessary;
# If it is not given, automatic differentiation is used
function jacobian_henon(J::EomMatrix, u::EomVector)
    J[3,1] = -1 - 2*u[2]; J[3,2] = -2*u[1]
    J[4,1] = -2*u[1]; J[4,2] =  -1 + 2*u[2]
    return nothing
end
# typical chaotic initial condition:
u0=[0, -0.25, 0.42081, 0]
# Initialize Jacobian matrix
J = zeros(eltype(u0), 4, 4)
J[1,:] = [0,    0,     1,    0]
J[2,:] = [0,    0,     0,    1]
J[3,:] = [ -1 - 2*u0[2],   -2*u0[1],   0,   0]
J[4,:] = [-2*u0[1],  -1 + 2*u0[2],  0,   0]
# Create the dynamical system structure
# Simply pass equations of motion and Jacobian to ContinuousDS
henon = ContinuousDS(u0, eom_henon, jacobian_henon, J)

# Calculate the MLE, by rescaling every 1.0 time
ml = lyapunov(henon, 1000.0; dt = 1.0)

println("MLE ≈ $(round(ml, 5))")
# Benchmark
@time lyapunov(henon, 1000.0; dt = 1.0);
```
The code prints `LE ≈ 0.05152` (specific numbers change from machine to machine) and the benchmark clocks at ~0.039704 seconds.

For GALI of rank `k` we do
```julia
k = 4 # rank of desired gali
total_t = 1000.0
# Get gali and accompanying time vector:
g, t = gali(henon, k, total_t; threshold = 1e-12)
println("GALI4 reached 1e-12 at time $(t[end]), with value $(g[end])")

# Benchmark
@time gali(henon, k, total_t; threshold = 1e-12);
```
The code prints `GALI4 reached 1e-12 at time 102.0, with value 8.191377735674049e-13`
which again changes from machine to machine, while the benchmark clocks at ~0.002683 seconds.

We want to stress that both functions `gali`, `lyapunov` (and in fact, all functions offered by ChaosTools.jl) work with any system type, continuous or discrete. See the documentation page for more.

## Information Dimension from Delay Coordinates Embedding
Here we show how one can handle numerical data with ChaosTools.jl. Because this is a publication and loading data from disk is not possible, we will first produce some timeseries of the Hénon map [@Henon1976].

```julia
ds = Systems.henon() # load one of the predefined systems

# Get a trajectory of the system:
traj = trajectory(ds, 100000)

# A timeseries is univariate:
ts = traj[:, 1]

# Now perform delay coordinates embedding of dimension 2 and delay 2:
R = Reconstruction(ts, 2, 2)

# Now e.g. calculate the Information dimension
id = information_dim(R)

# which is equivalent with:
sizes = estimate_boxsizes(R)
dd = zeros(sizes)
  for i in 1:length(sizes)
      dd[i] = genentropy(1, sizes[i], data)
  end
id = linear_region(-log.(sizes), dd)[2]

# For reference, we can compute the information dimension of the
# Henon attractor directly, because we have a trajectory
id_direct = information_dim(traj)

println("Dimensions: $(round(id, 4)), $(round(id_direct, 4))")

# Benchmark:
@time Reconstruction(ts, 2, 2);
@time information_dim(traj);
```
The code prints: `Dimensions: 1.233, 1.2292`. Performing the reconstruction clocks at
0.000822 seconds and calculating the dimension of the dataset clocks at 0.603827 seconds. We note that the function `information_dim` (and other similar ones) computes
a lot of automated steps by measuring entropies at many different partition sizes. That is the reason for being "slow".

# References
