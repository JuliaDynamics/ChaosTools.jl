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

Chaotic systems are everywhere [@Strogatz]; from celestial mechanics to biology to electron transport. Not only they cover many scales, but the phenomena that fall under the scope "nonlinear dynamics" are multi-faceted [@Faust2015].
This vast extend of chaotic systems requires the use of methods from nonlinear dynamics and chaos theory in many diverse areas of science.

On the other hand, chaotic systems are not "solvable", which requires computer hardware & software for their study. The best case scenario then would be the existence of a software package that has implementations for as many algorithms as possible of those  required for chaotic systems, have extensive unit tests, performant implementations and be easy-to-use. Unexpectedly however, in striking contrast with the spread of chaos and nonlinear dynamics, there is no such *panacea*.

# Enter ChaosTools.jl
ChaosTools.jl was created to fill this role. It is a Julia package that offers functions for quantities useful in the study of chaos and nonlinear dynamics.

Part of DynamicalSystems.jl software ecosystem.


## ChaosTools.jl Goals
Our goals with this package can be summarized in the following three:

1. Be concise, intuitive, and general. All functions we offer work just as well with any system, whether it is a simple continuous chaotic system, like the Lorenz attractor [@Lorenz1963], or a high dimensional discrete map like coupled standard maps [@Kantz1988].
2. Be accurate, reliable and performant.
3. Be completely transparent with respect to what is happening "under the hood", i.e. be clear about exactly what each function call does. We take care of this aspect in many ways; by being well-documented, giving references to scientific papers and having clear source code.

## Features

* Intuitive and general system definition. The interface that defines a dynamical system does not depend on the system itself; any dynamical system can be used with ChaosTools.jl and quickly defined in a couple lines of code (see the example).
* Dedicated interface for datasets, including IO.
* Delay coordinates embedding interface.

* Poincare surface of sections.
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


ChaosTools.jl uses [DynamicalSystemsBase.jl]().

## Other existing software

## ChaosTools.jl advantages vs other packages
* It is written in purely in Julia [cite].
  * Julia is (currently) the only open sourced & dynamic language that has performance equal to C/Fortran, allowing interactivity without adding computational costs.
* Offers the widest range of methods. No other packages has that broad spectrum.
* Completely transparent source code; On average each complete algorithm of our package requires 30 lines of code, making reading the source code very easy.
* It is concise, intuitive, performant and general.
* Extendable; adding completely new systems or functions requires small effort.
* Intuitive usage and clear documentation.
* Actively maintained and constantly growing.



# Example
lala

# References
