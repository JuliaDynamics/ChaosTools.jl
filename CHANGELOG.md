# main

# v3.0

Major release part of the DynamicalSystems.jl v3.
The DynamicalSystems.jl v3 changelog summarizes the highlights. Here we will list all changes to _this specific package_.

## Breaking
- All `diffeq` keywords to all functions have been removed as per update to DynamicalSystems v3.0. Arguments to DifferentialEquations.jl solvers are now given in system construction when making a `CoupledODEs` (previously `ContinuousDynamicalSystem`).
- The dependency list of ChaosTools.jl has been reduced by half; many things that used to be exported from here are no longer (see refactoring).
- Low-level call signatures have been adjusted to fit DynamicalSystems.jl v3.0

## Refactoring/removal
- All functionality related to attractors and basins of attraction (e.g., `AttractorMapper`, `basins_fractions`) has been moved to a new package Attractors.jl. Notice that functions such as `fixedpoints` and `periodicorbits` stay in ChaosTools.jl because they aren't only about attractors; they also find unstable fixed points and/or periodic orbits.
- All functionality related to Poincare maps has been moved to DynamicalSystemsBase.jl
- The state space samplers have been moved to StateSpaceSets.jl
- All fractal dimension related functionality has been moved to a new package FractalDimensions.jl. This includes the functionality for finding linear regions and fitting them.

## Rare events
- New dedicated folder structure and functionality targeting rare events in ChaosTools.jl. If it becomes extensive, it can be split off to a new package.
- Source code for `exit_entry_times` has been completely overhauled and is now
  much much clearer.
- Algorithm for `exit_entry_times` for continuous systems has been re-written from
  scratch, and is now much more accurate, and much faster. Two ways are provided
  for finding crossings: linear intersections and high order interpolation via optimization.
- `mean_return_times` is now just a wrapper function.
- New function `first_return_times` for efficiently computing only the first time to return to sets.

## Other enhancements
- `orbitdiagram` is a completely generic function that works for any kind of `DynamicalSystem`. `produce_orbitdiagram` is deprecated as it is now practically useless.
- ChaosTools.jl now has its own documentation as per DynamicalSystems.jl v3.0.
- More examples have been placed that also better highlight how to parallelize.
- Increased the default amount of `c` in `testchaos01`.
- Tests have been overhauled and never use predefined systems (which is a really bad practice when it comes to testing)

# 2.9
* Improved the `AttractorsViaFeaturizing` algorithm by improving the method for finding the optimal radius used in the clustering. This consisted in (i) maximizing the average silhouette values, instead of minimum (slight improvement), (ii) min-max rescaling the features for the clustering (big improvement); (iii) adding an alternative method ,called elbow method, that is faster but worse at clustering.
* Changed `attractor_mapping_tests.jl` to deal better with the Featurizing method.

# 2.8
* Brand new `AttractorMapper` infrastructure. It is a generic framework for mapping initial conditions to attractors and hence calculating basins of attraction and related quantities. Existing originally disparate functionality has been brought together under this framework.
* The old `basins_of_attraction` function has been completely deprecated in favor of using the version `basins_of_attraction(mapper::AttractorMapper, grid)`, which utilizes the new `AttractorMapper` interface and is more intuitive and generalizable.
* New method for mapping initial conditions to known attractors using proximity.
* Old functions related to clustering via featurizing have been removed in favor of just using `AttractorsViaFeaturizing`.
* **BREAKING** Function `basin_fractions_clustering` is removed.
* `basin_fractions` is deprecated in favor of `basins_fractions`.

# 2.7
* New function `basin_fractions_clustering` that estimates fractions of basins of attraction via a random sampling and clustering technique.
* New function `statespace_sampler` that conveniently creates functions that sample state space regions.

# 2.6
* New `yin` function that applies the YIN algorithm to detect a signal's fundamental frequency. Also added it as a possible method in `estimate_period`.

# 2.5.3
* Updated `lyapunovspectrum_convergence` for continuous systems to be similar to `lyapunovspectrum`, with the more performant `_buffered_qr` and allowing passing a `DynamicalSystem` instead of the integrator directly.
* Added `lyapunovspectrum_convergence` for discrete systems.

# 2.5
* New `fixedpoints` function that finds fixed points, and their stability, for either continuous or discrete dynamical systems.
* Automatic `Δt` estimation for `basins_of_attraction`.

# 2.4
* `linear_region` now detects a saturation regime also at the start of the curve y(x). Furthermore a keyword `sat` now decides what the saturation regime is.
# 2.3
* Correlation sum functions now have a `show_progress = false` keyword. It shows a progress bar!
* `linear_region` and `estimate_boxsizes` have `warning = true` keyword.
* The `estimate_boxsizes` function now has keywords `autoexpand = true, we = w, ze = z`.

# 2.2
* Functions `estimate_r0_buenoorovio, estimate_r0_theiler` now return `r0, ε0` with `ε0` the minimum inter-point distance. This increases performance of other methods by reducing duplicate computations.
* Fixed bug in `estimate_r0_buenoorovio` that didn't consider min pairwise distance = 0.
* Exported the already implemented `estimate_r0_theiler`.
* For accuracy improvement, the mean of maximal lengths along each dimension is used in `estimate_boxsizes`. Before it was the maximum of maxima.

# 2.1
* Various improvements to the boxed correlation sum method. Now it also uses an automatic prism dimension.

# 2.0
* The keyword `dt` that was used to denote a chunk of time in many functions that
  a `DynamicalSystem` has been changed to `Δt` due to conflicts with DifferentialEquations.jl solver options. This change is breaking and cannot be warned or deprecated. Functions affected: `lyapunov, lyapunovspectrum, gali, expansionentropy, orbitdiagram`
* All deprecations have been removed, switch to previous stable version to see any.

# 1.34
* New option in `basins_of_attraction` that allows refining already found basins.

# 1.33
* `lyapunovspectrum` now has a progress meter option.

# 1.32
* New functions for examining the fractal nature of basin boundaries: `basins_fractal_dimension, basin_entropy, basins_fractal_test`.

# 1.31
* New, general function `basins_of_attraction` that replaces the existing `basins_2D` and `basins_general` and can produce arbitrary-dimensional basins of attraction.

# 1.30
* New function `local_growth_rates`
* New function `match_attractors!`

# 1.29
* Performance improvements for basins of attraction
* New function `uncertainty_exponent`
* New function `basin_fractions`
* New function `tipping_probabilities`

# 1.28
* New functions `basins_2D` and `basins_general` that efficiently compute basins of attraction on a plane!

# 1.27
* new function `correlationsum_fixedmass` implements a fixed mass algorithm for the correlationsum given by Grassberger in 1988.

# 1.26
* new function `dyca()` to perform Dynamical Component Analysis

# 1.25
* new function `poincaremap` for iterating over the Poincare map step by step.
* `mean_return_times` has been improved for continuous systems and now also allows `diffeq...` keyword propagation. The keyword `m` is also deprecated in favor of `dmin`.

# 1.24
* Theiler window is now possible in `boxed_correlationsum`.

# 1.23
* Various improvements on the functionality of `estamte_boxsizes` and `linear_region`.

# 1.22
* Orbit diagrams now use the previous state at each new parameter, providing faster convergence to attractor for smaller `Ttr`. The previous option is still available by passing explicitly `u0 = get_state(ds)`.

# 1.21
* Using keyword `α` is deprecated in favor of `q` in all entropy-related discussions (`q` is more common in the literature).
* Added `boxed_correlationsum` and `boxed_correlationdim` that distribute the data into boxes before calculating the correlationsum.
* Added `estimate_r0_buenoorovio` to find the optimal boxsize for the former two functions.

# 1.20
* Keyword `u0` is now valid for `lyapunov`.

# 1.19
* A lot of functions have been deprecated in favor of the new syntax that uses Entropies.jl: `non0hist, binhist, genentropy`.
* `information_dim, capacity_dim, boxcounting_dim` are deprecated.
* Permutation entropy has been re-written from scratch to use the Entropies.jl version. This drops the (completely unnecessary) argument `interval`, however the old method is available as `ChaosTools.permentropy_old`. It will be removed completely in version 2.0.
* `correlationsum` now features the keyword `q` to calculate the q-order correlationsum.
* Add fractal dimension estimation method by Molteno et al `molteno_dim`.
* `lyapunovs` is deprecated in favor of `lyapunovspectrum`.

# 1.18
* `poincaresos` function now also works with input `Dataset` (and does linear interpolation between points sandwiching the hyperplane)

# 1.17
* `transit_time_statistics` deprecated in favor of `exit_entry_times`.
* Added `mean_return_times` function for discrete systems.
* Added `mean_return_times` function for continuous systems.

# 1.16
* `takens_best_estimate` now returns three arguments, the estimate plus the upper and lower 95%-confidence limits.

# 1.15
* New function `transit_time_statistics` that allows computing return times and transit times to subsets of the state space. (Currently for discrete systems only)
* Moved support to Julia 1.5+.

# 1.14
* `orbitdiagram` now allows only collecting states within user-provided limits.
# 1.13
* Takens' best estimate method for estimating the correlation dimension is available as `takens_best_estimate`.

# 1.12
* `binhist` method that returns data histogram and bin edges
* further optimization of `correlationsum` for vector `εs`.

# 1.11
* Theiler's correction is now possible in estimating the correlation sum (as a result, `norm` is now a keyword).

# 1.10
* Added method for estimating correlation dimension (and as a pre-requisite, also the correlation sum) based on the method of Grassberger-Proccacia.
* Added "kernel density nearest neighbor" estimator for probabilistic description of a dataset.

# 1.9
* The expansion entropy from Hunt and Ott (defining chaos 2015) is now included as `expansionentropy`! Thanks and welcome to our new contributor @yuxiliu1995 !

# 1.8
* new function `testchaos01` which implements the so called "0-1" test for chaos, that can test if a numeric timeseries is chaotic or not.

# 1.7
* New function `estimate_period` that attempts to estimate the period of a signal using the following methods:
  * The autocorrelation function (when it comes close to 1 again)

# 1.6
* Implementation of  Wernecke, H., Sándor, B. & Gros, C.
      *How to test for partially predictable chaos*. [Scientific Reports **7**, (2017)](https://www.nature.com/articles/s41598-017-01083-x). Thanks and welcome to our new contributor @efosong ! Implemented with the function `predictability`.

# 1.5
* Updated everything to new SciMLBase 5.0 and the new default integrator (`SimpleATsit5`)
* Simplified low-level call signature for `poincaresos`.
* Small documentation improvements throughout.


# 1.4
* `estimate_boxsizes` now slightly expands borders if upper and lower do not have 2 orders of magnitude difference.

# 1.3
* increased performance of `non0hist`
* Documentation improvements

# 1.2.1
* Actually fix the issue with `poincaresos` and states starting on the plane
* The default of `direction` had to change to `-1`. We realized that having `1` was wrong and unintuitive with how the PSOS data is returned and plotted and thus we consider this a bugfix.

# 1.2
* allow `u0` as keyword in `poincaresos`

# 1.1
* Fixed a bug in `poincaresos` when the initial condition was on the plane.
* Allowed `poinracesos` and `produce_orbitdiagram` to also configure the keywords of the root finding.

# 1.0 ≡ 0.13
First major release.

# 0.13
* Renamed all low-level methods that were exposed as part of the API to not start with an underscore anymore.
* Updated everything (REQUIRES etc) to julia 1.0.
* Added convergence return function for lyapunov exponents.

# 0.12
## BREAKING
* Dropped support for all julia < 0.7
* Method for estimating reconstruction dimension now estimate temporal neighbor number to account for `reconstruct`.
* Reworked a bit how `orbitdiagram`s and co. behave. Now if given multiple states it is assumed that each state is for a different parameter. Please read the documentation strings of the functions.

## New Features
* Brand new algorithm that computes poincare sections. Now uses interpolation of DiffEq and root finding of Roots.jl. This gives _at least an order of magnitude speedup_ in `produce_orbitdiagram` and makes the source code massively more clear!!!
* It is now possible to choose which variables to save in both discrete and continuous orbit diagrams.
* Added method to compute mutual information, from  A. Kraskov *et al.*, [Phys. Rev. E **69**, pp 066138 (2004)]
* Added method in finding delay time that uses mutual information. At the moment this method is vastly inferior to all others both in speed and in actual results.
* `lyapunovspectrum` is 1 to 2 orders of magnitude faster.


# 0.11
* Changed `gali` call signature to be the same as `lyapunovspectrum`.
* Bugfixed 1D lyapunov computation
* Bugfixed `set_deviations!` for continuous systems
* Updated for `DynamicalSystemsBase` 0.10 (using `get_state` etc.)
* Significantly reduced code repetition in the source.


# v0.9
* Upgraded `poincaresos` to work for any arbitrary hyperplane. The call signature had
  to change to make it work.

# v0.8.2
* Docstring typos fixed
* Removed aliases from `genentropy` .

# v0.8
* Methods for estimating Reconstruction parameters moved here.
* Added Cao's method for estimating dimension.
* Added methods to find delay time:
  * First minimum
  * First zero
  * Exp. decay
  * Mutual information (NOT exported, needs to be tested and documented)
* Generalized `numericallyapunov` to allow choosing Theiler window
  (and adopted it to new `neighborhood` function).
* Keyword `method` in `numericallyapunov` changed to

# v0.7
## Breaking
* The code for estimating reconstruction parameters was moved to `DynamicalSystemsBase`.
## Non-Breaking
* Theiler window is now configurable in `numericallyapunov`.
* Updated source and documentation for new `Reconstruction`.
* Type-stability tests and improvements.

# v0.6
This version updates all internals to be on par with `DynamicalSystemsBase` 0.6

*All* internal implementations of methods that use a `DynamicalSystem` have been
reworked almost from the ground up.

This lead to (only minor) changes to some of the high level interfaces. Please see
the documentation strings of each function before using it.

# v0.5.1
## Non-breaking
* Bugfix: gali was invasive in the diff_eq_kwargs

# v0.5.0
## Breaking
* Updated everything to be on par with the changes of DynamicalSystemsBase v0.5

## Non-breaking
* Minor bug fixes resulting from typos.
* New method for permutation entropy: `permentropy` !

# v0.4.3
* Added Broomhead King function
* Estimate_delay and other functions about reconstructions are now in this repo.

# v0.4.1
* Many bugfixes relating evolution of continuous systems and correct
  callback propagation.
# v0.4.0
## Breaking
* Updated everything to be on par with Base 0.3.1
* Added more tests
* Propagate callbacks etc in orbit diagrams


# v0.3.0
## BREAKING
* Overhauled the way `estimate_boxsizes` works.

# v0.2.0
* added method for Poincare surface of section
* added orbit diagram for maps
* added production of orbit diagram for flows through Poincare surface of sections

# v0.1.0
Initial release, see: https://juliadynamics.github.io/DynamicalSystems.jl/v1.0.0/
for the features up to this point.
