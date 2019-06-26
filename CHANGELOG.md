# 1.7
* New function `estimate_period` that attempts to estimate the period of a signal using the following methods:
  * The autocorrelation function (when it comes close to 1 again)

# 1.6
* Implementation of  Wernecke, H., Sándor, B. & Gros, C.
      *How to test for partially predictable chaos*. [Scientific Reports **7**, (2017)](https://www.nature.com/articles/s41598-017-01083-x). Thanks and welcome to our new contributor @efosong ! Implemented with the function `predictability`.

# 1.5
* Updated everything to new DiffEqBase 5.0 and the new default integrator (`SimpleATsit5`)
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
* `lyapunovs` is 1 to 2 orders of magnitude faster.


# 0.11
* Changed `gali` call signature to be the same as `lyapunovs`.
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
* Minor bug fixes resulting from misstypos.
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
