# ChaosTools.jl

[![docsdev](https://img.shields.io/badge/docs-dev-lightblue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/dev/)
[![docsstable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/stable/)
[![](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--91032--7-purple)](https://link.springer.com/book/10.1007/978-3-030-91032-7)
[![CI](https://github.com/JuliaDynamics/ChaosTools.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/ChaosTools.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/JuliaDynamics/ChaosTools.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDynamics/ChaosTools.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/ChaosTools)](https://pkgs.genieframework.com?packages=ChaosTools)

A Julia module that offers various tools for analysing nonlinear dynamics and chaotic behaviour.
It can be used as a standalone package, or as part of [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystems/dev/).

To install it, run `import Pkg; Pkg.add("ChaosTools")`.

All further information is provided in the documentation, which you can either find online or build locally by running the `docs/make.jl` file.

_ChaosTools.jl is the jack-of-all-trades package of the DynamicalSystems.jl library: methods that are not extensive enough to be a standalone package are added here. You should see the full DynamicalSystems.jl library for other packages that may contain functionality you are looking for but did not find in ChaosTools.jl._