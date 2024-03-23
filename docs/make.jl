cd(@__DIR__)
using ChaosTools
using ChaosTools.DynamicalSystemsBase
using ChaosTools.Neighborhood

pages = [
    "index.md",
    "orbitdiagram.md",
    "lyapunovs.md",
    "chaos_detection.md",
    "dimreduction.md",
    "periodicity.md",
    "rareevents.md",
]

import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/build_docs_with_style.jl",
    joinpath(@__DIR__, "build_docs_with_style.jl")
)
include("build_docs_with_style.jl")

# TODO: Port all citations to use this:
# using DocumenterCitations

# bib = CitationBibliography(
#     joinpath(@__DIR__, "refs.bib");
#     style=:authoryear
# )

build_docs_with_style(pages, ChaosTools, DynamicalSystemsBase, Neighborhood;
    # bib, # TODO: Enable bib
    # TODO: Fix warnings so that instead of:
    warnonly = true,
    # we can have:
    # warnonly = [:doctest, :missing_docs, :cross_references],
)