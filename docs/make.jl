cd(@__DIR__)
using ChaosTools, DynamicalSystemsBase, Neighborhood

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

build_docs_with_style(pages, ChaosTools, DynamicalSystemsBase, Neighborhood)