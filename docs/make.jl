cd(@__DIR__)

import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/apply_style.jl",
    joinpath(@__DIR__, "apply_style.jl")
)
include("apply_style.jl")

using ChaosTools

CHAOSTOOLS_PAGES = [
    "index.md",
    "orbitdiagram.md",
    "lyapunovs.md",
    "chaos_detection.md",
    "dimreduction.md",
    "periodicity.md",
    "rareevents.md",
]

makedocs(
    modules = [ChaosTools, DynamicalSystemsBase],
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        collapselevel = 3,
    ),
    sitename = "ChaosTools.jl",
    authors = "George Datseris",
    pages = CHAOSTOOLS_PAGES,
    doctest = false,
    draft = false,
)

if CI
    deploydocs(
        repo = "github.com/JuliaDynamics/ChaosTools.jl.git",
        target = "build",
        push_preview = true
    )
end
