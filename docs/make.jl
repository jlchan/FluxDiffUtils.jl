push!(LOAD_PATH,"src/") # necessary for unregistered packages

using Documenter
using FluxDiffUtils

makedocs(
    sitename = "FluxDiffUtils.jl",
    repo = "https://github.com/jlchan/FluxDiffUtils.jl",
    modules=[FluxDiffUtils],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/jlchan/FluxDiffUtils.jl",
)
