using BatchedMetrics
using Documenter

DocMeta.setdocmeta!(BatchedMetrics, :DocTestSetup, :(using BatchedMetrics); recursive=true)

makedocs(;
    modules=[BatchedMetrics],
    authors="Joshua Billson",
    sitename="BatchedMetrics.jl",
    format=Documenter.HTML(;
        canonical="https://JoshuaBillson.github.io/BatchedMetrics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaBillson/BatchedMetrics.jl",
    devbranch="main",
)
