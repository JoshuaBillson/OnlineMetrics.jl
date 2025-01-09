using OnlineMetrics
using Documenter

DocMeta.setdocmeta!(OnlineMetrics, :DocTestSetup, :(using OnlineMetrics); recursive=true)

makedocs(;
    modules=[OnlineMetrics],
    authors="Joshua Billson",
    sitename="OnlineMetrics.jl",
    format=Documenter.HTML(;
        canonical="https://JoshuaBillson.github.io/OnlineMetrics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaBillson/OnlineMetrics.jl",
    devbranch="main",
)
