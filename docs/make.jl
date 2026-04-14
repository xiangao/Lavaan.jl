using Documenter, Lavaan

makedocs(
    sitename = "Lavaan.jl",
    modules = [Lavaan],
    pages = [
        "Home" => "index.md",
        "Vignettes" => [
            "Introduction"          => "vignettes/Introduction.md",
            "Model Syntax"          => "vignettes/Model_Syntax.md",
            "Mediation Analysis"    => "vignettes/Mediation_Analysis.md",
            "Ordinal Data"          => "vignettes/Ordinal_Data.md",
            "Multilevel & Crossed"  => "vignettes/Multilevel_Crossed.md",
            "GSEM"                  => "vignettes/GSEM.md",
            "SAM"                   => "vignettes/SAM.md",
        ],
    ],
    warnonly = true,
)

deploydocs(
    repo = "github.com/xiangao/Lavaan.jl.git",
    devbranch = "main",
    push_preview = false,
)
