using MonteCarloRadiativeTransfer
using Documenter

DocMeta.setdocmeta!(MonteCarloRadiativeTransfer, :DocTestSetup,
                    :(using MonteCarloRadiativeTransfer); recursive = true)

makedocs(;
         modules = [MonteCarloRadiativeTransfer],
         authors = "Gabriel Wu <wuzihua@pku.edu.cn> and contributors",
         repo = "https://github.com/lucifer1004/MonteCarloRadiativeTransfer.jl/blob/{commit}{path}#{line}",
         sitename = "MonteCarloRadiativeTransfer.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://lucifer1004.github.io/MonteCarloRadiativeTransfer.jl",
                                  edit_link = "main",
                                  assets = String[]),
         pages = [
             "Home" => "index.md",
         ])

deploydocs(;
           repo = "github.com/lucifer1004/MonteCarloRadiativeTransfer.jl",
           devbranch = "main")
