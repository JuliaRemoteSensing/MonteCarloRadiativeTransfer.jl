mean_free_path(s::AbstractScatterer) = s.kl

single_scattering_albedo(s::AbstractScatterer) = s.ω

include("rayleigh.jl")
