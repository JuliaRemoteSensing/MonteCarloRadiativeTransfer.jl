export Config

const DEFAULT_THEB = π / 180.0 * Float64[0.00,
                             0.02,
                             0.04,
                             0.06,
                             0.08,
                             0.10,
                             0.15,
                             0.20,
                             0.25,
                             0.30,
                             0.35,
                             0.40,
                             0.45,
                             0.50,
                             0.60,
                             0.70,
                             0.80,
                             0.90,
                             1.00,
                             1.50,
                             2.00,
                             2.50,
                             3.00,
                             4.00,
                             5.00,
                             6.00,
                             7.00,
                             8.00,
                             9.00,
                             12.0,
                             15.0,
                             18.0,
                             21.0,
                             24.0,
                             27.0,
                             30.0,
                             40.0,
                             50.0,
                             60.0,
                             70.0,
                             80.0,
                             90.0,
                             100.0,
                             110.0,
                             120.0,
                             130.0,
                             140.0,
                             150.0,
                             160.0,
                             170.0,
                             180.0]

Base.@kwdef struct Config{T <: AbstractGeometry, V <: AbstractVector}
    "Geometry to simulate"
    geometry::T

    "Number of rays to simulate"
    number_of_rays::Int = 1000

    "Threshold to stop ray tracing"
    minimum_intensity::Float64 = 1e-6

    "After how many scattering events should the ray be forcefully terminated"
    maximum_scattering_times::Int = 10000

    "Maximum number of threads to use"
    maximum_threads::Int = 8192

    "Maximum optical depth to consider"
    τ₀::Float64 = 50.0

    "Number of azimuthal angles"
    Nϕ::Int = 48

    "Azimuthal angles"
    ϕ::V = collect(range(0, stop = 2π * (Nϕ - 1) / Nϕ, length = Nϕ))

    "Cosine of azimuthal angles for coherent backscattering"
    cosϕ::V = cos.(ϕ)

    "Sine of azimuthal angles for coherent backscattering"
    sinϕ::V = sin.(ϕ)

    "Number of polar angles"
    Nθ::Int = 64

    "Cosine of polar angles"
    cosθ::V = initial_weights(geometry, Nθ, Nϕ)[1]

    "Sine of polar angles"
    sinθ::V = initial_weights(geometry, Nθ, Nϕ)[2]

    "Normalized weights"
    inorm::V = initial_weights(geometry, Nθ, Nϕ)[3]

    "Polar angles for coherent backscattering"
    θb::V = DEFAULT_THEB

    "Number of polar angles for coherent backscattering"
    Nθb::Int = length(θb)

    "Cosine of polar angles for coherent backscattering"
    cosθb::V = cos.(θb)

    "Sine of polar angles for coherent backscattering"
    sinθb::V = sin.(θb)

    "Number of azimuthal angles for coherent backscattering"
    Nϕb::Int = 8

    "Azimuthal angles for coherent backscattering"
    ϕb::V = collect(range(0, stop = 2π * (Nϕb - 1) / Nϕb, length = Nϕb))

    "Cosine of azimuthal angles for coherent backscattering"
    cosϕb::V = cos.(ϕb)

    "Sine of azimuthal angles for coherent backscattering"
    sinϕb::V = sin.(ϕb)
end

Adapt.@adapt_structure Config
