export SphereGeometry

"""
A spherical region containing scatterers.
"""
Base.@kwdef struct SphereGeometry{T <: AbstractScatterer} <: AbstractGeometry
    "Actual radius of the sphere region"
    R::Float64

    "Refractive index of the hosting media"
    n::Float64 = 1.0

    "Scatterer in the sphere region"
    scatterer::T

    "Optical radius of the sphere region"
    τ::Float64 = R / mean_free_path(scatterer)

    "Single scattering albedo of the scatterers"
    ω::Float64 = single_scattering_albedo(scatterer)
end

Base.firstindex(::SphereGeometry) = 1
Base.lastindex(::SphereGeometry) = 1

layers(::SphereGeometry) = 1
optical_depth(s::SphereGeometry, _idx) = s.τ
albedo(s::SphereGeometry, _idx) = s.ω
scatterer(s::SphereGeometry, _idx) = s.scatterer
phase_matrix(s::SphereGeometry, _idx, cosθ) = phase_matrix(s.scatterer, cosθ)

function validate(s::SphereGeometry)
    s.R > 0 || throw(ArgumentError("Radius must be positive"))
end

can_direct_transmit(::SphereGeometry) = true

incidence_angle(::SphereGeometry) = (Float64(π), 0.0)

function initial_x(s::SphereGeometry)
    r = s.τ * √rand()
    ϕ = 2π * rand()
    return Vec3(r * cos(ϕ), r * sin(ϕ), √(s.τ^2 - r^2))
end

initial_I(::SphereGeometry) = @SMatrix [1.0 1.0
                                        -1.0 0.0
                                        0.0 0.0
                                        0.0 -1.0]

function initial_propagation(cfg, ::SphereGeometry, x, k)
    τ′ = 2.0 * x[3]
    if τ′ <= cfg.τ₀
        t = -log(1 - rand() * (1 - exp(-τ′)))
    else
        t = τ′ + 1
        while t > τ′
            t = -log(1 - rand())
        end
    end

    x += t * k
    return x
end

function distance_to_boundary(s::SphereGeometry, x, k, _idx)
    kx = k ⋅ x
    xx = x ⋅ x
    return -kx + √(kx^2 + s.τ^2 - xx)
end

pass_boundary(s::SphereGeometry, x, _idx) = norm(x) > s.τ ? -2 : 0

minimum_distance_to_first_boundary(s::SphereGeometry, x, _idx) = s.τ - norm(x)

upper_theta_range(cfg::Config, ::SphereGeometry) = 1:(cfg.Nθ)

minimum_distance_to_last_boundary(::SphereGeometry, _x, _idx) = Inf

lower_theta_range(cfg::Config, ::SphereGeometry) = 0:-1

function initial_weights(::SphereGeometry, Nθ, Nϕ)
    cosθ, w = gausslegendre(Nθ)
    sinθ = @. sqrt(1.0 - cosθ^2)
    inorm = w ./ 2Nϕ
    return cosθ, sinθ, inorm
end

function normalize_coefficient(cfg::Config, ::SphereGeometry)
    return @. 1.0 / cfg.inorm
end
