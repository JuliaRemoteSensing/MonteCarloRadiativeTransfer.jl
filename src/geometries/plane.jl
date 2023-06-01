export Layer, PlaneGeometry

"""
A planar layer containing scatterers.
"""
Base.@kwdef struct Layer{T <: AbstractScatterer}
    "Thickness of the layer"
    d::Float64

    "Refractive index of the hosting media"
    n::Float64 = 1.0

    "Scatterer in the layer"
    scatterer::T

    "Optical thickness of the layer"
    τ::Float64 = d / mean_free_path(scatterer)

    "Single scattering albedo of the scatterers"
    ω::Float64 = single_scattering_albedo(scatterer)
end

Base.isinf(l::Layer) = l.d < 0

"""
A multi-layered geometry.
"""
Base.@kwdef struct PlaneGeometry{T <: AbstractArray} <:
                   AbstractGeometry
    "Incidence polar angle"
    θ::Float64

    "Incidence azimuthal angle"
    ϕ::Float64

    "Layers of the plane geometry"
    layers::T
end

Adapt.@adapt_structure PlaneGeometry

Base.firstindex(p::PlaneGeometry) = firstindex(p.layers)
Base.lastindex(p::PlaneGeometry) = lastindex(p.layers)

layers(p::PlaneGeometry) = length(p.layers)
optical_depth(p::PlaneGeometry, idx) = p.layers[idx].τ
albedo(p::PlaneGeometry, idx) = p.layers[idx].ω
scatterer(p::PlaneGeometry, idx) = p.layers[idx].scatterer
phase_matrix(p::PlaneGeometry, idx, cosθ) = phase_matrix(p.layers[idx].scatterer, cosθ)

function validate(p::PlaneGeometry)
    π / 2 < p.θ <= π || throw(ArgumentError("θ must be in (π/2, π]"))
    0 <= p.ϕ < 2π || throw(ArgumentError("ϕ must be in [0, 2π)"))
    !isempty(p.layers) || throw(ArgumentError("There needs to be at least one layer"))

    for i in eachindex(p.layers)
        if i != lastindex(p.layers) && isinf(p.layers[i])
            throw(ArgumentError("Only the last layer can be infinite"))
        end
    end
end

can_direct_transmit(p::PlaneGeometry) = length(p.layers) == 1 && !isinf(p.layers[1])

incidence_angle(p::PlaneGeometry) = (p.θ, p.ϕ)

initial_x(::PlaneGeometry) = Vec3(0.0, 0.0, 0.0)

function initial_I(::PlaneGeometry)
    @SMatrix [1.0 1.0 1.0 1.0 1.0 1.0
              -1.0 1.0 0.0 0.0 0.0 0.0
              0.0 0.0 -1.0 1.0 0.0 0.0
              0.0 0.0 0.0 0.0 -1.0 1.0]
end

function initial_propagation(cfg::Config, p::PlaneGeometry, x, k)
    if isinf(p.layers[1])
        t = -log(rand())
    else
        has_dt = can_direct_transmit(p)
        τ′ = distance_to_boundary(p, x, k, 1)
        if has_dt && τ′ <= cfg.τ₀
            t = -log(1 - rand() * (1 - exp(-τ′)))
        else
            t = τ′ + 1
            while t > τ′
                t = -log(rand())
            end
        end
    end

    x += t * k
    return x
end

function distance_to_boundary(p::PlaneGeometry, x, k, idx)
    if k[3] > 0
        return -x[3] / k[3]
    elseif !isinf(p.layers[idx])
        return (x[3] + p.layers[idx].τ) / -k[3]
    else
        return Inf
    end
end

function pass_boundary(p::PlaneGeometry, x, idx)
    if x[3] > 0
        return -1
    elseif !isinf(p.layers[idx]) && x[3] < -p.layers[idx].τ
        return 1
    else
        return 0
    end
end

minimum_distance_to_first_boundary(::PlaneGeometry, x, idx) = idx == 1 ? abs(x[3]) : Inf

upper_theta_range(cfg::Config, ::PlaneGeometry) = (cfg.Nθ ÷ 2 + 1):(cfg.Nθ)

function minimum_distance_to_last_boundary(p::PlaneGeometry, x, idx)
    if idx == lastindex(p) && !isinf(p.layers[idx])
        return optical_depth(p, idx) - abs(x[3])
    else
        return Inf
    end
end

lower_theta_range(cfg::Config, ::PlaneGeometry) = 1:(cfg.Nθ ÷ 2)

function initial_weights(::PlaneGeometry, Nθ, Nϕ)
    x, w = gausslegendre(Nθ ÷ 2)
    cosθ = zeros(Nθ)
    inorm = zeros(Nθ)
    @views cosθ[1:(Nθ ÷ 2)] .= (x .- 1.0) / 2.0
    @views inorm[1:(Nθ ÷ 2)] .= w ./ 2.0
    @views cosθ[(Nθ ÷ 2 + 1):Nθ] .= -reverse(cosθ[1:(Nθ ÷ 2)])
    @views inorm[(Nθ ÷ 2 + 1):Nθ] .= reverse(inorm[1:(Nθ ÷ 2)])
    inorm .*= 1.0 / 2Nϕ
    sinθ = @. sqrt(1.0 - cosθ^2)
    return cosθ, sinθ, inorm
end

function normalize_coefficient(cfg::Config, p::PlaneGeometry)
    return @. abs(cos(p.θ) / cfg.cosθ) / (4 * cfg.inorm)
end
