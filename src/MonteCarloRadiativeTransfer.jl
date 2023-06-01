module MonteCarloRadiativeTransfer

using Adapt
using CUDA
using DataFrames: DataFrame
using DocStringExtensions: SIGNATURES
using FastGaussQuadrature: gausslegendre
using LinearAlgebra: norm, ⋅, ×
using Printf: @sprintf
using Rotations: RotZY, RotYZ
using StaticArrays: @SVector, @SMatrix, SVector

export run_rt

abstract type AbstractScatterer end
abstract type AbstractGeometry end

include("scatterers/index.jl")
include("utils.jl")
include("config.jl")
include("geometries/index.jl")

"""
Generate a new direction for the given ray.

$(SIGNATURES)
"""
function scatter(cfg::Config, g::AbstractGeometry, I, k, idx)
    ks = spherical_from_cartesian(k)

    # Generate polar scattering angle and compute scattering phase matrix
    P, cosθ = phase_matrix(g, idx, -2.0)

    # Generate azimuthal scattering angle
    eq = -P[1, 2] * I[2] / (P[1, 1] * I[1])
    eu = -P[1, 2] * I[3] / (P[1, 1] * I[1])
    ee = √(eq^2 + eu^2)
    if ee > 1e-12
        γ = acos(eq / ee)
        if eu < 0
            γ = 2π - γ
        end
        ma = 4π * rand() + γ - ee * sin(γ)
        ea = kepler_solver(ma, ee) # Use iterative Kepler solver according to Prof. Muinonen's advice
        ϕ = 0.5 * (ea - γ)
    else
        ϕ = 2π * rand()
    end

    sinθ = √(1 - cosθ^2)
    k′ = ray_to_norm(Vec3(sinθ * cos(ϕ), sinθ * sin(ϕ), cosθ), ks.θ, ks.ϕ)
    I′, Eₕ, Eᵥ = update_I(scatterer(g, idx), I, k′, ks.θ, ks.ϕ)
    I′ = I′ * (I[1] / I′[1])
    return I′, k′, Eₕ, Eᵥ
end

"""
RT peel-off.

$(SIGNATURES)
"""
function collect_rt!(cfg::Config, g::AbstractGeometry, irt, I, k, x, idx, theta_range)
    ks = spherical_from_cartesian(k)

    acc = 0.0
    sca = scatterer(g, idx)
    for j2 in 1:(cfg.Nϕ)
        for j1 in theta_range
            k = Vec3(cfg.sinθ[j1] * cfg.cosϕ[j2], cfg.sinθ[j1] * cfg.sinϕ[j2], cfg.cosθ[j1])
            I′, _, _ = update_I(sca, I, k, ks.θ, ks.ϕ)

            # No normalization is done here since we need to 
            # track the actual intensity change.

            t = distance_to_boundary(g, x, k, idx)

            if t <= cfg.τ₀
                next = exp(-t)
                I′′ = I′ * cfg.inorm[j1] * next
                for i in 1:4
                    CUDA.@atomic irt[i, j1, j2] += I′′[i]
                end
                acc += I′′[1]
            end
        end
    end

    return acc
end

function kernel(cfg, g, I₀, k₀, Eₕ₀, Eᵥ₀, kpath, xpath, idxpath, A, Irt)
    tid = threadIdx().x
    bid = blockIdx().x
    nid = (bid - 1) * blockDim().x + tid
    stride = blockDim().x * gridDim().x

    for _ in nid:stride:(cfg.number_of_rays)
        I = I₀
        k = k₀
        Eₕ = Eₕ₀
        Eᵥ = Eᵥ₀
        x = initial_x(g)
        idx = 1
        scattered_times = 0

        # Direct transmission
        if can_direct_transmit(g)
            τ′ = distance_to_boundary(g, x, k, idx)
            dt = τ′ <= cfg.τ₀ ? exp(-τ′) : 0.0
            CUDA.@atomic A[1] += I[1] * dt
            I *= 1 - dt
        end

        # Initial propagation
        x = initial_propagation(cfg, g, x, k)

        # Conditional propagation
        while I[1] >= cfg.minimum_intensity &&
            scattered_times < cfg.maximum_scattering_times
            scattered_times += 1
            kpath[scattered_times, nid] = k
            xpath[scattered_times, nid] = x
            idxpath[scattered_times, nid] = idx

            I′, k′, x′, idx′, Eₕ′, Eᵥ′ = I, k, x, idx, Eₕ, Eᵥ
            while true
                # Generate a new direction
                I′, k′, Eₕ′, Eᵥ′ = scatter(cfg, g, I, k, idx)

                # Generate the new position after propagation
                t = -log(rand())
                x′ = x + t * k′

                # Check the new position
                pb = pass_boundary(g, x′, idx)
                idx′ = idx + pb

                if pb == -2
                    # Ray goes out of a closed region (invalid)
                    continue
                elseif pb == -1
                    # Ray goes through the upper bound
                    if idx′ < firstindex(g)
                        # Ray goes out of the first layer (invalid)
                        continue
                    else
                        # TODO: handle upper reflection/refraction

                        x′ = Vec3(x[1], x[2], -optical_depth(g, idx))
                        break
                    end
                elseif pb == 1
                    # Ray goes through the lower bound
                    if idx′ > lastindex(g)
                        # Ray goes out of the last layer (invalid)
                        continue
                    else
                        # TODO: handle lower reflection/refraction

                        x′ = Vec3(x[1], x[2], 0.0)
                        break
                    end
                elseif pb == 0
                    # New position is within the current region
                    break
                end
            end

            # Absorption
            CUDA.@atomic A[2] += (1.0 - albedo(g, idx)) * I[1]
            I *= albedo(g, idx)

            # Energy collection
            Iref = 0.0
            Itrans = 0.0

            if minimum_distance_to_first_boundary(g, x, idx) <= cfg.τ₀
                # CB

                # RT
                theta_range = upper_theta_range(cfg, g)
                Iref = collect_rt!(cfg, g, Irt, I, k, x, idx, theta_range)
            end

            if minimum_distance_to_last_boundary(g, x, idx) <= cfg.τ₀
                # CB

                # RT
                theta_range = lower_theta_range(cfg, g)
                Itrans = collect_rt!(cfg, g, Irt, I, k, x, idx, theta_range)
            end

            CUDA.@atomic A[3] += Iref
            CUDA.@atomic A[4] += Itrans

            # Renormalize the Stokes vector
            Iesc = Iref + Itrans
            I = I′ * (I[1] - Iesc) / I′[1]
            k, x, idx, Eₕ, Eᵥ = k′, x′, idx′, Eₕ′, Eᵥ′
        end

        CUDA.@atomic A[5] += I[1]
    end
end

function simulate(cfg::Config, g::AbstractGeometry, I₀)
    θ₀, ϕ₀ = incidence_angle(g)
    k₀ = ray_to_norm(Vec3(0.0, 0.0, 1.0), θ₀, ϕ₀)
    Eₕ₀ = ray_to_norm(Vec3(0.0, -1.0, 0.0), θ₀, ϕ₀)
    Eᵥ₀ = ray_to_norm(Vec3(1.0, 0.0, 0.0), θ₀, ϕ₀)

    # Adt, Aabs, Aref, Atrans, Arem
    A = CUDA.zeros(Float64, 5)
    threads = min(cfg.maximum_threads, cfg.number_of_rays)
    kpath = CUDA.zeros(Vec3, cfg.maximum_scattering_times, threads) # Direction
    xpath = CUDA.zeros(Vec3, cfg.maximum_scattering_times, threads) # Position
    idxpath = CUDA.zeros(Int, cfg.maximum_scattering_times, threads) # Layer index

    Irt_gpu = CUDA.zeros(Float64, 4, cfg.Nθ, cfg.Nϕ)

    @cuda threads=256 blocks=cld(threads, 256) kernel(cfg, g, I₀, k₀, Eₕ₀, Eᵥ₀, kpath,
                                                      xpath, idxpath, A,
                                                      Irt_gpu)

    Adt, Aabs, Aref, Atrans, Arem = Array(A)
    Irt_cpu = Array(Irt_gpu)
    CUDA.unsafe_free!(Irt_gpu)

    return Adt / cfg.number_of_rays,
           Aabs / cfg.number_of_rays,
           Aref / cfg.number_of_rays,
           Atrans / cfg.number_of_rays,
           Arem / cfg.number_of_rays,
           Irt_cpu / cfg.number_of_rays
end

function run_rt(cfg::Config, s::SphereGeometry)
    validate(s)

    IS = initial_I(s)
    Gdt, Gabs, Gref, Grem = zeros(4)
    MRT = zeros(4, 4, cfg.Nθ)
    # MCB = zeros(4, 4, cfg.Nθb)
    NFI = @SVector [i * cfg.Nϕ ÷ 8 + 1 for i in 0:7]
    # NFIB = @SVector [i * cfg.Nϕb ÷ 8 + 1 for i in 0:7]

    for i in axes(IS, 2)
        I = IS[:, i]
        println("==============================")
        @show I
        println("==============================")
        Adt, Aabs, Aref, _, Arem, IRT = simulate(cfg, s, I)

        if iszero(I[4]) # Linear
            Gdt += Adt
            Gabs += Aabs
            Gref += Aref
            Grem += Arem

            for j1 in 1:(cfg.Nθ)
                @. @views MRT[:, 1, j1] += (IRT[:, j1, NFI[1]] + IRT[:, j1, NFI[3]] +
                                            IRT[:, j1, NFI[5]] + IRT[:, j1, NFI[7]]) / 4
                @. @views MRT[:, 2, j1] += (-IRT[:, j1, NFI[1]] + IRT[:, j1, NFI[3]] -
                                            IRT[:, j1, NFI[5]] + IRT[:, j1, NFI[7]]) / 4
                @. @views MRT[:, 3, j1] += (IRT[:, j1, NFI[2]] - IRT[:, j1, NFI[4]] +
                                            IRT[:, j1, NFI[6]] - IRT[:, j1, NFI[8]]) / 4
            end
        else
            for j1 in 1:(cfg.Nθ)
                # s14 & s24 do not change.
                # MRT[1, 4, j1] += 0.0
                # MRT[2, 4, j1] += 0.0
                MRT[3, 4, j1] -= (IRT[3, j1, NFI[1]] + IRT[3, j1, NFI[5]]) / 2
                MRT[4, 4, j1] -= sum(IRT[4, j1, j2] for j2 in 1:(cfg.Nϕ)) / cfg.Nϕ
            end
        end
    end

    @info @sprintf "Direct Transmission (%%) = %.4f\n" Gdt*100
    @info @sprintf "Absorption (%%)          = %.4f\n" Gabs*100
    @info @sprintf "Reflection (%%)          = %.4f\n" Gref*100
    @info @sprintf "Remaining (%%)           = %.4f\n" Grem*100

    Gsca = Gdt + Gref
    renorm = 1 / Gsca
    norm = Array(normalize_coefficient(cfg, s))

    for i in 1:4, j in 1:4
        @views MRT[i, j, :] .*= renorm .* norm
    end

    cosθ = Array(cfg.cosθ)
    SM_rt = DataFrame(:θ => 180.0 / π * acos.(cosθ),
                      [Symbol(:s, i, j) => MRT[i, j, :] for i in 1:4 for j in 1:4]...)

    return SM_rt
end

function run_rt(cfg::Config, p::PlaneGeometry)
    validate(p)

    IS = initial_I(p)
    Gdt, Gabs, Gref, Gtrans, Grem = zeros(5)
    MRT = zeros(4, 4, cfg.Nθ, cfg.Nϕ)
    # MCB = zeros(4, 4, cfg.Nθb, cfg.Nϕb)

    for i in axes(IS, 2)
        I = IS[:, i]
        println("==============================")
        @show I
        println("==============================")
        Adt, Aabs, Aref, Atrans, Arem, IRT = simulate(cfg, p, I)
        k1 = 0
        k2 = 0.0
        for j1 in 2:4
            if abs(I[j1]) ≈ 1.0
                k1 = j1
                k2 = I[j1]
                break
            end
        end

        if k1 == 2
            Gdt += Adt / 2.0
            Gabs += Aabs / 2.0
            Gref += Aref / 2.0
            Gtrans += Atrans / 2.0
            Grem += Arem / 2.0

            for j2 in 1:(cfg.Nϕ)
                for j1 in 1:(cfg.Nθ)
                    @. @views MRT[:, 1, j1, j2] += IRT[:, j1, j2] / 2.0
                end
            end
        end

        for j2 in 1:(cfg.Nϕ)
            for j1 in 1:(cfg.Nθ)
                @. @views MRT[:, k1, j1, j2] += k2 * IRT[:, j1, j2] / 2.0
            end
        end
    end

    @info @sprintf "Direct Transmission (%%) = %.4f\n" Gdt*100
    @info @sprintf "Absorption (%%)          = %.4f\n" Gabs*100
    @info @sprintf "Reflection (%%)          = %.4f\n" Gref*100
    @info @sprintf "Transmission (%%)        = %.4f\n" Gtrans*100
    @info @sprintf "Remaining (%%)           = %.4f\n" Grem*100

    # TODO: RT-CB does not do renorm here, why?
    # Gsca = Gdt + Gref + Gtrans
    # renorm = 1 / Gsca
    norm = Array(normalize_coefficient(cfg, p))

    for j2 in 1:(cfg.Nϕ)
        for i in 1:4, j in 1:4
            @views MRT[i, j, :, j2] .*= norm
        end
    end

    cosθ = Array(cfg.cosθ)
    ϕ = Array(cfg.ϕ)
    SM_rt = DataFrame(:θ => repeat(180.0 / π * acos.(cosθ); outer = cfg.Nϕ),
                      :ϕ => repeat(180.0 / π * ϕ; inner = cfg.Nθ),
                      [Symbol(:s, i, j) => reshape(MRT[i, j, :, :], (:)) for i in 1:4
                       for j in 1:4]...)

    return SM_rt
end

end
