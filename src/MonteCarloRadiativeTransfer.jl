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
include("config.jl")
include("utils.jl")
include("geometries/index.jl")
include("cpu.jl")
include("gpu.jl")

function simulate(cfg::Config, g::AbstractGeometry, I₀; use_gpu)
    θ₀, ϕ₀ = incidence_angle(g)
    cosθ₀ = cos(θ₀)
    sinθ₀ = sin(θ₀)
    cosϕ₀ = cos(ϕ₀)
    sinϕ₀ = sin(ϕ₀)
    k₀ = ray_to_norm(Vec3(0.0, 0.0, 1.0), cosθ₀, sinθ₀, cosϕ₀, sinϕ₀)
    Eₕ₀ = ray_to_norm(Vec3(0.0, -1.0, 0.0), cosθ₀, sinθ₀, cosϕ₀, sinϕ₀)
    Eᵥ₀ = ray_to_norm(Vec3(1.0, 0.0, 0.0), cosθ₀, sinθ₀, cosϕ₀, sinϕ₀)

    if use_gpu
        # Adt, Aabs, Aref, Atrans, Arem
        A = CUDA.zeros(Float64, 5)
        threads = min(cfg.maximum_threads, cfg.number_of_rays)
        kpath = CUDA.zeros(Vec3, cfg.maximum_scattering_times, threads) # Direction
        xpath = CUDA.zeros(Vec3, cfg.maximum_scattering_times, threads) # Position
        idxpath = CUDA.zeros(Int8, cfg.maximum_scattering_times, threads) # Layer index
        Irt_gpu = CUDA.zeros(Float64, 4, cfg.Nθ, cfg.Nϕ)
        Icb_gpu = CUDA.zeros(Float64, 4, cfg.Nθb, cfg.Nϕb)

        @cuda threads=256 blocks=cld(threads, 256) gpu_kernel(cfg, g, I₀, k₀, Eₕ₀, Eᵥ₀,
                                                              kpath,
                                                              xpath, idxpath, A,
                                                              Irt_gpu, Icb_gpu)

        Irt_cpu = Array(Irt_gpu)
        CUDA.unsafe_free!(Irt_gpu)

        Icb_cpu = Array(Icb_gpu)
        CUDA.unsafe_free!(Icb_gpu)
        Adt, Aabs, Aref, Atrans, Arem = Array(A)
    else
        # Adt, Aabs, Aref, Atrans, Arem
        threads = Threads.nthreads()
        As = [zeros(Float64, 5) for _ in 1:threads]
        kpaths = [zeros(Vec3, cfg.maximum_scattering_times) for _ in 1:threads] # Direction
        xpaths = [zeros(Vec3, cfg.maximum_scattering_times) for _ in 1:threads] # Position
        idxpaths = [zeros(Int8, cfg.maximum_scattering_times) for _ in 1:threads]  # Layer index
        Irt_cpus = [zeros(Float64, 4, cfg.Nθ, cfg.Nϕ) for _ in 1:threads]
        Icb_cpus = [zeros(Float64, 4, cfg.Nθb, cfg.Nϕb) for _ in 1:threads]

        Threads.@threads for _ in 1:(cfg.number_of_rays)
            tid = Threads.threadid()
            cpu_kernel(cfg, g, I₀, k₀, Eₕ₀, Eᵥ₀, kpaths[tid], xpaths[tid], idxpaths[tid],
                       As[tid], Irt_cpus[tid], Icb_cpus[tid])
        end

        Adt, Aabs, Aref, Atrans, Arem = sum(As)
        Irt_cpu = sum(Irt_cpus)
        Icb_cpu = sum(Icb_cpus)
    end

    return Adt / cfg.number_of_rays,
           Aabs / cfg.number_of_rays,
           Aref / cfg.number_of_rays,
           Atrans / cfg.number_of_rays,
           Arem / cfg.number_of_rays,
           Irt_cpu / cfg.number_of_rays,
           Icb_cpu / cfg.number_of_rays
end

function run_rt(cfg::Config, s::SphereGeometry; use_gpu = false, compute_cb = true)
    validate(s)

    IS = initial_I(s)
    Gdt, Gabs, Gref, Grem = zeros(4)
    MRT = zeros(4, 4, cfg.Nθ)
    MCB = zeros(4, 4, cfg.Nθb)
    NFI = @SVector [i * cfg.Nϕ ÷ 8 + 1 for i in 0:7]
    NFIB = @SVector [i * cfg.Nϕb ÷ 8 + 1 for i in 0:7]

    if use_gpu
        cfg = adapt(CuArray, cfg)
    end

    for i in axes(IS, 2)
        I = IS[:, i]
        println("==============================")
        @show I
        println("==============================")
        Adt, Aabs, Aref, _, Arem, IRT, ICB = simulate(cfg, s, I; use_gpu = use_gpu)

        if iszero(I[4]) # Linear
            Gdt += Adt
            Gabs += Aabs
            Gref += Aref
            Grem += Arem

            for j1 in 1:(cfg.Nθ)
                @. @views MRT[:, 1, j1] += (IRT[:, j1, NFI[1]] + IRT[:, j1, NFI[3]] +
                                            IRT[:, j1, NFI[5]] + IRT[:, j1, NFI[7]]) * 0.25
                @. @views MRT[:, 2, j1] += (-IRT[:, j1, NFI[1]] + IRT[:, j1, NFI[3]] -
                                            IRT[:, j1, NFI[5]] + IRT[:, j1, NFI[7]]) * 0.25
                @. @views MRT[:, 3, j1] += (IRT[:, j1, NFI[2]] - IRT[:, j1, NFI[4]] +
                                            IRT[:, j1, NFI[6]] - IRT[:, j1, NFI[8]]) * 0.25
            end

            for j1 in 1:(cfg.Nθb)
                @. @views MCB[:, 1, j1] += (ICB[:, j1, NFIB[1]] +
                                            ICB[:, j1, NFIB[3]] +
                                            ICB[:, j1, NFIB[5]] +
                                            ICB[:, j1, NFIB[7]]) * 0.25
                @. @views MCB[:, 2, j1] += (-ICB[:, j1, NFIB[1]] +
                                            ICB[:, j1, NFIB[3]] -
                                            ICB[:, j1, NFIB[5]] +
                                            ICB[:, j1, NFIB[7]]) * 0.25
                @. @views MCB[:, 3, j1] += (ICB[:, j1, NFIB[2]] -
                                            ICB[:, j1, NFIB[4]] +
                                            ICB[:, j1, NFIB[6]] -
                                            ICB[:, j1, NFIB[8]]) * 0.25
            end
        else
            for j1 in 1:(cfg.Nθ)
                # s14 & s24 do not change.
                # MRT[1, 4, j1] += 0.0
                # MRT[2, 4, j1] += 0.0
                MRT[3, 4, j1] -= (IRT[3, j1, NFI[1]] + IRT[3, j1, NFI[5]]) * 0.5
                MRT[4, 4, j1] -= sum(IRT[4, j1, j2] for j2 in 1:(cfg.Nϕ)) / cfg.Nϕ
            end

            for j1 in 1:(cfg.Nθb)
                # s14 & s24 do not change.
                # MCB[1, 4, j1] += 0.0
                # MCB[2, 4, j1] += 0.0
                MCB[3, 4, j1] -= (ICB[3, j1, NFIB[1]] + ICB[3, j1, NFIB[5]]) * 0.5
                MCB[4, 4, j1] -= sum(ICB[4, j1, j2] for j2 in 1:(cfg.Nϕb)) / cfg.Nϕb
            end
        end
    end

    @info @sprintf "Direct Transmission (%%) = %.4f\n" Gdt*100
    @info @sprintf "Absorption (%%)          = %.4f\n" Gabs*100
    @info @sprintf "Reflection (%%)          = %.4f\n" Gref*100
    @info @sprintf "Remaining (%%)           = %.4f\n" Grem*100

    Gsca = Gdt + Gref
    renorm = 1 / Gsca
    inorm = Array(cfg.inorm)
    norm = inv.(inorm)

    for i in 1:4, j in 1:4
        @. @views MRT[i, j, :] *= renorm * norm
    end

    MCB .*= renorm

    cosθ = Array(cfg.cosθ)
    cosθb = Array(cfg.cosθb)
    SM_rt = DataFrame(:θ => 180.0 / π * acos.(cosθ),
                      [Symbol(:s, i, j) => MRT[i, j, :] for i in 1:4 for j in 1:4]...)
    SM_cb = DataFrame(:θ => 180.0 / π * acos.(cosθb),
                      [Symbol(:s, i, j) => MCB[i, j, :] for i in 1:4 for j in 1:4]...)
    return SM_rt, SM_cb
end

function run_rt(cfg::Config, p::PlaneGeometry; use_gpu = false)
    validate(p)

    IS = initial_I(p)
    Gdt, Gabs, Gref, Gtrans, Grem = zeros(5)
    MRT = zeros(4, 4, cfg.Nθ, cfg.Nϕ)
    MCB = zeros(4, 4, cfg.Nθb, cfg.Nϕb)

    if use_gpu
        p = adapt(CuArray, p)
        names = propertynames(cfg)
        cfg = Config(; map(Pair, names, getproperty.(Ref(cfg), names))..., geometry = p)
        cfg = adapt(CuArray, cfg)
    end

    for i in axes(IS, 2)
        I = IS[:, i]
        println("==============================")
        @show I
        println("==============================")
        Adt, Aabs, Aref, Atrans, Arem, IRT, ICB = simulate(cfg, p, I; use_gpu = use_gpu)
        k1 = 0
        k2 = 0.0
        for j1 in 2:4
            if abs(I[j1]) > 1.0 - eps()
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

            for j2 in 1:(cfg.Nϕb)
                for j1 in 1:(cfg.Nθb)
                    @. @views MCB[:, 1, j1, j2] += ICB[:, j1, j2] / 2.0
                end
            end
        end

        for j2 in 1:(cfg.Nϕ)
            for j1 in 1:(cfg.Nθ)
                @. @views MRT[:, k1, j1, j2] += k2 * IRT[:, j1, j2] / 2.0
            end
        end

        for j2 in 1:(cfg.Nϕb)
            for j1 in 1:(cfg.Nθb)
                @. @views MCB[:, k1, j1, j2] += k2 * ICB[:, j1, j2] / 2.0
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
    cosθ = Array(cfg.cosθ)
    cosθb = Array(cfg.cosθb)
    inorm = Array(cfg.inorm)
    norm = @. (abs ∘ inv)(4.0 * cosθ * inorm)

    for j2 in 1:(cfg.Nϕ)
        for i in 1:4, j in 1:4
            @views MRT[i, j, :, j2] .*= norm
        end
    end

    ϕ = Array(cfg.ϕ)
    ϕb = Array(cfg.ϕb)

    for j1 in 1:(cfg.Nθb)
        norm1 = inv(abs(4.0 * cosθb[j1]))
        for j2 in 1:(cfg.Nϕb)
            @views MCB[:, :, j1, j2] .*= norm1
        end
    end

    SM_rt = DataFrame(:θ => repeat(180.0 / π * acos.(cosθ); outer = cfg.Nϕ),
                      :ϕ => repeat(180.0 / π * ϕ; inner = cfg.Nθ),
                      [Symbol(:s, i, j) => reshape(MRT[i, j, :, :], (:)) for i in 1:4
                       for j in 1:4]...)

    SM_cb = DataFrame(:θ => repeat(180.0 / π * acos.(cosθb); outer = cfg.Nϕb),
                      :ϕ => repeat(180.0 / π * ϕb; inner = cfg.Nθb),
                      [Symbol(:s, i, j) => reshape(MCB[i, j, :, :], (:)) for i in 1:4
                       for j in 1:4]...)

    return SM_rt, SM_cb
end

end
