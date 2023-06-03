"""
RT peel-off.

$(SIGNATURES)
"""
function collect_rt_cpu!(cfg::Config, g::AbstractGeometry, Irt, I, k, x, idx, theta_range)
    ks = spherical_from_cartesian(k)

    acc = 0.0
    sca = scatterer(g, idx)
    for j2 in 1:(cfg.Nϕ)
        for j1 in theta_range
            k = Vec3(cfg.sinθ[j1] * cfg.cosϕ[j2], cfg.sinθ[j1] * cfg.sinϕ[j2], cfg.cosθ[j1])
            I′, _, _ = update_I(sca, I, k, ks.cosθ, ks.sinθ, ks.cosϕ, ks.sinϕ)

            # No normalization is done here since we need to 
            # track the actual intensity change.

            t = distance_to_boundary(g, x, k, idx)

            if t <= cfg.τ₀
                next = exp(-t)
                I′′ = I′ * cfg.inorm[j1] * next
                @views Irt[:, j1, j2] .+= I′′
                acc += I′′[1]
            end
        end
    end

    return acc
end

"""
Do coherent backscattering.

$(SIGNATURES)
"""
function collect_cb_cpu!(cfg::Config, g::AbstractGeometry, Icb, I, k, x, idx, I₀, xpath,
                         kpath,
                         idxpath, scattered_times)
    xa = x
    ka = k
    idxa = idx
    kas = spherical_from_cartesian(ka)
    kf = -kpath[1]
    kfs = spherical_from_cartesian(kf)
    θ₀, ϕ₀ = incidence_angle(g)
    cosθ₀, sinθ₀ = cos(θ₀), sin(θ₀)
    cosϕ₀, sinϕ₀ = cos(ϕ₀), sin(ϕ₀)
    sca = scatterer(g, 1) # TODO: fix it

    if scattered_times > 1 # Only consider CB effects when scattered more than once
        ki = kpath[1]
        xb = xpath[1]
        kb = -kpath[2]
        idxb = idxpath[1]
        kbs = spherical_from_cartesian(kb)

        τa = distance_to_boundary(g, xa, kf, idxa)
        τb = distance_to_boundary(g, xb, kf, idxb)
        τab = τa - τb

        Iaf, _, _ = update_I(sca, I, kf, kas.cosθ, kas.sinθ, kas.cosϕ, kas.sinϕ)
        Ei₁, Ei₂ = E_from_I(I₀)
        Ea₁, Ea₂, enorma = forward_E(sca, Ei₁, Ei₂, kpath, scattered_times, cosθ₀, sinθ₀,
                                     cosϕ₀, sinϕ₀)
        Eb₁, Eb₂, enormb = backward_E(sca, Ei₁, Ei₂, kpath, scattered_times, cosθ₀, sinθ₀,
                                      cosϕ₀, sinϕ₀)
        Eb₁ *= exp(enorma - enormb)
        Eb₂ *= exp(enorma - enormb)

        Eaf₁, Eaf₂, _, _ = update_E(sca, Ea₁, Ea₂, kf, kas.cosθ, kas.sinθ, kas.cosϕ,
                                    kas.sinϕ)
        Ebf₁, Ebf₂, _, _ = update_E(sca, Eb₁, Eb₂, kf, kas.cosθ, kas.sinθ, kas.cosϕ,
                                    kas.sinϕ)

        raa = √(Iaf[1] / (real(Eaf₁)^2 + imag(Eaf₁)^2 + real(Eaf₂)^2 + imag(Eaf₂)^2))
        Ea₁ *= raa
        Ea₂ *= raa
        Eb₁ *= raa
        Eb₂ *= raa

        for j2 in 1:(cfg.Nϕb)
            for j1 in 1:(cfg.Nθb)
                kf = Vec3(cfg.sinθb[j1] * cfg.cosϕb[j2], cfg.sinθb[j1] * cfg.sinϕb[j2],
                          cfg.cosθb[j1])
                kf = ray_to_norm(kf, kfs.cosθ, kfs.sinθ, kfs.cosϕ, kfs.sinϕ)

                Eaf₁, Eaf₂, _, _ = update_E(sca, Ea₁, Ea₂, kf, kas.cosθ, kas.sinθ, kas.cosϕ,
                                            kas.sinϕ)
                Ebf₁, Ebf₂, Eₕ, Eᵥ = update_E(sca, Eb₁, Eb₂, kf, kbs.cosθ, kbs.sinθ,
                                              kbs.cosϕ, kbs.sinϕ)

                ΔΦ = optical_depth(g, 1) * ((ki + kf) ⋅ (xa - xb))
                τa = distance_to_boundary(g, xa, kf, idxa)
                if τa <= cfg.τ₀
                    nexta = exp(-0.5 * τa)
                    Eaf₁ *= nexta
                    Eaf₂ *= nexta
                    Iaf = I_from_E(Eaf₁, Eaf₂)

                    τb = distance_to_boundary(g, xb, kf, idxb)
                    if abs(τb + τab) <= cfg.τ₀
                        nextb = exp(-0.5 * (τb + τab))
                        Ebf₁ *= nextb
                        Ebf₂ *= nextb
                    else
                        Ebf₁ = 0.0im
                        Ebf₂ = 0.0im
                    end

                    Ibf = I_from_E(Ebf₁, Ebf₂)
                    Eabf₁ = Eaf₁ * exp(ΔΦ * 1.0im) + Ebf₁
                    Eabf₂ = Eaf₂ * exp(ΔΦ * 1.0im) + Ebf₂
                    Iabf = I_from_E(Eabf₁, Eabf₂)
                    Iabf = Iabf * (Iaf[1] / (Iaf[1] + Ibf[1]))

                    Eₕ = norm_to_ray(Eₕ, kfs.cosθ, kfs.sinθ, kfs.cosϕ, kfs.sinϕ)
                    Eᵥ = norm_to_ray(Eᵥ, kfs.cosθ, kfs.sinθ, kfs.cosϕ, kfs.sinϕ)
                    cosψ = cfg.sinϕb[j2] * Eₕ[1] - cfg.cosϕb[j2] * Eₕ[2]
                    sinψ = -cfg.sinϕb[j2] * Eᵥ[1] + cfg.cosϕb[j2] * Eᵥ[2]
                    cos2ψ = cosψ^2 - sinψ^2
                    sin2ψ = 2.0 * cosψ * sinψ
                    I1 = rotate(Iabf, cos2ψ, sin2ψ)

                    @views Icb[:, j1, j2] .+= I1
                end
            end
        end
    else
        for j2 in 1:(cfg.Nϕb)
            for j1 in 1:(cfg.Nθb)
                kf = Vec3(cfg.sinθb[j1] * cfg.cosϕb[j2], cfg.sinθb[j1] * cfg.sinϕb[j2],
                          cfg.cosθb[j1])
                kf = ray_to_norm(kf, kfs.cosθ, kfs.sinθ, kfs.cosϕ, kfs.sinϕ)

                Iaf, Eₕ, Eᵥ = update_I(sca, I, kf, kas.cosθ, kas.sinθ, kas.cosϕ,
                                       kas.sinϕ)

                τa = distance_to_boundary(g, xa, kf, idxa)
                if τa <= cfg.τ₀
                    nexta = exp(-τa)
                    Iaf = Iaf * nexta

                    Eₕ = norm_to_ray(Eₕ, kfs.cosθ, kfs.sinθ, kfs.cosϕ, kfs.sinϕ)
                    Eᵥ = norm_to_ray(Eᵥ, kfs.cosθ, kfs.sinθ, kfs.cosϕ, kfs.sinϕ)
                    cosψ = cfg.sinϕb[j2] * Eₕ[1] - cfg.cosϕb[j2] * Eₕ[2]
                    sinψ = -cfg.sinϕb[j2] * Eᵥ[1] + cfg.cosϕb[j2] * Eᵥ[2]
                    I2 = Iaf

                    cos2ψ = cosψ^2 - sinψ^2
                    sin2ψ = 2.0 * cosψ * sinψ
                    I1 = rotate(I2, cos2ψ, sin2ψ)

                    @views Icb[:, j1, j2] .+= I1
                end
            end
        end
    end
end

function cpu_kernel(cfg, g, I₀, k₀, Eₕ₀, Eᵥ₀, kpath, xpath, idxpath, A, Irt, Icb)
    upper = upper_theta_range(cfg, g)
    lower = lower_theta_range(cfg, g)

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
        A[1] += I[1] * dt
        I *= 1 - dt
    end

    # Initial propagation
    x = initial_propagation(cfg, g, x, k)

    # Conditional propagation
    while I[1] >= cfg.minimum_intensity &&
        scattered_times < cfg.maximum_scattering_times
        scattered_times += 1
        kpath[scattered_times] = k
        xpath[scattered_times] = x
        idxpath[scattered_times] = idx

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
        A[2] += (1.0 - albedo(g, idx)) * I[1]
        I *= albedo(g, idx)

        # Energy collection
        Iref = 0.0
        Itrans = 0.0

        near_upper = minimum_distance_to_first_boundary(g, x, idx) <= cfg.τ₀
        near_lower = minimum_distance_to_last_boundary(g, x, idx) <= cfg.τ₀

        # CB
        if cfg.compute_cb && (near_upper || near_lower)
            collect_cb_cpu!(cfg, g, Icb, I, k, x, idx, I₀, xpath, kpath, idxpath,
                            scattered_times)
        end

        # RT
        if near_upper
            Iref = collect_rt_cpu!(cfg, g, Irt, I, k, x, idx, upper)
        end

        if near_lower
            Itrans = collect_rt_cpu!(cfg, g, Irt, I, k, x, idx, lower)
        end

        A[3] += Iref
        A[4] += Itrans

        # Renormalize the Stokes vector
        Iesc = Iref + Itrans
        I = I′ * (I[1] - Iesc) / I′[1]
        k, x, idx, Eₕ, Eᵥ = k′, x′, idx′, Eₕ′, Eᵥ′
    end

    A[5] += I[1]
end
