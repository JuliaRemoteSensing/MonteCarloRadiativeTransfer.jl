const Vec3 = SVector{3, Float64}
const Vec4 = SVector{4, Float64}

"""
Calculate the spherical coordinates from Cartesian coordinates.

$(SIGNATURES)
"""
function spherical_from_cartesian(x)
    # r = 1 always holds for this program
    s² = x[1]^2 + x[2]^2
    if s² < 1e-24
        cosθ, sinθ = sign(x[3]), 0.0
        cosϕ, sinϕ = 1.0, 0.0
    else
        cosθ = x[3]
        sinθ = √(1.0 - cosθ^2)
        s⁻¹ = 1.0 / √s²
        cosϕ = x[1] * s⁻¹
        sinϕ = x[2] * s⁻¹
    end
    return (cosθ = cosθ, sinθ = sinθ, cosϕ = cosϕ, sinϕ = sinϕ)
end

"""
Rotate the direction vector from the ray to the normal coordinate system. The ray is assumed to propagate in the +z direction in its own coordinate system.

$(SIGNATURES)
"""
function ray_to_norm(x, cosθ, sinθ, cosϕ, sinϕ)
    return Vec3(x[1] * cosθ * cosϕ - x[2] * sinϕ + x[3] * sinθ * cosϕ,
                x[1] * cosθ * sinϕ + x[2] * cosϕ + x[3] * sinθ * sinϕ,
                -x[1] * sinθ + x[3] * cosθ)
end

"""
Rotate the direction vector from the normal to the ray coordinate system. The ray is assumed to propagate in the +z direction in its own coordinate system.

$(SIGNATURES)
"""
function norm_to_ray(x, cosθ, sinθ, cosϕ, sinϕ)
    return Vec3(x[1] * cosθ * cosϕ + x[2] * cosθ * sinϕ - x[3] * sinθ,
                -x[1] * sinϕ + x[2] * cosϕ,
                x[1] * sinθ * cosϕ + x[2] * sinθ * sinϕ + x[3] * cosθ)
end

"""
Calculate the directions of Eₕ and Eᵥ given the wave direction k.

$(SIGNATURES)
"""
function get_basis(k)
    if abs(k[3]) > 1.0 - eps()
        return Vec3(0, -1, 0), Vec3(sign(k[3]), 0, 0)
    else
        Eᵥ₃ = -√(1.0 - k[3]^2)
        Eᵥ₃⁻¹ = 1.0 / Eᵥ₃
        Eₕ₁ = k[2] * -Eᵥ₃⁻¹
        Eₕ₂ = k[1] * Eᵥ₃⁻¹
        return Vec3(Eₕ₁, Eₕ₂, 0), Vec3(-k[3] * Eₕ₂, -k[3] * Eₕ₁, Eᵥ₃)
    end
end

@doc raw"""
Rotate a Stokes vector by the rotation matrix

```math
\mathbf{R}(\phi)=\left[\begin{matrix}
1 & 0 & 0 & 0 \\
0 & \cos(2\phi) & \sin(2\phi) & 0 \\
0 & -\sin(2\phi) & \cos(2\phi) & 0 \\
0 & 0 & 0 & 1 \\
\end{matrix}\right]
```

Notice that in the code, the matrix is generated column-wise.
"""
@inline function rotate(I, ϕ)
    return rotate(I, cos(2ϕ), sin(2ϕ))
end

@inline function rotate(I, cos2ϕ, sin2ϕ)
    return Vec4(I[1], cos2ϕ * I[2] + sin2ϕ * I[3], -sin2ϕ * I[2] + cos2ϕ * I[3], I[4])
end

"""
Given a Stokes vector `I`, the incident direction `(θ, ϕ)`, which is given in the normal coordinate system, and the scatter direction `k`, calculate the rotated Stokes vector and its electric field components.

$(SIGNATURES)
"""
function update_I(scatterer, I, k, cosθ, sinθ, cosϕ, sinϕ)
    Eₕ, Eᵥ = get_basis(k)

    k = norm_to_ray(k, cosθ, sinθ, cosϕ, sinϕ)
    ks = spherical_from_cartesian(k)

    cosθₖ = ks.cosθ
    sinθₖ = ks.sinθ
    cosϕₖ = ks.cosϕ
    sinϕₖ = ks.sinϕ
    Eₕₖ = ray_to_norm(Vec3(sinϕₖ, -cosϕₖ, 0), cosθ, sinθ, cosϕ, sinϕ)
    Eᵥₖ = ray_to_norm(Vec3(cosθₖ * cosϕₖ, cosθₖ * sinϕₖ, -sinθₖ), cosθ, sinθ, cosϕ, sinϕ)

    cosψ = Eₕ[1] * Eₕₖ[1] + Eₕ[2] * Eₕₖ[2]
    sinψ = -Eₕ[1] * Eᵥₖ[1] - Eₕ[2] * Eᵥₖ[2]

    P, _ = phase_matrix(scatterer, k[3])
    I′ = P * rotate(I, ks.cosϕ^2 - ks.sinϕ^2, -2 * ks.sinϕ * ks.cosϕ) # Note that we rotate the Stokes vector by -ϕ instead of ϕ here
    I′ = rotate(I′, cosψ^2 - sinψ^2, 2 * sinψ * cosψ)

    return I′, Eₕ, Eᵥ
end

"""
Get electric field `E` generated from Stokes vector `I`.

$(SIGNATURES)
"""
function E_from_I(I)
    tol = 1e-14

    if I[1] - abs(I[2]) > tol * I[1]
        A₁ = √(0.5 * (I[1] - I[2]))
        A₂ = √(0.5 * (I[1] + I[2]))
        q = √(I[1]^2 - I[2]^2)
        if I[3] >= q
            ϕ₂ = 0.0
        elseif I[3] <= -q
            ϕ₂ = Float64(π)
        else
            ϕ₂ = acos(I[3] / q)
            if I[4] > 0.0
                ϕ₂ = 2π - ϕ₂
            end
        end
    elseif I[1] + I[2] <= tol * I[1]
        A₁ = √I[1]
        A₂ = 0.0
        ϕ₂ = 0.0
    else
        A₁ = 0.0
        A₂ = √I[1]
        ϕ₂ = 0.0
    end

    E₁ = A₁ * complex(1.0, 0.0) # ϕ₁ = 0.0 always holds
    E₂ = A₂ * complex(cos(ϕ₂), sin(ϕ₂))

    return E₁, E₂
end

"""
Get Stokes vector `I` from electric field `E`.

$(SIGNATURES)
"""
function I_from_E(E₁, E₂)
    qh = real(E₁)^2 + imag(E₁)^2
    qv = real(E₂)^2 + imag(E₂)^2

    return Vec4(qh + qv, -qh + qv, real(E₂ * E₁' + E₂' * E₁),
                real((E₂ * E₁' - E₂' * E₁) * 1.0im))
end

"""
Given electric field `(E₁, E₂)`, the incident direction `(θ, ϕ)`, which is given in the normal coordinate system, and the scatter direction `k`, calculate the rotated electric field. This function is similar to `update_I`.

$(SIGNATURES)
"""
function update_E(scatterer, E₁, E₂, k, cosθ, sinθ, cosϕ, sinϕ)
    Eₕ, Eᵥ = get_basis(k)
    k = norm_to_ray(k, cosθ, sinθ, cosϕ, sinϕ)
    ks = spherical_from_cartesian(k)

    cosθₖ = ks.cosθ
    sinθₖ = ks.sinθ
    cosϕₖ = ks.cosϕ
    sinϕₖ = ks.sinϕ
    Eₕₖ = ray_to_norm(Vec3(sinϕₖ, -cosϕₖ, 0), cosθ, sinθ, cosϕ, sinϕ)
    Eᵥₖ = ray_to_norm(Vec3(cosθₖ * cosϕₖ, cosθₖ * sinϕₖ, -sinθₖ), cosθ, sinθ, cosϕ, sinϕ)

    cosψ = Eₕ[1] * Eₕₖ[1] + Eₕ[2] * Eₕₖ[2]
    sinψ = Eₕ[1] * Eᵥₖ[1] + Eₕ[2] * Eᵥₖ[2]

    S = amplitude_matrix(scatterer, k[3])

    E₁, E₂ = E₁ * cosϕₖ + E₂ * sinϕₖ, -E₁ * sinϕₖ + E₂ * cosϕₖ
    E₁, E₂ = S[1, 1] * E₁ + S[1, 2] * E₂, S[2, 1] * E₁ + S[2, 2] * E₂
    E₁, E₂ = E₁ * cosψ + E₂ * sinψ, -E₁ * sinψ + E₂ * cosψ

    return E₁, E₂, Eₕ, Eᵥ
end

"""
Go through the scattering path from the second to the last scatterer.

$(SIGNATURES)
"""
function forward_E(scatterer, E₁, E₂, kpath, scattered_times, cosθ, sinθ, cosϕ, sinϕ)
    enorm = 0.0
    norm0 = √(abs(E₁)^2 + abs(E₂)^2)
    E₁′ = E₁
    E₂′ = E₂
    ks = (cosθ = cosθ, sinθ = sinθ, cosϕ = cosϕ, sinϕ = sinϕ)
    for k in 2:scattered_times
        E₁′, E₂′, _, _ = update_E(scatterer, E₁′, E₂′, kpath[k], ks.cosθ, ks.sinθ, ks.cosϕ,
                                  ks.sinϕ)
        norm2 = √(abs(E₁′)^2 + abs(E₂′)^2)
        E₁′ *= norm0 / norm2
        E₂′ *= norm0 / norm2
        enorm += log(norm0 / norm2)
        ks = spherical_from_cartesian(kpath[k])
    end

    return E₁′, E₂′, enorm
end

"""
Go through the scattering path from the last to the second scatterer.

$(SIGNATURES)
"""
function backward_E(scatterer, E₁, E₂, kpath, scattered_times, cosθ, sinθ, cosϕ, sinϕ)
    enorm = 0.0
    norm0 = √(abs(E₁)^2 + abs(E₂)^2)

    E₁′ = E₁
    E₂′ = E₂
    ks = (cosθ = cosθ, sinθ = sinθ, cosϕ = cosϕ, sinϕ = sinϕ)
    for k in scattered_times:-1:2
        E₁′, E₂′, _, _ = update_E(scatterer, E₁′, E₂′, -kpath[k], ks.cosθ, ks.sinθ, ks.cosϕ,
                                  ks.sinϕ)
        norm2 = √(abs(E₁′)^2 + abs(E₂′)^2)
        E₁′ *= norm0 / norm2
        E₂′ *= norm0 / norm2
        enorm += log(norm0 / norm2)
        ks = spherical_from_cartesian(-kpath[k])
    end

    return E₁′, E₂′, enorm
end

function kepler_solver(M, e)
    tol = 1e-12
    MM = M - floor(M / 2π) * 2π
    ea = M + 0.85 * e * sign(sin(MM))
    dea = 1.0

    while abs(dea) > tol
        f3 = e * cos(ea)
        f2 = e * sin(ea)
        f1 = 1.0 - f3
        f = ea - f2 - M
        dea = -f / f1
        dea = -f / (f1 + f2 * dea / 2.0)
        dea = -f / (f1 + f2 * dea / 2.0 + f3 * dea^2 / 6.0)
        ea += dea
    end

    return ea
end

function calculate_csrn(itp11, n)
    csrn = zeros(n + 1)
    csrn[1] = -1.0
    csrn[n + 1] = 1.0

    xv = Vector((0:n) ./ (n / π))
    ex = Vector((0:n) ./ n)
    yv = cumsum(map(x -> itp11(x) * sin(x), xv))

    for i in 1:(n + 1)
        xv[i] = π - xv[i]
        yv[i] = 1 - yv[i] / yv[end]
    end

    for i in 1:(n + 1)
        xv[i], yv[i] = yv[i], -cos(xv[i])
    end

    reverse!(xv)
    reverse!(yv)

    # Use linear interpolation to ensure csrn is monotonically increasing
    itp = linear_interpolation(xv[2:end], yv[2:end])
    csrn = itp(ex)
end

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
    k′ = ray_to_norm(Vec3(sinθ * cos(ϕ), sinθ * sin(ϕ), cosθ), ks.cosθ, ks.sinθ, ks.cosϕ,
                     ks.sinϕ)
    I′, Eₕ, Eᵥ = update_I(scatterer(g, idx), I, k′, ks.cosθ, ks.sinθ, ks.cosϕ,
                          ks.sinϕ)
    I′ = I′ * (I[1] / I′[1])
    return I′, k′, Eₕ, Eᵥ
end

# TODO: Geometric optics

function reflect_and_refract()
end

function mirror()
end

function snel()
end

function fresnel()
end

function ref_matrix()
end
