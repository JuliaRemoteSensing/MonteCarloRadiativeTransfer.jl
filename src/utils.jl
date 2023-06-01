const Vec3 = SVector{3, Float64}
const Vec4 = SVector{4, Float64}

"""
Calculate the spherical coordinates from Cartesian coordinates.

$(SIGNATURES)
"""
function spherical_from_cartesian(x)
    # r = 1 always holds for this program
    θ = acos(x[3])
    ϕ = hypot(x[1], x[2]) ≈ 0 ? 0.0 : atan(x[2], x[1])
    return (θ = θ, ϕ = ϕ)
end

"""
Rotate the direction vector from the ray to the normal coordinate system. The ray is assumed to propagate in the +z direction in its own coordinate system.

$(SIGNATURES)
"""
function ray_to_norm(x, θ, ϕ)
    return RotZY(ϕ, θ) * x
end

"""
Rotate the direction vector from the normal to the ray coordinate system. The ray is assumed to propagate in the +z direction in its own coordinate system.

$(SIGNATURES)
"""
function norm_to_ray(x, θ, ϕ)
    return RotYZ(-θ, -ϕ) * x
end

"""
Calculate the directions of Eₕ and Eᵥ given the wave direction k.

$(SIGNATURES)
"""
function get_basis(k)
    if k[3] ≈ 1.0
        return Vec3(0, -1, 0), Vec3(1, 0, 0)
    elseif k[3] ≈ -1.0
        return Vec3(0, -1, 0), Vec3(-1, 0, 0)
    else
        Eᵥ₃ = -√(1.0 - k[3]^2)
        return Vec3(k[2] / -Eᵥ₃, k[1] / Eᵥ₃, 0),
               Vec3(-k[3] * k[1] / Eᵥ₃, -k[3] * k[2] / Eᵥ₃, Eᵥ₃)
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
function rotate(I, ϕ)
    return rotate(I, cos(2ϕ), sin(2ϕ))
end

function rotate(I, cos2ϕ, sin2ϕ)
    return Vec4(I[1], cos2ϕ * I[2] + sin2ϕ * I[3], -sin2ϕ * I[2] + cos2ϕ * I[3], I[4])
end

"""
Given a Stokes vector `I`, the incident direction `(θ, ϕ)`, which is given in the normal coordinate system, and the scatter direction `k`, calculate the rotated Stokes vector and its electric field components.

$(SIGNATURES)
"""
function update_I(scatterer, I, k, θ, ϕ)
    Eₕ, Eᵥ = get_basis(k)
    k = norm_to_ray(k, θ, ϕ)
    ks = spherical_from_cartesian(k)

    sinθₖ = sin(ks.θ)
    cosθₖ = cos(ks.θ)
    sinϕₖ = sin(ks.ϕ)
    cosϕₖ = cos(ks.ϕ)
    Eₕₖ = ray_to_norm(Vec3(sinϕₖ, -cosϕₖ, 0), θ, ϕ)
    Eᵥₖ = ray_to_norm(Vec3(cosθₖ * cosϕₖ, cosθₖ * sinϕₖ, -sinθₖ), θ, ϕ)

    cosψ = Eₕ[1] * Eₕₖ[1] + Eₕ[2] * Eₕₖ[2]
    sinψ = -Eₕ[1] * Eᵥₖ[1] - Eₕ[2] * Eᵥₖ[2]

    P, _ = phase_matrix(scatterer, k[3])
    I′ = P * rotate(I, -ks.ϕ) # Note that we rotate the Stokes vector by -ϕ instead of ϕ here
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
    qh = abs(E₁)^2
    qv = abs(E₂)^2

    return Vec4(qh + qv, -qh + qv, E₂ * E₁' + E₂' * E₁, (E₂ * E₁' - E₂' * E₁) * 1.0im)
end

"""
Given electric field `(E₁, E₂)`, the incident direction `(θ, ϕ)`, which is given in the normal coordinate system, and the scatter direction `k`, calculate the rotated electric field. This function is similar to `update_I`.

$(SIGNATURES)
"""
function update_E(scatterer, E₁, E₂, k, θ, ϕ)
    Eₕ, Eᵥ = get_basis(k)
    k = norm_to_ray(k, θ, ϕ)
    ks = spherical_from_cartesian(k)

    sinθₖ = sin(ks.θ)
    cosθₖ = cos(ks.θ)
    sinϕₖ = sin(ks.ϕ)
    cosϕₖ = cos(ks.ϕ)
    Eₕₖ = ray_to_norm(Vec3(sinϕₖ, -cosϕₖ, 0), θ, ϕ)
    Eᵥₖ = ray_to_norm(Vec3(cosθₖ * cosϕₖ, cosθₖ * sinϕₖ, -sinθₖ), θ, ϕ)

    cosψ = Eₕ[1] * Eₕₖ[1] + Eₕ[2] * Eₕₖ[2]
    sinψ = Eₕ[1] * Eᵥₖ[1] + Eₕ[2] * Eᵥₖ[2]

    S = amplitude_matrix(scatterer, k[3])
    E₁′ = cosψ * S[1] * (E₁ * cosϕₖ + E₂ * sinϕₖ) + sinψ * S[2] * (-E₁ * sinϕₖ + E₂ * cosϕₖ)
    E₂′ = -sinψ * S[1] * (E₁ * cosϕₖ + E₂ * sinϕₖ) +
          cosψ * S[2] * (-E₁ * sinϕₖ + E₂ * cosϕₖ)

    return E₁′, E₂′, Eₕ, Eᵥ
end

"""
Go through the scattering path from the second to the last scatterer.

$(SIGNATURES)
"""
function forward_E(scatterer, E₁, E₂, kpath, scattered_times, θ, ϕ)
    enorm = 0.0
    norm0 = √(abs(E₁)^2 + abs(E₂)^2)

    E₁′ = E₁
    E₂′ = E₂
    ks = (θ = θ, ϕ = ϕ)
    for k in 2:scattered_times
        E₁′, E₂′, _, _ = update_E(scatterer, E₁′, E₂′, kpath[k], ks.θ, ks.ϕ)
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
function backward_E(scatterer, E₁, E₂, kpath, scattered_times, θ, ϕ)
    enorm = 0.0
    norm0 = √(abs(E₁)^2 + abs(E₂)^2)

    E₁′ = E₁
    E₂′ = E₂
    ks = (θ = θ, ϕ = ϕ)
    for k in scattered_times:-1:2
        E₁′, E₂′, _, _ = update_E(scatterer, E₁′, E₂′, -kpath[k], ks.θ, ks.ϕ)
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
