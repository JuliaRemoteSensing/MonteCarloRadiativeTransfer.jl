export MieScatterer

struct MieScatterer{T <: AbstractVector, CT <: AbstractVector} <: AbstractScatterer
    ω::Float64
    kl::Float64
    csrn::T
    F11::T
    F12::T
    F33::T
    F34::T
    S11::CT
    S22::CT
    Nθ::Int
    random_points::Int
end

Adapt.@adapt_structure MieScatterer

"""
Create a Mie-type scatterer.

Arguments:

- `r`: radius of the scatterer
- `m`: complex refractive index of the scatterer
- `λ`: wavelength of the incident light
- `ν`: volume fraction of the scatterer

Keyword arguments:

- `Nθ`: number of angles to evaluate the phase matrix and the amplitude matrix
- `random_points`: number of points to use for the spline interpolation of the random number

$(SIGNATURES)
"""
function MieScatterer(r, m, λ, ν; Nθ = 181, random_points = 1000)
    x = 2π * r / λ
    (; q_ext, q_sca, S, F) = bhmie(x, m; Nθ = Nθ)
    ω = q_sca / q_ext
    kl = ((4 / 3) * r) / (ν * q_ext)
    itpf11 = linear_interpolation(range(0.0, Float64(π), Nθ), F[1])
    csrn = calculate_csrn(itpf11, random_points)
    return MieScatterer(ω, kl, csrn, F..., S..., Nθ, random_points)
end

function phase_matrix(s::MieScatterer, cosθ = -2.0)
    if cosθ < -1
        rn = rand() * s.random_points
        mrn = Int(trunc(rn))
        cosθ = (mrn + 1 - rn) * s.csrn[mrn + 1] + (rn - mrn) * s.csrn[mrn + 2]
    end

    θ = acos(cosθ)
    sn = θ / π * (s.Nθ - 1)
    msn = Int(trunc(sn)) + 1
    F11 = (msn - sn) * s.F11[msn]
    F12 = (msn - sn) * s.F12[msn]
    F33 = (msn - sn) * s.F33[msn]
    F34 = (msn - sn) * s.F34[msn]

    if msn + 1 <= s.Nθ
        F11 += (sn - msn + 1) * s.F11[msn + 1]
        F12 += (sn - msn + 1) * s.F12[msn + 1]
        F33 += (sn - msn + 1) * s.F33[msn + 1]
        F34 += (sn - msn + 1) * s.F34[msn + 1]
    end

    P = @SMatrix [F11 F12 0 0
                  F12 F11 0 0
                  0 0 F33 F34
                  0 0 -F34 F33]

    return P, cosθ
end

function amplitude_matrix(s::MieScatterer, cosθ)
    θ = acos(cosθ)
    sn = θ / π * (s.Nθ - 1)
    msn = Int(trunc(sn)) + 1
    S11 = (msn + 1 - sn) * s.S11[msn]
    S22 = (msn + 1 - sn) * s.S22[msn]

    if msn + 1 <= s.Nθ
        S11 += (sn - msn) * s.S11[msn + 1]
        S22 += (sn - msn) * s.S22[msn + 1]
    end

    S = @SMatrix [S11 0.0
                  0.0 S22]

    return S
end

@doc raw"""
`bhmie([T=Float64,], x, m; Nθ=181, nextra=15, custom_nstop=0)`

Arguments:

- `T`: Type used for calculation. All real numbers will be stored as `T`, while all complex numbers will be stored as `Complex{T}`.
- `x`: Size parameter of the sphere scatterer. Defined as ``\frac{2\pi r}{\lambda}``
- `m`: Relative refractive index of the scatterer.

Keyword arguments:

- `Nθ`: Number of scattering angles to calculate. Default is `181`.
- `nextra`: Extra terms used for the downward calculation of the `d` function. Default is `15`.
- `custom_nstop`: Custom truncation point. Default is `0`, and the empirical formula 

```math
n_{\mathrm{stop}}=\max(x+4\sqrt[3]{x}+2, |m|x)
``` 

will be used.

Scattering information is outputed as a named tuple, including the following fields:

- `q_ext`: Extinction efficiency. Defined as ``C_{\mathrm{ext}}/\pi r^2``, where ``C_{\mathrm{ext}}`` is the extinction cross section.
- `q_sca`: Scattering efficiency.
- `q_abs`: Absorption efficiency.
- `q_back`: Backscattering efficiency. Defined as ``4(\mathrm{d}C_\mathrm{ext}/\mathrm{d}\Omega)/r^2``.
- `asymm`: Asymmetry factor ``\langle\cos(\theta)\rangle``.
- `S` = (`S₁`, `S₂`): Amplitude scattering matrix components. See Eq. (3.12) in Bohren and Huffman (1983). Both S₁ and S₂ are vectors containing `Nθ` values.
- `F` = (`F₁₁`, `F₁₂`, `F₃₃`, `F₃₄`): Mueller scattering matrix components. See Eq. (3.16) in Bohren and Huffman (1983). All Fᵢⱼ are vectors containing `Nθ` values.

References:

- Bohren, C.F., Huffman, D.R., 1983. Absorption and scattering of light by small particles. John Wiley & Sons.
"""
function bhmie(T, x, m; Nθ = 181, nextra = 15, custom_nstop = 0)
    x = T(x)
    m = Complex{T}(m)
    y = m * x
    nstop = iszero(custom_nstop) ? Int(floor(Float64(max(x + 4 * ∛x + 2, x * abs(m))))) :
            custom_nstop
    nmax = nstop + nextra
    θ = collect(range(zero(T), T(π), Nθ))
    μ = cos.(θ)
    d = zeros(Complex{T}, nmax)
    for n in (nmax - 1):-1:1
        d[n] = (n + 1) / y - (1.0 / (d[n + 1] + (n + 1) / y))
    end

    π_ = zeros(T, Nθ)
    τ = zeros(Nθ)
    π₀ = zeros(Nθ)
    π₁ = ones(Nθ)
    S₁ = zeros(Complex{T}, Nθ)
    S₂ = zeros(Complex{T}, Nθ)
    ψ₀ = cos(x)
    ψ₁ = sin(x)
    χ₀ = -sin(x)
    χ₁ = cos(x)
    ξ₁ = complex(ψ₁, -χ₁)
    a = zero(T)
    b = zero(T)
    q_sca = zero(T)
    asymm = zero(T)
    for n in 1:nstop
        fn = T((2n + 1) / (n * (n + 1)))
        ψ = (2n - 1) * ψ₁ / x - ψ₀
        χ = (2n - 1) * χ₁ / x - χ₀
        ξ = complex(ψ, -χ)
        aₙ = ((d[n] / m + n / x) * ψ - ψ₁)
        aₙ /= ((d[n] / m + n / x) * ξ - ξ₁)
        bₙ = ((d[n] * m + n / x) * ψ - ψ₁) / ((d[n] * m + n / x) * ξ - ξ₁)
        q_sca += (abs(aₙ)^2 + abs(bₙ)^2) * (2n + 1)
        asymm += (real(aₙ) * real(bₙ) + imag(aₙ) * imag(bₙ)) * fn
        if n > 1
            asymm += (real(a) * real(aₙ) + imag(a) * imag(aₙ) + real(b) * real(bₙ) +
                      imag(b) * imag(bₙ)) * (n - 1) * (n + 1) / n
        end

        for i in 1:((Nθ + 1) ÷ 2)
            ii = Nθ + 1 - i
            π_[i] = π₁[i]
            τ[i] = n * μ[i] * π_[i] - (n + 1) * π₀[i]
            S₁[i] += fn * (aₙ * π_[i] + bₙ * τ[i])
            S₂[i] += fn * (aₙ * τ[i] + bₙ * π_[i])
            if i < ii
                p = n & 1 == 0 ? -1 : 1
                S₁[ii] += fn * (aₙ * π_[i] - bₙ * τ[i]) * p
                S₂[ii] += fn * (-aₙ * τ[i] + bₙ * π_[i]) * p
            end
        end

        ψ₀, ψ₁ = ψ₁, ψ
        χ₀, χ₁ = χ₁, χ
        ξ₁ = complex(ψ₁, -χ₁)
        a, b = aₙ, bₙ

        @. π₁ = ((2n + 1) * μ * π_ - (n + 1) * π₀) / n
        π₀ .= π_
    end

    asymm *= 2 / q_sca
    q_sca *= 2 / x^2
    q_ext = 4 / x^2 * real(S₁[1])
    q_abs = q_ext - q_sca
    q_back = 4 / x^2 * abs(S₁[end])^2

    F₁₁ = zeros(T, Nθ)
    F₁₂ = zeros(T, Nθ)
    F₃₃ = zeros(T, Nθ)
    F₃₄ = zeros(T, Nθ)
    coeff = 4.0 / (q_sca * x^2)

    for i in 1:Nθ
        F₁₁[i] = (abs2(S₁[i]) + abs2(S₂[i])) * 0.5 * coeff
        F₁₂[i] = -(abs2(S₁[i]) - abs2(S₂[i])) * 0.5 * coeff
        F₃₃[i] = real(S₂[i] * conj(S₁[i])) * coeff
        F₃₄[i] = imag(S₂[i] * conj(S₁[i])) * coeff
    end

    return (q_ext = q_ext, q_sca = q_sca, q_abs = q_abs, q_back = q_back, asymm = asymm,
            S = (S₁, S₂), F = (F₁₁, F₁₂, F₃₃, F₃₄))
end

function bhmie(x, m; Nθ = 181, nextra = 15, custom_nstop = 0)
    bhmie(Float64, x, m; Nθ = Nθ, nextra = nextra, custom_nstop = custom_nstop)
end
