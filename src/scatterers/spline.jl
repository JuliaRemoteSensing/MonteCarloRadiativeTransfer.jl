export SplineScatterer

struct SplineScatterer{T <: AbstractVector, CT <: AbstractVector} <: AbstractScatterer
    ω::Float64
    kl::Float64
    csrn::T
    F11::T
    F12::T
    F22::T
    F33::T
    F34::T
    F44::T
    S11::CT
    S22::CT
end

Adapt.@adapt_structure SplineScatterer

function SplineScatterer(ω, kl, F11, F12, F22, F33, F34, F44, S11, S22;
                         random_points = 1000)
    itpf11 = linear_interpolation(range(0.0, Float64(π), length(F11)), F11)
    csrn = calculate_csrn(itpf11, random_points)
    return SplineScatterer(ω, kl, csrn, F11, F12, F22, F33, F34, F44, S11, S22)
end

function phase_matrix(s::SplineScatterer, cosθ = -2.0)
    if cosθ < -1
        rn = rand() * (length(s.csrn) - 1)
        mrn = Int(trunc(rn))
        cosθ = (mrn + 1 - rn) * s.csrn[mrn + 1] + (rn - mrn) * s.csrn[mrn + 2]
    end

    θ = acos(cosθ)
    sn = θ / π * self.NSPL
    msn = Int(trunc(sn)) + 1
    F11 = (msn + 1 - sn) * s.F11[msn]
    F12 = (msn + 1 - sn) * s.F12[msn]
    F22 = (msn + 1 - sn) * s.F22[msn]
    F33 = (msn + 1 - sn) * s.F33[msn]
    F34 = (msn + 1 - sn) * s.F34[msn]
    F44 = (msn + 1 - sn) * s.F44[msn]

    if msn + 1 <= length(s.F11)
        F11 += (sn - msn) * s.F11[msn + 1]
        F12 += (sn - msn) * s.F12[msn + 1]
        F22 += (sn - msn) * s.F22[msn + 1]
        F33 += (sn - msn) * s.F33[msn + 1]
        F34 += (sn - msn) * s.F34[msn + 1]
        F44 += (sn - msn) * s.F44[msn + 1]
    end

    P = @SMatrix [F11 F12 0 0
                  F12 F22 0 0
                  0 0 F33 F34
                  0 0 -F34 F44]

    return P, cosθ
end

function amplitude_matrix(s::SplineScatterer, cosθ)
    θ = acos(cosθ)
    sn = θ / π * self.NSPL
    msn = Int(trunc(sn)) + 1
    S11 = (msn + 1 - sn) * s.S11[msn]
    S22 = (msn + 1 - sn) * s.S22[msn]

    if msn + 1 <= length(s.S11)
        S11 += (sn - msn) * s.S11[msn + 1]
        S22 += (sn - msn) * s.S22[msn + 1]
    end

    S = @SMatrix [S11 0.0
                  0.0 S22]

    return S
end
