export RayleighScatterer

struct RayleighScatterer <: AbstractScatterer
    ω::Float64
    kl::Float64
end

function phase_matrix(::RayleighScatterer, cosθ = -2.0)
    if cosθ < -1
        rn = 2.0 * (1.0 - 2.0 * rand())
        cosθ = ∛(√(1.0 + rn^2) + rn) - ∛(√(1.0 + rn^2) - rn)
    end
    P = @SMatrix [0.75*(1 + cosθ^2) -0.75*(1 - cosθ^2) 0 0
                  -0.75*(1 - cosθ^2) 0.75*(1 + cosθ^2) 0 0
                  0 0 1.5cosθ 0
                  0 0 0 1.5cosθ]
    return P, cosθ
end

function amplitude_matrix(::RayleighScatterer, cosθ)
    return @SMatrix [1.0im 0.0
                     0.0 cosθ*1.0im]
end
