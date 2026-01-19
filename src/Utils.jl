module Utils

using LinearAlgebra

export compute_eig_and_dU_reg1, expand_spatial_to_spinful, add_term_ij_PH

function compute_eig_and_dU_reg1(H::AbstractMatrix{T}, H_alphas; eta::Real=1e-8) where T
    H_sym = Hermitian(H)
    ε, U = eigen(H_sym)

    diff_mat = ε .- ε' .+ im * eta
    F = -real.(1.0 ./ diff_mat)
    F[diagind(F)] .= 0.0

    keys_iter = keys(H_alphas)
    dE = Dict{eltype(keys_iter),Vector{Float64}}()
    dU = Dict{eltype(keys_iter),Matrix{ComplexF64}}()

    for k in keys_iter
        dH = H_alphas[k]

        dH_MO = U' * dH * U
        dE[k] = real.(diag(dH_MO))

        # dU = sum_{m!=n} |m> <m|dH|n> / (En - Em)
        dU[k] = U * (F .* dH_MO)
    end

    return ε, U, dE, dU
end

function expand_spatial_to_spinful(U_spatial::Matrix{T}) where T
    Nlat, Norb_spatial = size(U_spatial)
    U_spin = zeros(T, 2 * Nlat, 2 * Norb_spatial)
    U_spin[1:2:end, 1:Norb_spatial] = U_spatial[:, :]
    U_spin[2:2:end, Norb_spatial+1:end] = U_spatial[:, :]
    return U_spin
end

function add_term_ij_PH(tmat, i, j, chi, eta; singlet::Bool=true)
    sign = singlet ? 1 : -1
    @assert i != j
    tmat[(i-1)*2+1, (j-1)*2+1] += chi        # Julia 1-based
    tmat[(i-1)*2+2, (j-1)*2+2] -= conj(chi) * sign
    tmat[(i-1)*2+1, (j-1)*2+2] += eta
    tmat[(j-1)*2+1, (i-1)*2+2] += eta * sign
end

end
