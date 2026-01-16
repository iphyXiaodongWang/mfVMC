module PartonSquare

push!(LOAD_PATH, "./src/") 
using LinearAlgebra
using OrderedCollections
using Utils

export U1SFlux, make_ansatz_and_derivs

"""
    U1SFlux
    
对应 Python 代码中的 U1SFlux 类。
参数：
- Lx, Ly: 晶格尺寸
- t: 跳跃参数 (默认为 1.0)
- eta1: 控制磁通量的变分参数, Phi = arctan(eta1)
- bcx, bcy: 边界条件相位/系数 (默认为 1.0, 对应 Python 代码逻辑)
"""
struct U1SFluxParams
    Lx::Int
    Ly::Int
    t::Float64
    phi::Float64
    bcx::Float64
    bcy::Float64
end

function U1SFluxParams(;Lx=4, Ly=4, t=-1.0, phi=0.1, bcx=1.0, bcy=1.0)
    return U1SFluxParams(Lx, Ly, t, phi, bcx, bcy)
end


@inline function xy_to_idx(x::Int, y::Int, Ly::Int)
    return y + (x - 1) * Ly
end

function make_ansatz_and_derivs(p::U1SFluxParams)
    Lx, Ly = p.Lx, p.Ly
    Nlat = Lx * Ly
    phi = p.phi
    
    H = zeros(ComplexF64, Nlat, Nlat)
    dH_phi = zeros(ComplexF64, Nlat, Nlat) 
    
    for x in 1:Lx
        for y in 1:Ly
            id0 = xy_to_idx(x, y, Ly)
            
            # --- Y 方向 ---
            idy = (y == Ly) ? xy_to_idx(x, 1, Ly) : xy_to_idx(x, y+1, Ly)
            bc_y = (y == Ly) ? p.bcy : 1.0
            
            sign_y = (-1)^(x + y - 1)
            phase_y = exp(1im * sign_y * phi)
            
            val_y = p.t * phase_y * bc_y
            H[id0, idy] += val_y
            
            # dH = val * i * sign
            dH_phi[id0, idy] += val_y * (1im * sign_y)

            # --- X 方向 ---
            idx = (x == Lx) ? xy_to_idx(1, y, Ly) : xy_to_idx(x+1, y, Ly)
            bc_x = (x == Lx) ? p.bcx : 1.0
            
            sign_x = (-1)^(x + y)
            phase_x = exp(1im * sign_x * phi)
            
            val_x = p.t * phase_x * bc_x
            H[id0, idx] += val_x
            
            dH_phi[id0, idx] += val_x * (1im * sign_x)
        end
    end
    
    H = Hermitian(H + H')
    dH_phi = Hermitian(dH_phi + dH_phi')
    
    # 2. 对角化并计算导数 (Utils)
    H_alphas = Dict(:phi => Matrix(dH_phi))
    ε, U_full, dE, dU_dict = Utils.compute_eig_and_dU_reg1(Matrix(H), H_alphas)

    eig_eq_error = norm(Matrix(H) * U_full - U_full * Diagonal(ε))
    println("Eigen equation error (HU - Uε): ", eig_eq_error)
    
    # 3. 截取占据态 (Half-filling)
    n_occ = Nlat ÷ 2
    println("ε is",  ε[n_occ-4:n_occ+4])
    U_occ = U_full[:, 1:n_occ]
    dU_occ = dU_dict[:phi][:, 1:n_occ]
    
    # 4. 扩展为 Spinful 形式 (使用你的切片逻辑)
    gs_U = expand_spatial_to_spinful(U_occ)
    
    # 5. 扩展并转置为 dUt 形式 (用于 VMC 导数)
    dU_matrix = expand_spatial_to_spinful(dU_occ)
    
    # 6. 封装
    dUt_params = OrderedDict{Symbol, Matrix{ComplexF64}}()
    dUt_params[:phi] = transpose(dU_matrix)
    
    return ε, gs_U, dUt_params
end

end # module