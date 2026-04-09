module PartonSquare

push!(LOAD_PATH, "./src/")
using LinearAlgebra
using OrderedCollections
using Utils
using MPI

export U1SFlux, make_ansatz_and_derivs
export DefectHeisenbergParams, make_defect_ansatz_and_derivs, make_defect_ansatz_and_derivs_MPI

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

function U1SFluxParams(; Lx = 4, Ly = 4, t = -1.0, phi = 0.1, bcx = 1.0, bcy = 1.0)
	return U1SFluxParams(Lx, Ly, t, phi, bcx, bcy)
end

"""
	is_root_rank()

判断当前 MPI rank 是否为 0. 如果 MPI 未初始化, 视为单进程返回 true.
"""
function is_root_rank()
	if !MPI.Initialized()
		return true
	end
	return MPI.Comm_rank(MPI.COMM_WORLD) == 0
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
			idy = (y == Ly) ? xy_to_idx(x, 1, Ly) : xy_to_idx(x, y + 1, Ly)
			bc_y = (y == Ly) ? p.bcy : 1.0

			sign_y = (-1)^(x + y - 1)
			phase_y = exp(1im * sign_y * phi)

			val_y = p.t * phase_y * bc_y
			H[id0, idy] += val_y

			# dH = val * i * sign
			dH_phi[id0, idy] += val_y * (1im * sign_y)

			# --- X 方向 ---
			idx = (x == Lx) ? xy_to_idx(1, y, Ly) : xy_to_idx(x + 1, y, Ly)
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
	if is_root_rank()
		println("Eigen equation error (HU - Uε): ", eig_eq_error)
	end

	# 3. 截取占据态 (Half-filling)
	n_occ = Nlat ÷ 2
	if is_root_rank()
		println("ε is", ε[(n_occ-4):(n_occ+4)])
	end
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

#Heisenberg
struct HeisenbergParams
	Lx::Int
	Ly::Int
	chi1::Float64
	etad1::Float64
	etas1::Float64
	mz::Float64
	bcx::Float64
	bcy::Float64
end
function HeisenbergParams(; Lx = 4, Ly = 4, chi1 = 0.0, etad1 = 0.0, etas1 = 0.0, mz = 0.0, bcx = 1.0, bcy = 1.0)
	return HeisenbergParams(Lx, Ly, chi1, etad1, etas1, mz, bcx, bcy)
end
function build_ham_PH(p::HeisenbergParams)
	Lx, Ly = p.Lx, p.Ly
	Nlat = Lx * Ly
	chi1 = p.chi1
	etad1 = p.etad1
	etas1 = p.etas1
	mz = p.mz
	H = zeros(Float64, 2 * Nlat, 2 * Nlat)
	for x in 1:Lx
		for y in 1:Ly
			id0 = xy_to_idx(x, y, Ly)
			Q = (-1)^(x + y)
			# --- Y 方向 ---
			idy = (y == Ly) ? xy_to_idx(x, 1, Ly) : xy_to_idx(x, y + 1, Ly)
			bc_y = (y == Ly) ? p.bcy : 1.0
			# --- X 方向 ---
			idx = (x == Lx) ? xy_to_idx(1, y, Ly) : xy_to_idx(x + 1, y, Ly)
			bc_x = (x == Lx) ? p.bcx : 1.0
			add_term_ij_PH(H, id0, idx, chi1 * bc_x, (+etas1 - etad1) * bc_x)
			add_term_ij_PH(H, id0, idy, chi1 * bc_y, (etas1 + etad1) * bc_y)
			H[2*(id0-1)+1, 2*(id0-1)+1] += Q * mz / 2
			H[2*(id0-1)+2, 2*(id0-1)+2] += Q * mz / 2
		end
	end

	H = Hermitian(H + H')
	return H
end
function build_ham_pfa(p::HeisenbergParams)
	Lx, Ly = p.Lx, p.Ly
	Nlat = Lx * Ly
	chi1 = p.chi1
	etad1 = p.etad1
	etas1 = p.etas1
	mz = p.mz
	hblock = zeros(Float64, 2 * Nlat, 2 * Nlat)
	pblock = zeros(Float64, 2 * Nlat, 2 * Nlat)
	H = zeros(Float64, 4 * Nlat, 4 * Nlat)
	for x in 1:Lx
		for y in 1:Ly
			id0 = xy_to_idx(x, y, Ly)
			Q = (-1)^(x + y)
			# --- Y 方向 ---
			idy = (y == Ly) ? xy_to_idx(x, 1, Ly) : xy_to_idx(x, y + 1, Ly)
			bc_y = (y == Ly) ? p.bcy : 1.0
			# --- X 方向 ---
			idx = (x == Lx) ? xy_to_idx(1, y, Ly) : xy_to_idx(x + 1, y, Ly)
			bc_x = (x == Lx) ? p.bcx : 1.0
			add_term_ij_nonPH(hblock, id0, idx, chi1 * bc_x)
			add_term_ij_nonPH(hblock, id0, idy, chi1 * bc_y)
			add_term_ij_pfa_pairing(pblock, id0, idx, (+etas1 - etad1) * bc_x)
			add_term_ij_pfa_pairing(pblock, id0, idy, (+etas1 + etad1) * bc_y)
			hblock[2*(id0-1)+1, 2*(id0-1)+1] += Q * mz / 2
			hblock[2*(id0-1)+2, 2*(id0-1)+2] += -Q * mz / 2
		end
	end
	hblock.+=hblock'
	pblock.-=pblock'
	H=Hermitian([-hblock -pblock; pblock hblock])
	return H
end
function make_ansatz_and_derivs(p::HeisenbergParams; para_names::Vector{Symbol} = [:etad1, :etas1, :mz], target_sz::Int = 0)
	H = build_ham_PH(p)
	H_alphas = Dict{Symbol, Matrix{Float64}}()
	for name in para_names
		pp = HeisenbergParams(; (; name => 1.0, :Lx => p.Lx, :Ly => p.Ly, :bcx => p.bcx, :bcy => p.bcy)...)
		dH = build_ham_PH(pp)
		H_alphas[name] = dH
	end
	# 2. 对角化并计算导数 (Utils)
	ε, U_full, dE, dU_dict = Utils.compute_eig_and_dU_reg1(H, H_alphas)
	eig_eq_error = norm(Matrix(H) * U_full - U_full * Diagonal(ε))
	if is_root_rank()
		println("Eigen equation error (HU - Uε): ", eig_eq_error)
	end
	#做了PH变换后粒子数不再守恒，守恒的只有total Sz，根据输入的target_sz截取
	# 3. 截取占据态并封装
	Nlat = p.Lx * p.Ly
	n_occ = Nlat + target_sz
	if is_root_rank()
		println("ε is", ε[(n_occ-4):(n_occ+4)])
	end
	U_occ = U_full[:, 1:n_occ]
	dUt_occ = OrderedDict(alpha => permutedims(real.(dU_dict[alpha][:, 1:n_occ])) for alpha in para_names)
	return ε, U_occ, dUt_occ
end

function make_ansatz_and_derivs_pfa(p::HeisenbergParams; para_names::Vector{Symbol} = [:etad1, :etas1, :mz])
	H = build_ham_pfa(p)
	H_alphas = Dict{Symbol, Matrix{Float64}}()
	for name in para_names
		pp = HeisenbergParams(; (; name => 1.0, :Lx => p.Lx, :Ly => p.Ly, :bcx => p.bcx, :bcy => p.bcy)...)
		dH = build_ham_pfa(pp)
		H_alphas[name] = dH
	end
	# 2. 对角化并计算导数 (Utils)
	ε, U_full, dE, dU_dict = Utils.compute_eig_and_dU_reg1(H, H_alphas)
	eig_eq_error = norm(Matrix(H) * U_full - U_full * Diagonal(ε))
	if is_root_rank()
		println("Eigen equation error (HU - Uε): ", eig_eq_error)
	end
	# 3. 计算F矩阵及其导数
	N=div(length(ε), 2)
	umatrix = U_full[1:N, 1:N]
	vmatrix = U_full[(N+1):end, 1:N]
	if is_root_rank()
		parity = det(umatrix + vmatrix) * det(umatrix - vmatrix)
		println("parity=", parity)
	end
	pinv_umatrix=pinv(umatrix)
	F=vmatrix*pinv_umatrix
	F .= 0.5 .* (F .- F')
	F_alphas=OrderedDict{Symbol, Matrix{Float64}}()
	for name in para_names
		dumatrix=dU_dict[name][1:N, 1:N]
		dvmatrix=dU_dict[name][(N+1):end, 1:N]
		dF=dvmatrix*pinv_umatrix+vmatrix * pinv_derivative(umatrix, dumatrix, pinv_umatrix)
		dF.=0.5 .* (dF .- dF')
		F_alphas[name] = dF
	end

	return ε, F, F_alphas
end
# ======================================================================
# Defect Heisenberg (PH, determinant)
# ======================================================================
struct DefectHeisenbergParams
	Lx::Int
	Ly::Int
	bcx::Float64
	bcy::Float64
	mu::Float64
	chi1::Float64
	etad1::Float64
	etas1::Float64
	mz::Dict{Symbol, Float64}
	chi2::Dict{Symbol, Float64}
	etad2::Dict{Symbol, Float64}
	etas2::Dict{Symbol, Float64}
	chi3::Dict{Symbol, Float64}
	etad3::Dict{Symbol, Float64}
	etas3::Dict{Symbol, Float64}
	defect_positions::Vector{Tuple{Int, Int}}
	defect_index::Vector{Int}
end

"""
	DefectHeisenbergParams(; Lx, Ly, bcx, bcy, mu, chi1, etad1, etas1, mz, chi2, etad2, etas2, chi3, etad3, etas3, defect_positions, defect_index)

构造 defect 版 Heisenberg 参数.
参数:
- Lx, Ly: 晶格尺寸.
- bcx, bcy: 边界条件.
- mu, chi1, etad1, etas1: 均匀参数.
- mz, chi2, etad2, etas2, chi3, etad3, etas3: site 依赖参数 (Dict).
- defect_positions: defect 坐标 (x, y), 1-based.
- defect_index: defect 对应的 1-based site index.
返回:
- DefectHeisenbergParams.
"""
function DefectHeisenbergParams(;
	Lx::Int,
	Ly::Int,
	bcx::Float64 = 1.0,
	bcy::Float64 = 1.0,
	mu::Float64 = 0.0,
	chi1::Float64 = 0.0,
	etad1::Float64 = 0.0,
	etas1::Float64 = 0.0,
	mz::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
	chi2::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
	etad2::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
	etas2::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
	chi3::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
	etad3::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
	etas3::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
	defect_positions::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[],
	defect_index::Vector{Int} = Int[],
)
	return DefectHeisenbergParams(
		Lx, Ly, bcx, bcy, mu, chi1, etad1, etas1,
		mz, chi2, etad2, etas2, chi3, etad3, etas3,
		defect_positions, defect_index,
	)
end

"""
	xy_to_id_1based(x::Int, y::Int, lx::Int, ly::Int)

将 (x, y) 转成 1-based site index (使用周期边界).
参数:
- x::Int, x 坐标 (1-based).
- y::Int, y 坐标 (1-based).
- lx::Int, x 方向长度.
- ly::Int, y 方向长度.
返回:
- Int, 1-based site index.
"""
function xy_to_id_1based(x::Int, y::Int, lx::Int, ly::Int)::Int
	return mod1(y, ly) + (mod1(x, lx) - 1) * ly
end

"""
	calc_bc(lx::Int, ly::Int, bcx::Float64, bcy::Float64, id0::Int, id1::Int)

计算两个 1-based site 的边界条件因子.
参数:
- lx, ly: 晶格尺寸.
- bcx, bcy: 边界条件.
- id0, id1: 1-based site index.
返回:
- Float64, 边界条件因子.
"""
function calc_bc(lx::Int, ly::Int, bcx::Float64, bcy::Float64, id0::Int, id1::Int)::Float64
	x0 = (id0 - 1) ÷ ly + 1
	y0 = (id0 - 1) % ly + 1
	x1 = (id1 - 1) ÷ ly + 1
	y1 = (id1 - 1) % ly + 1
	bc = 1.0
	# 跨边界判定： |dx| >= lx/2
	if abs(x0 - x1) >= lx / 2
		bc *= bcx
	end
	# 跨边界判定： |dy| >= ly/2
	if abs(y0 - y1) >= ly / 2
		bc *= bcy
	end
	return bc
end

"""
	freeze_hamiltonian(tmat::AbstractMatrix, defect_index::Vector{Int})

冻结 defect 位置, 返回删除 defect 行列后的矩阵.
参数:
- tmat::AbstractMatrix, 原始哈密顿量矩阵 (2N x 2N).
- defect_index::Vector{Int}, defect 的 1-based site index.
返回:
- Matrix{Float64}, 去掉 defect 后的矩阵.
"""
function freeze_hamiltonian(tmat::AbstractMatrix, defect_index::Vector{Int})
	n_full = size(tmat, 1)
	mask = trues(n_full)
	for id0 in defect_index
		idx = (id0 - 1) * 2 + 1
		mask[idx] = false
		mask[idx+1] = false
	end
	return tmat[mask, mask]
end

"""
	construct_defect_ph_tmat(p::DefectHeisenbergParams; mu, chi1, etad1, etas1, mz, chi2, etad2, etas2, chi3, etad3, etas3)

构造 defect 情况下的 PH 哈密顿量矩阵.
参数:
- p::DefectHeisenbergParams, 模型参数.
- 其余为覆盖参数, 用于构造导数矩阵.
返回:
- Matrix{Float64}, 冻结 defect 后的矩阵.
"""
function construct_defect_ph_tmat(
	p::DefectHeisenbergParams;
	mu::Float64 = p.mu,
	chi1::Float64 = p.chi1,
	etad1::Float64 = p.etad1,
	etas1::Float64 = p.etas1,
	mz::Dict{Symbol, Float64} = p.mz,
	chi2::Dict{Symbol, Float64} = p.chi2,
	etad2::Dict{Symbol, Float64} = p.etad2,
	etas2::Dict{Symbol, Float64} = p.etas2,
	chi3::Dict{Symbol, Float64} = p.chi3,
	etad3::Dict{Symbol, Float64} = p.etad3,
	etas3::Dict{Symbol, Float64} = p.etas3,
)
	lx = p.Lx
	ly = p.Ly
	n_lat = lx * ly
	tmat = zeros(Float64, 2 * n_lat, 2 * n_lat)

	for x in 1:lx, y in 1:ly
		id0 = xy_to_id_1based(x, y, lx, ly)
		idpx = xy_to_id_1based(x + 1, y, lx, ly)
		idpy = xy_to_id_1based(x, y + 1, lx, ly)

		mz0 = get(mz, Symbol("mz_$(x)_$(y)"), 0.0)

		idx = (id0 - 1) * 2
		tmat[idx+1, idx+1] += mu / 2 + mz0 / 2
		tmat[idx+2, idx+2] += -mu / 2 + mz0 / 2

		if x == lx
			add_term_ij_PH(tmat, id0, idpx, chi1 * p.bcx, (etas1 - etad1) * p.bcx)
		else
			add_term_ij_PH(tmat, id0, idpx, chi1, etas1 - etad1)
		end

		if y == ly
			add_term_ij_PH(tmat, id0, idpy, chi1 * p.bcy, (etas1 + etad1) * p.bcy)
		else
			add_term_ij_PH(tmat, id0, idpy, chi1, etas1 + etad1)
		end
	end

	for (x, y) in p.defect_positions
		idpx = xy_to_id_1based(x + 1, y, lx, ly)
		idpy = xy_to_id_1based(x, y + 1, lx, ly)
		idmx = xy_to_id_1based(x - 1, y, lx, ly)
		idmy = xy_to_id_1based(x, y - 1, lx, ly)

		chi20 = get(chi2, Symbol("chi2_$(x)_$(y)"), 0.0)
		etad20 = get(etad2, Symbol("etad2_$(x)_$(y)"), 0.0)
		etas20 = get(etas2, Symbol("etas2_$(x)_$(y)"), 0.0)
		chi30 = get(chi3, Symbol("chi3_$(x)_$(y)"), 0.0)
		etad30 = get(etad3, Symbol("etad3_$(x)_$(y)"), 0.0)
		etas30 = get(etas3, Symbol("etas3_$(x)_$(y)"), 0.0)

		add_term_ij_PH(
			tmat,
			idpx,
			idpy,
			chi20 * calc_bc(lx, ly, p.bcx, p.bcy, idpx, idpy),
			(etas20 + etad20) * calc_bc(lx, ly, p.bcx, p.bcy, idpx, idpy),
		)
		add_term_ij_PH(
			tmat,
			idpx,
			idmy,
			chi20 * calc_bc(lx, ly, p.bcx, p.bcy, idpx, idmy),
			(etas20 - etad20) * calc_bc(lx, ly, p.bcx, p.bcy, idpx, idmy),
		)
		add_term_ij_PH(
			tmat,
			idmx,
			idpy,
			chi20 * calc_bc(lx, ly, p.bcx, p.bcy, idmx, idpy),
			(etas20 - etad20) * calc_bc(lx, ly, p.bcx, p.bcy, idmx, idpy),
		)
		add_term_ij_PH(
			tmat,
			idmx,
			idmy,
			chi20 * calc_bc(lx, ly, p.bcx, p.bcy, idmx, idmy),
			(etas20 + etad20) * calc_bc(lx, ly, p.bcx, p.bcy, idmx, idmy),
		)
		add_term_ij_PH(
			tmat,
			idpx,
			idmx,
			chi30 * calc_bc(lx, ly, p.bcx, p.bcy, idpx, idmx),
			(etas30 + etad30) * calc_bc(lx, ly, p.bcx, p.bcy, idpx, idmx),
		)
		add_term_ij_PH(
			tmat,
			idpy,
			idmy,
			chi30 * calc_bc(lx, ly, p.bcx, p.bcy, idpy, idmy),
			(etas30 - etad30) * calc_bc(lx, ly, p.bcx, p.bcy, idpy, idmy),
		)
	end

	tmat = Hermitian(tmat + tmat')
	return freeze_hamiltonian(Matrix(tmat), p.defect_index)
end

"""
	build_defect_dh_dparam(p::DefectHeisenbergParams, name::Symbol) -> Matrix{Float64}

用途: 构造 defect ansatz 中单个参数 `name` 对应的导数矩阵 `dH/dname`.
参数:
- p::DefectHeisenbergParams, 当前 defect 参数.
- name::Symbol, 目标参数名.
返回:
- Matrix{Float64}, 与 `construct_defect_ph_tmat(p)` 同维度的导数矩阵.
说明:
- 对于形如 `mz_x_y`、`chi2_x_y` 的 site 依赖参数, 只在对应坐标位置置 1.0.
- 对于全局参数 (如 `mu`、`etad1`、`etas1`), 直接将该参数置 1.0.
"""
function build_defect_dh_dparam(
	p::DefectHeisenbergParams,
	name::Symbol
)::Matrix{Float64}
	if occursin(r"_\d+_\d+$", String(name))
		idx = findfirst('_', String(name))
		str = String(name)[1:(idx-1)]
		p_alpha = DefectHeisenbergParams(;
			(; :Lx => p.Lx,
				:Ly => p.Ly,
				:bcx => p.bcx,
				:bcy => p.bcy,
				Symbol(str) => Dict(name => 1.0),
				:defect_positions => p.defect_positions,
				:defect_index => p.defect_index)...,
		)
	else
		p_alpha = DefectHeisenbergParams(;
			(; :Lx => p.Lx,
				:Ly => p.Ly,
				:bcx => p.bcx,
				:bcy => p.bcy,
				name => 1.0,
				:defect_positions => p.defect_positions,
				:defect_index => p.defect_index)...,
		)
	end
	return construct_defect_ph_tmat(p_alpha)
end

"""
	make_defect_ansatz_and_derivs(p::DefectHeisenbergParams; param_names::Vector{Symbol}, target_sz::Int)

生成 defect 版波函数与导数 (串行版本).
参数:
- p::DefectHeisenbergParams, 模型参数.
- param_names::Vector{Symbol}, 需要求导的参数列表.
- target_sz::Int, 目标 total Sz.
返回:
- ε, gs_U, dUt_params (OrderedDict).
"""
function make_defect_ansatz_and_derivs(
	p::DefectHeisenbergParams;
	param_names::Vector{Symbol},
	target_sz::Int = 0,
)
	H = construct_defect_ph_tmat(p)
	H_alphas = OrderedDict{Symbol, Matrix{Float64}}()
	for name in param_names
		H_alphas[name] = build_defect_dh_dparam(p, name)
	end
	ε, U_full, dE, dU_dict = Utils.compute_eig_and_dU_reg1(H, H_alphas)
	eig_eq_error = norm(Matrix(H) * U_full - U_full * Diagonal(ε))
	if is_root_rank()
		println("Eigen equation error (HU - Uε): ", eig_eq_error)
	end

	n_lat = size(H, 1) ÷ 2
	n_occ = n_lat + target_sz
	if is_root_rank()
		left_idx = max(1, n_occ - 4)
		right_idx = min(length(ε), n_occ + 4)
		println("ε is", ε[left_idx:right_idx])
	end

	U_occ = U_full[:, 1:n_occ]
	dUt_occ = OrderedDict(
		alpha => permutedims(real.(dU_dict[alpha][:, 1:n_occ]))
		for alpha in param_names
	)
	return ε, real.(U_occ), dUt_occ
end

"""
	make_defect_ansatz_and_derivs_MPI(
		p::DefectHeisenbergParams;
		param_names::Vector{Symbol},
		target_sz::Int,
		world_comm::MPI.Comm,
		leaders_comm::Union{Nothing,MPI.Comm},
		is_shared_root::Bool
	)

用途: defect ansatz 的 MPI 导数版本.
参数:
- p::DefectHeisenbergParams, 模型参数.
- param_names::Vector{Symbol}, 需要求导的参数列表.
- target_sz::Int, 目标 total Sz.
- world_comm::MPI.Comm, 全局 communicator.
- leaders_comm::Union{Nothing,MPI.Comm}, 仅共享内存 root 组成的 communicator.
- is_shared_root::Bool, 当前 rank 是否为本节点共享内存 root.
返回:
- ε, gs_U, dUt_params.
  - ε: 本征值向量.
  - gs_U: 占据轨道矩阵.
  - dUt_params: 仅在 `is_shared_root=true` 的 rank 上返回完整 `OrderedDict`, 其它 rank 返回空字典.
公式:
- dU = U * (F .* (U' * dH * U)),
  其中 F_{mn} = -Re[1 / (ε_m - ε_n + i*eta)], 且 F_{mm}=0.
并行流程:
- 全局 root 对角化并广播 `ε` 与 `U`.
- 参数按 world rank 做 cyclic 分配.
- 各 rank 计算自己的 dUt 切片后, 用 `Reduce(SUM)` 仅汇总到全局 root.
- 全局 root 再向 leaders_comm 广播完整 dUt, 供每个节点共享内存 root 写入共享矩阵.
"""
function make_defect_ansatz_and_derivs_MPI(
	p::DefectHeisenbergParams;
	param_names::Vector{Symbol},
	target_sz::Int = 0,
	world_comm::MPI.Comm = MPI.COMM_WORLD,
	leaders_comm::Union{Nothing,MPI.Comm} = nothing,
	is_shared_root::Bool = true,
	eta::Real = 1e-8,
)
	if !MPI.Initialized() || MPI.Comm_size(world_comm) <= 1
		return make_defect_ansatz_and_derivs(p; param_names=param_names, target_sz=target_sz)
	end

	world_rank = MPI.Comm_rank(world_comm)
	world_size = MPI.Comm_size(world_comm)
	world_root = 0

	H_root = nothing
	ε = nothing
	U_full = nothing
	if world_rank == world_root
		H_root = construct_defect_ph_tmat(p)
		H_sym = Hermitian(H_root)
		ε, U_full = eigen(H_sym)
	end
	ε = MPI.bcast(ε, world_root, world_comm)
	U_full = MPI.bcast(U_full, world_root, world_comm)

	diff_mat = ε .- ε' .+ im * eta
	F = -real.(1.0 ./ diff_mat)
	F[diagind(F)] .= 0.0

	n_spin = size(U_full, 1)
	n_lat = n_spin ÷ 2
	n_occ = n_lat + target_sz
	n_param = length(param_names)

	local_dut_tensor = zeros(Float64, n_occ, n_spin, n_param)
	for alpha_idx in (world_rank+1):world_size:n_param
		alpha_name = param_names[alpha_idx]
		dH_alpha = build_defect_dh_dparam(p, alpha_name)
		dH_mo = U_full' * dH_alpha * U_full
		dU_alpha = U_full * (F .* dH_mo)
		local_dut_tensor[:, :, alpha_idx] .= permutedims(real.(dU_alpha[:, 1:n_occ]))
	end

	global_dut_tensor = world_rank == world_root ? similar(local_dut_tensor) : local_dut_tensor
	MPI.Reduce!(local_dut_tensor, global_dut_tensor, MPI.SUM, world_root, world_comm)

	if world_rank == world_root
		eig_eq_error = norm(Matrix(H_root) * U_full - U_full * Diagonal(ε))
		println("Eigen equation error (HU - Uε): ", eig_eq_error)
		left_idx = max(1, n_occ - 4)
		right_idx = min(length(ε), n_occ + 4)
		println("ε is", ε[left_idx:right_idx])
	end

	shared_dut_tensor = nothing
	if is_shared_root
		if leaders_comm === nothing
			if world_rank != world_root
				error("leaders_comm is required for non-root shared leaders in MPI mode.")
			end
			shared_dut_tensor = global_dut_tensor
		else
			leaders_root = 0
			if world_rank == world_root
				shared_dut_tensor = global_dut_tensor
			end
			shared_dut_tensor = MPI.bcast(shared_dut_tensor, leaders_root, leaders_comm)
		end
	end

	U_occ = U_full[:, 1:n_occ]
	dUt_occ = OrderedDict{Symbol, Matrix{Float64}}()
	if is_shared_root
		for (alpha_idx, alpha_name) in enumerate(param_names)
			dUt_occ[alpha_name] = copy(@view(shared_dut_tensor[:, :, alpha_idx]))
		end
	end
	return ε, real.(U_occ), dUt_occ
end
# ======================================================================
# Defect Heisenberg (PH, determinant)
# ======================================================================
struct HubbardParams
	Lx::Int
	Ly::Int
	bcx::Float64
	bcy::Float64
	chi1::Float64
	etad1::Float64
	etas1::Float64
	mu::Dict{Symbol, Float64}
	mz::Dict{Symbol, Float64}
end
function HubbardParams(;
	Lx::Int,
	Ly::Int,
	bcx::Float64 = 1.0,
	bcy::Float64 = 1.0,
	chi1::Float64 = 0.0,
	etad1::Float64 = 0.0,
	etas1::Float64 = 0.0,
	mu::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
	mz::Dict{Symbol, Float64} = Dict{Symbol, Float64}()
)
	return HubbardParams(Lx, Ly, bcx, bcy, chi1, etad1, etas1, mu, mz)
end
function build_ham_PH(p::HubbardParams)
	Lx, Ly = p.Lx, p.Ly
	Nlat = Lx * Ly
	chi1 = p.chi1
	etad1 = p.etad1
	etas1 = p.etas1
	mz = p.mz
	mu = p.mu
	H = zeros(Float64, 2 * Nlat, 2 * Nlat)
	for x in 1:Lx
		mz0 = get(mz, Symbol("mz_$(x)"), 0.0)
		mu0 = get(mu, Symbol("mu_$(x)"), 0.0)
		for y in 1:Ly
			id0 = xy_to_idx(x, y, Ly)
			Q = (-1)^(x + y)
			# --- Y 方向 ---
			idy = (y == Ly) ? xy_to_idx(x, 1, Ly) : xy_to_idx(x, y + 1, Ly)
			bc_y = (y == Ly) ? p.bcy : 1.0
			# --- X 方向 ---
			idx = (x == Lx) ? xy_to_idx(1, y, Ly) : xy_to_idx(x + 1, y, Ly)
			bc_x = (x == Lx) ? p.bcx : 1.0
			add_term_ij_PH(H, id0, idx, chi1 * bc_x, (+etas1 - etad1) * bc_x)
			add_term_ij_PH(H, id0, idy, chi1 * bc_y, (etas1 + etad1) * bc_y)
			H[2*(id0-1)+1, 2*(id0-1)+1] += Q * mz0 / 2 + mu0 / 2
			H[2*(id0-1)+2, 2*(id0-1)+2] += Q * mz0 / 2 - mu0 / 2
		end
	end

	H = Hermitian(H + H')
	return H
end
function make_ansatz_and_derivs(p::HubbardParams; param_names::Vector{Symbol} = [:etad1, :etas1, :mz], target_sz::Int = 0)
	H = build_ham_PH(p)

	H_alphas = OrderedDict{Symbol, Matrix{Float64}}()
	for name in param_names
		if occursin(r"_\d+$", String(name))
			idx = findfirst('_', String(name))
			str = String(name)[1:(idx-1)]
			p_alpha = HubbardParams(;
				(; :Lx => p.Lx,
					:Ly => p.Ly,
					:bcx => p.bcx,
					:bcy => p.bcy,
					Symbol(str) => Dict(name => 1.0))...,
			)
		else
			p_alpha = HubbardParams(;
				(; :Lx => p.Lx,
					:Ly => p.Ly,
					:bcx => p.bcx,
					:bcy => p.bcy,
					name => 1.0)...,
			)
		end
		H_alphas[name] = build_ham_PH(p_alpha)
	end

	# 2. 对角化并计算导数 (Utils)
	ε, U_full, dE, dU_dict = Utils.compute_eig_and_dU_reg1(H, H_alphas)
	eig_eq_error = norm(Matrix(H) * U_full - U_full * Diagonal(ε))
	if is_root_rank()
		println("Eigen equation error (HU - Uε): ", eig_eq_error)
	end
	#做了PH变换后粒子数不再守恒，守恒的只有total Sz，根据输入的target_sz截取
	# 3. 截取占据态并封装
	Nlat = p.Lx * p.Ly
	n_occ = Nlat + target_sz
	if is_root_rank()
		println("ε is", ε[(n_occ-4):(n_occ+4)])
	end
	U_occ = U_full[:, 1:n_occ]
	dUt_occ = OrderedDict(alpha => permutedims(real.(dU_dict[alpha][:, 1:n_occ])) for alpha in param_names)
	return ε, U_occ, dUt_occ
end
end
# module
