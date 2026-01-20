module mfVMC

include("Sampler.jl")
using .Sampler




# ... existing exports ...
export config_Hubbard, config_Heisenberg, MoveProposal, copy_config
export total_elec, ifPH
export init_config_Heisenberg!, initialize_lists!, init_config_Hubbard!, init_config_by_state_char, init_config_rand!
export propose_move, commit_move!, update_site_config!
export build_exchange, build_single_hop, build_spin_flip, build_spin_flip_hop, build_double_spin_flip
export can_hop, can_exchange, can_flip
export get_state_char
export check_consistency, print_config_debug, print_config, print_state
export HOLE, UP, DN, DB
export AbstractMCMCKernel, HubbardKernel, HeisenbergKernel
export count_choices, count_choices_reserve

# ------------------
include("Utils.jl")
using .Utils
using .Utils
export compute_eig_and_dU_reg1, expand_spatial_to_spinful, add_term_ij_PH

# 导出 VMC 模块的内容0
include("vmc_det.jl") # 确保这个文件包含更新后的代码
using .VMC  # 确保加载 VMC 模块
export vwf_det, VMCRunner, update_vwf_params!
export init_gswf!, mcmc_step!, calc_ham_eng, accept_move!, rebuild_inverse!
export measure_green, measure_SzSz, measure_SplusSminus, measure_SiSj, get_Sz, calc_ratio, compute_grad_log_psi!
export measure_total_Sz, measure_total_Hole, measure_total_Doublon
export measure_SxSx, measure_SplusSplus
export get_Sz, get_Hole, get_Doublon

# 导出 Model 模块的内容
include("Model.jl")
using .Model
export HeisenbergModel, HubbardModel, local_energy

include("MPI_VMC_Utils.jl")
using .MPI_VMC_Utils

export MPISession, init_mpi_session
export ObservableBuffer, register_scalar!, register_vector!, register_matrix!
export reset_buffers!, increment_counter!, accumulate_sample!, accumulate_sr_matrix!
export record_scalar!
export mpi_reduce_all, mpi_gather_scalar

include("Driver.jl")
using .Driver
export VMCParams, SRParams
export run_simulation, run_sr_optimization


end # module
