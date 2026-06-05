#!/bin/bash
# Defect three-band Hubbard VMC/SR submission script for the root mfVMC code.
# This script targets ./DefectThreeBandHubbard.jl and only passes arguments
# supported by the current root driver.

#SBATCH --job-name=defect_threeband_vmc
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --partition=v6_384
#SBATCH --output=slurm_out/%x_%j.log
#SBATCH --error=slurm_out/%x_%j.err

set -euo pipefail

export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE
export PATH="${HOME}/.julia/bin:${PATH}"

# Submit from the root mfVMC directory, or set CODE_DIR explicitly before sbatch.
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_PROJECT="${JULIA_PROJECT:-@v1.11}"
MPIEXECJL="${MPIEXECJL:-mpiexecjl}"
NTASKS="${SLURM_NTASKS:-1}"

mkdir -p "${CODE_DIR}/logs"
mkdir -p "${CODE_DIR}/slurm_out"

# Lattice, boundary conditions, and filling.
LX="${LX:-12}"
LY="${LY:-12}"
BCX="${BCX:-0.99}"
BCY="${BCY:-0.98}"

# If NELEC <= 0, DefectThreeBandHubbard.jl uses LX*LY - NHOLE.
NELEC="${NELEC:-156}"
NHOLE="${NHOLE:-0}"
TARGET_SZ="${TARGET_SZ:-0}"

# Clean three-band Hubbard Hamiltonian plus defect onsite energy shift.
TPD="${TPD:-1.1}"
TPP="${TPP:-0.55}"
DELTA_PD="${DELTA_PD:-3.3}"
UDD="${UDD:-8.8}"
UP="${UP:-0.0}"
VPD="${VPD:-0.0}"
DEFECT_EPP="${DEFECT_EPP:--2.0}"

# Clean background mean-field initial parameters.
CHI1_00="${CHI1_00:-0.0}"
CHI1_01="${CHI1_01:-1.0}"
CHI1_11="${CHI1_11:-0.38}"
MU0="${MU0:-0.0}"
MU1="${MU1:-2.0}"
UNIFORM_NONDEFECT_MU="${UNIFORM_NONDEFECT_MU:-false}"
SITE_RESOLVED_OXYGEN_MZ="${SITE_RESOLVED_OXYGEN_MZ:-false}"
MZ_00="${MZ_00:-0.3}"
MZ_11="${MZ_11:-0.0}"

# Defect anchors and defect-local mean-field defaults.
DEFECT_ANCHORS="${DEFECT_ANCHORS:-1,1;7,1;3,2;2,4;5,5;8,5;11,5;1,8;4,8;7,8;4,11;10,11}"
DFT_CHI1_00="${DFT_CHI1_00:-0.05}"
DFT_CHI1_01="${DFT_CHI1_01:-1.0}"
DFT_CHI1_11="${DFT_CHI1_11:-0.38}"
MU0_D0="${MU0_D0:-0.0}"
MU1_D0="${MU1_D0:-0.5}"
MZ_00_D0="${MZ_00_D0:-0.0}"
MZ_11_D0="${MZ_11_D0:-0.0}"

# Initial parameters listed here remain fixed during SR.
NOT_OPT_PARAMS="${NOT_OPT_PARAMS:-chi1_01,mz_11,mz_11_d0}"

# Gutzwiller and Jastrow projectors supported by the root driver.
GUTZWILLER_ORBITAL="${GUTZWILLER_ORBITAL:-true}"
G_D="${G_D:-0.9}"
G_PY="${G_PY:-0.0}"
G_PX="${G_PX:-0.0}"

SITE_GUTZWILLER="${SITE_GUTZWILLER:-true}"
G_SITE_INIT="${G_SITE_INIT:-${G_D}}"

JASTROW_SHELLS="${JASTROW_SHELLS:-1}"
JASTROW_INIT_FILE="${JASTROW_INIT_FILE:-}"
JASTROW_INIT="${JASTROW_INIT:-0.0}"

DEFECT_JASTROW="${DEFECT_JASTROW:-false}"
DEFECT_JASTROW_INIT="${DEFECT_JASTROW_INIT:-0.0}"

# SR optimizer controls.
NSR="${NSR:-300}"
LR="${LR:-0.04}"
LR_END="${LR_END:-0.02}"
DIAG_SHIFT="${DIAG_SHIFT:-1e-3}"
EPS_WF="${EPS_WF:-1e-4}"
MAX_STEP_SIZE="${MAX_STEP_SIZE:-0.1}"
NUMA_TENSOR_REPLICA="${NUMA_TENSOR_REPLICA:-true}"

# Monte Carlo controls. NMC is the global total samples per SR step.
NMC="${NMC:-48000}"
WMC="${WMC:-200}"
DMC="${DMC:-20}"
RMC="${RMC:-50}"
SEED="${SEED:-1234}"

# Optional restart from a previous full parameter JSON or two-column parameter txt.
INIT_PARAMS_JSON="${INIT_PARAMS_JSON:-}"
INIT_PARAMS_TXT="${INIT_PARAMS_TXT:-}"

if [[ -n "${INIT_PARAMS_JSON}" && -n "${INIT_PARAMS_TXT}" ]]; then
    echo "ERROR: set only one of INIT_PARAMS_JSON or INIT_PARAMS_TXT." >&2
    exit 1
fi

JOB="${JOB:-SR}"
JOB_LABEL="${JOB_LABEL:-defect_L${LX}x${LY}_Nh${NHOLE}_U${UDD}_Up${UP}_Vpd${VPD}_J${JASTROW_SHELLS}}"
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${CODE_DIR}/logs/${JOB_LABEL}_${RUN_ID}}"
mkdir -p "${RUN_DIR}"

echo "Code directory:            ${CODE_DIR}"
echo "Run directory:             ${RUN_DIR}"
echo "Job label:                 ${JOB_LABEL}"
echo "Julia project:             ${JULIA_PROJECT}"
echo "MPI tasks:                 ${NTASKS}"
echo "Defect anchors:            ${DEFECT_ANCHORS}"
echo "Init params json:          ${INIT_PARAMS_JSON}"
echo "Init params txt:           ${INIT_PARAMS_TXT}"
echo "Fixed params:              ${NOT_OPT_PARAMS}"
echo "Uniform non-defect mu:     ${UNIFORM_NONDEFECT_MU}"
echo "Site-resolved oxygen mz:   ${SITE_RESOLVED_OXYGEN_MZ}"
echo "NUMA tensor replica:       ${NUMA_TENSOR_REPLICA}"
echo "Job:                       ${JOB}"

cd "${RUN_DIR}"

"${MPIEXECJL}" -n "${NTASKS}" "${JULIA_BIN}" --project="${JULIA_PROJECT}" "${CODE_DIR}/DefectThreeBandHubbard.jl" \
  --Lx "${LX}" --Ly "${LY}" \
  --bcx "${BCX}" --bcy "${BCY}" \
  --tpd "${TPD}" --tpp "${TPP}" \
  --Delta_pd "${DELTA_PD}" \
  --Udd "${UDD}" --Up "${UP}" --Vpd "${VPD}" \
  --defect_anchors "${DEFECT_ANCHORS}" \
  --defect_Epp "${DEFECT_EPP}" \
  --nelec "${NELEC}" --Nhole "${NHOLE}" \
  --target_sz "${TARGET_SZ}" \
  --chi1_00 "${CHI1_00}" --chi1_01 "${CHI1_01}" --chi1_11 "${CHI1_11}" \
  --chi_def_dd_init "${DFT_CHI1_00}" \
  --chi_def_pd_init "${DFT_CHI1_01}" \
  --chi_def_pp_init "${DFT_CHI1_11}" \
  --mu0 "${MU0}" --mu1 "${MU1}" \
  --uniform_nondefect_mu "${UNIFORM_NONDEFECT_MU}" \
  --site_resolved_oxygen_mz "${SITE_RESOLVED_OXYGEN_MZ}" \
  --mu0_d0 "${MU0_D0}" --mu1_d0 "${MU1_D0}" \
  --mz_00 "${MZ_00}" --mz_11 "${MZ_11}" \
  --mz_00_d0 "${MZ_00_D0}" --mz_11_d0 "${MZ_11_D0}" \
  --gutzwiller_orbital "${GUTZWILLER_ORBITAL}" \
  --g_d "${G_D}" --g_py "${G_PY}" --g_px "${G_PX}" \
  --site_gutzwiller "${SITE_GUTZWILLER}" \
  --g_site_init "${G_SITE_INIT}" \
  --jastrow_shells "${JASTROW_SHELLS}" \
  --jastrow_init_file "${JASTROW_INIT_FILE}" \
  --jastrow_init "${JASTROW_INIT}" \
  --defect_jastrow "${DEFECT_JASTROW}" \
  --defect_jastrow_init "${DEFECT_JASTROW_INIT}" \
  --nSR "${NSR}" \
  --lr "${LR}" --lr_end "${LR_END}" \
  --diag_shift "${DIAG_SHIFT}" \
  --eps_wf "${EPS_WF}" \
  --max_step_size "${MAX_STEP_SIZE}" \
  --numa_tensor_replica "${NUMA_TENSOR_REPLICA}" \
  --not_opt_params "${NOT_OPT_PARAMS}" \
  --nMC "${NMC}" --wMC "${WMC}" --dMC "${DMC}" --rMC "${RMC}" \
  --seed "${SEED}" \
  --init_params_json "${INIT_PARAMS_JSON}" \
  --init_params_txt "${INIT_PARAMS_TXT}" \
  --job "${JOB}"
