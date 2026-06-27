#!/bin/bash
#SBATCH -J ca_h8_100m_1step
#SBATCH -N 8
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --exclusive
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "Job failed at line ${LINENO:-unknown} with exit code ${code}" >&2; fi' EXIT
echo "Batch script entered shell: $(date)"

# 700 * 400 * 120 * 3 = 100,800,000 displacement DOF.
# This is a one-iteration production-scale smoke run:
# no VTK, checkpoint enabled, PETSc log enabled.

echo "Loading compiler/MPI environment"
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
  [ -f /usr/share/Modules/init/bash ] && . /usr/share/Modules/init/bash || true
fi
if command -v module >/dev/null 2>&1; then
  module load gcc/11.3.0 2>/dev/null || true
  module load intel/2020u4 2>/dev/null || true
else
  echo "Warning: module command is unavailable; relying on inherited PATH/LD_LIBRARY_PATH" >&2
fi
echo "Compiler/MPI environment checked"

PETSC_DIR="${PETSC_DIR:-/data/home/dlut_ycx/petsc/petsc-v3.19.3}"
PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-opt-hypre}"
export PETSC_DIR
export PETSC_ARCH
if [ ! -f "${PETSC_DIR}/lib/petsc/conf/variables" ]; then
  echo "PETSc configuration not found:" >&2
  echo "  PETSC_DIR=${PETSC_DIR}" >&2
  echo "  PETSC_ARCH=${PETSC_ARCH}" >&2
  echo "Expected file: ${PETSC_DIR}/lib/petsc/conf/variables" >&2
  exit 2
fi
export PATH="${PETSC_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -s unlimited || true
ulimit -l unlimited || true
echo "Runtime environment configured"

NX="${NX:-700}"
NY="${NY:-400}"
NZ="${NZ:-120}"
DOMAIN_HEIGHT="${DOMAIN_HEIGHT:-0.08}"
OPT_MAX_ITER="${OPT_MAX_ITER:-1}"
VOLFRAC="${VOLFRAC:-0.30}"
OPT_FILTER_RADIUS="${OPT_FILTER_RADIUS:-1.5}"
OPT_Z_DRAFT_CLOSURE="${OPT_Z_DRAFT_CLOSURE:-true}"
OPT_Z_DRAFT_ETA="${OPT_Z_DRAFT_ETA:-0.5}"
AB_TRIANGLE_RETRACT="${AB_TRIANGLE_RETRACT:-0.22}"
OPT_MOVE="${OPT_MOVE:-0.025}"
OPT_MOVE_MIN="${OPT_MOVE_MIN:-0.004}"
OPT_MOVE_SHRINK="${OPT_MOVE_SHRINK:-0.5}"
OPT_MOVE_GROWTH="${OPT_MOVE_GROWTH:-1.05}"
OPT_MAX_COMPLIANCE_INCREASE="${OPT_MAX_COMPLIANCE_INCREASE:-0.08}"
OPT_HEAVISIDE="${OPT_HEAVISIDE:-true}"
OPT_HEAVISIDE_ETA="${OPT_HEAVISIDE_ETA:-0.50}"
OPT_HEAVISIDE_BETA_INITIAL="${OPT_HEAVISIDE_BETA_INITIAL:-1.0}"
OPT_HEAVISIDE_BETA_MAX="${OPT_HEAVISIDE_BETA_MAX:-16.0}"
OPT_HEAVISIDE_BETA_INTERVAL="${OPT_HEAVISIDE_BETA_INTERVAL:-0}"
OPT_MATLAB_Z_PROJECTION="${OPT_MATLAB_Z_PROJECTION:-false}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
PENAL="${PENAL:-3.0}"
EMIN="${EMIN:-1e-6}"
VOID_DENSITY="${VOID_DENSITY:-0.02}"
OPT_RHO_MIN="${OPT_RHO_MIN:-1e-3}"
KSP_RTOL="${KSP_RTOL:-1e-2}"
KSP_MAX_IT="${KSP_MAX_IT:-1500}"
KSP_TYPE="${KSP_TYPE:-fgmres}"
H8_PC_TYPE="${H8_PC_TYPE:-aux_hypre}"
H8_REUSE_INITIAL_GUESS="${H8_REUSE_INITIAL_GUESS:-true}"
OPT_STOP_ON_KSP_DIVERGENCE="${OPT_STOP_ON_KSP_DIVERGENCE:-true}"
H8_LOAD_CASE="${H8_LOAD_CASE:-0}"
H8_INCLUDE_SPRING_LOAD="${H8_INCLUDE_SPRING_LOAD:-true}"
PETSC_EXTRA_OPTIONS="${PETSC_EXTRA_OPTIONS:--ksp_gmres_restart 200 -ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_hypre_type boomeramg -pc_hypre_boomeramg_strong_threshold 0.5}"
CXXSTD="${CXXSTD:-c++11}"
JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-manual}}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PREFIX="${OUTPUT_PREFIX:-result/h8_100m_1step_${NX}_${NY}_${NZ}_${JOB_ID}}"
NODES_ALLOC="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
TASKS_PER_NODE_ALLOC="${SLURM_NTASKS_PER_NODE:-${SLURM_CPUS_ON_NODE:-28}}"
TOTAL_TASKS="${SLURM_NPROCS:-${SLURM_NTASKS:-$((NODES_ALLOC * TASKS_PER_NODE_ALLOC))}}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_LAUNCHER_ARGS="${MPI_LAUNCHER_ARGS:-}"
echo "Slurm variables normalized"

H8_DM_ARGS=""
if [ -n "${H8_DM_PX:-}" ] || [ -n "${H8_DM_PY:-}" ] || [ -n "${H8_DM_PZ:-}" ]; then
  if [ -z "${H8_DM_PX:-}" ] || [ -z "${H8_DM_PY:-}" ] || [ -z "${H8_DM_PZ:-}" ]; then
    echo "Set H8_DM_PX, H8_DM_PY, and H8_DM_PZ together." >&2
    exit 2
  fi
  H8_DM_ARGS="-h8_dm_px ${H8_DM_PX} -h8_dm_py ${H8_DM_PY} -h8_dm_pz ${H8_DM_PZ}"
fi

cd "$SUBMIT_DIR"
mkdir -p bin result logs
echo "Start time: $(date)"
echo "Submit dir: ${SUBMIT_DIR}"
echo "Run host: $(hostname)"
echo "Job id: ${JOB_ID}"
echo "Slurm nodes: ${SLURM_JOB_NODELIST:-unknown}"
if [ -f scripts/h8_slurm_guard.sh ]; then
  . scripts/h8_slurm_guard.sh
  h8_print_slurm_diagnostics
  h8_guard_slurm_memory_accounting
  h8_guard_required_hypre_pc
  h8_guard_petsc_hypre
else
  echo "Warning: scripts/h8_slurm_guard.sh not found; Slurm memory guard skipped" >&2
fi
echo "PETSC_DIR=${PETSC_DIR}"
echo "PETSC_ARCH=${PETSC_ARCH:-<empty>}"
echo "CXXSTD=${CXXSTD}"
echo "Compiler wrapper:"
command -v mpicxx || true
mpicxx --version 2>&1 | head -n 3 || true
make clean || true
make -B all CXXSTD="${CXXSTD}" PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}"

echo "Running 100M-DOF H8 matrix-free one-step test"
echo "  nodes=${SLURM_JOB_NUM_NODES:-unknown}"
echo "  tasks=${TOTAL_TASKS}"
echo "  nx=${NX} ny=${NY} nz=${NZ}"
echo "  domain_height=${DOMAIN_HEIGHT} m"
echo "  dof=$((NX * NY * NZ * 3))"
echo "  opt_max_iter=${OPT_MAX_ITER}"
echo "  volfrac=${VOLFRAC}"
echo "  opt_filter_radius=${OPT_FILTER_RADIUS}"
echo "  ab_triangle_retract=${AB_TRIANGLE_RETRACT}"
echo "  heaviside=${OPT_HEAVISIDE} beta0=${OPT_HEAVISIDE_BETA_INITIAL} beta_max=${OPT_HEAVISIDE_BETA_MAX} beta_interval=${OPT_HEAVISIDE_BETA_INTERVAL}"
echo "  young_modulus=${YOUNG_MODULUS}"
echo "  penal=${PENAL}"
echo "  emin=${EMIN}"
echo "  void_density=${VOID_DENSITY}"
echo "  opt_rho_min=${OPT_RHO_MIN}"
echo "  ksp_rtol=${KSP_RTOL}"
echo "  ksp_max_it=${KSP_MAX_IT}"
echo "  ksp_type=${KSP_TYPE}"
echo "  h8_pc_type=${H8_PC_TYPE}"
echo "  opt_matlab_z_projection=${OPT_MATLAB_Z_PROJECTION}"
echo "  h8_reuse_initial_guess=${H8_REUSE_INITIAL_GUESS}"
echo "  opt_stop_on_ksp_divergence=${OPT_STOP_ON_KSP_DIVERGENCE}"
echo "  h8_load_case=${H8_LOAD_CASE}"
echo "  h8_include_spring_load=${H8_INCLUDE_SPRING_LOAD}"
echo "  output_prefix=${PREFIX}"
echo "  mpi_launcher=${MPI_LAUNCHER}"
echo "  petsc_extra_options=${PETSC_EXTRA_OPTIONS:-<none>}"
if [ "${MPI_LAUNCHER}" = "srun" ]; then
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -n ${TOTAL_TASKS}"
else
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -np ${TOTAL_TASKS}"
fi
echo "  launch_command=${LAUNCH_CMD} ./bin/control_arm_cpp ..."

${LAUNCH_CMD} ./bin/control_arm_cpp \
  -mode optimize \
  -operator h8_matrix_free \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -domain_height "${DOMAIN_HEIGHT}" \
  -control_arm_mask true \
  -void_density "${VOID_DENSITY}" \
  -young_modulus "${YOUNG_MODULUS}" \
  -penal "${PENAL}" \
  -emin "${EMIN}" \
  -mask_threshold 0.5 \
  -ab_triangle_retract "${AB_TRIANGLE_RETRACT}" \
  -volfrac "${VOLFRAC}" \
  -opt_max_iter "${OPT_MAX_ITER}" \
  -opt_move "${OPT_MOVE}" \
  -opt_move_min "${OPT_MOVE_MIN}" \
  -opt_move_shrink "${OPT_MOVE_SHRINK}" \
  -opt_move_growth "${OPT_MOVE_GROWTH}" \
  -opt_max_compliance_increase "${OPT_MAX_COMPLIANCE_INCREASE}" \
  -opt_stability_guard true \
  -opt_filter_radius "${OPT_FILTER_RADIUS}" \
  -opt_z_draft_closure "${OPT_Z_DRAFT_CLOSURE}" \
  -opt_z_draft_eta "${OPT_Z_DRAFT_ETA}" \
  -opt_rho_min "${OPT_RHO_MIN}" \
  -opt_heaviside_projection "${OPT_HEAVISIDE}" \
  -opt_heaviside_eta "${OPT_HEAVISIDE_ETA}" \
  -opt_heaviside_beta_initial "${OPT_HEAVISIDE_BETA_INITIAL}" \
  -opt_heaviside_beta_max "${OPT_HEAVISIDE_BETA_MAX}" \
  -opt_heaviside_beta_interval "${OPT_HEAVISIDE_BETA_INTERVAL}" \
  -opt_matlab_z_projection "${OPT_MATLAB_Z_PROJECTION}" \
  -opt_load 1.0 \
  -h8_load_case "${H8_LOAD_CASE}" \
  -h8_include_spring_load "${H8_INCLUDE_SPRING_LOAD}" \
  -opt_ksp_rtol "${KSP_RTOL}" \
  -opt_ksp_max_it "${KSP_MAX_IT}" \
  -opt_write_final_vtk false \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval 1 \
  -opt_checkpoint_prefix "${PREFIX}_checkpoint" \
  -output_prefix "${PREFIX}" \
  -ksp_type "${KSP_TYPE}" \
  -h8_pc_type "${H8_PC_TYPE}" \
  -h8_reuse_initial_guess "${H8_REUSE_INITIAL_GUESS}" \
  -opt_stop_on_ksp_divergence "${OPT_STOP_ON_KSP_DIVERGENCE}" \
  -ksp_rtol "${KSP_RTOL}" \
  -ksp_max_it "${KSP_MAX_IT}" \
  -ksp_converged_reason \
  ${H8_DM_ARGS} \
  ${PETSC_EXTRA_OPTIONS} \
  -log_view ":${PREFIX}_petsc.log"
