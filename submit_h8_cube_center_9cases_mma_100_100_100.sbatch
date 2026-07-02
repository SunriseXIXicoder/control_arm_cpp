#!/bin/bash
# H8 cube-center bracket benchmark, 100x100x100 elements.
# Array cases:
#   0:no draft  1:+x  2:-x  3:+y  4:-y  5:+z  6:-z  7:+z/-z  8:six-axis
#
# Matches the tip-center cantilever 9-case MMA settings unless noted:
#   MMA, penal=3, filter radius=5, HYPRE xstrong, final VTK only.

#SBATCH -J h8_cube100_9
#SBATCH -N 8
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --exclusive
#SBATCH --array=0-8%9
#SBATCH --output=result/%A_%a.%x.out
#SBATCH --error=result/%A_%a.%x.err

set -euo pipefail
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "Job failed at line ${LINENO:-unknown} with exit code ${code}" >&2; fi' EXIT

if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
  [ -f /usr/share/Modules/init/bash ] && . /usr/share/Modules/init/bash || true
fi
if command -v module >/dev/null 2>&1; then
  module load gcc/11.3.0 2>/dev/null || true
  module load intel/2020u4 2>/dev/null || true
fi

PETSC_DIR="${PETSC_DIR:-/data/home/dlut_ycx/petsc/petsc-v3.19.3}"
PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-opt-hypre}"
export PETSC_DIR PETSC_ARCH
export PATH="${PETSC_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export I_MPI_HYDRA_BOOTSTRAP="${I_MPI_HYDRA_BOOTSTRAP:-slurm}"

CASE_ID="${CASE_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
RUN_NAMES=(no_draft plus_x minus_x plus_y minus_y plus_z minus_z plus_minus_z six_axis)
DRAFT_FLAGS=(false true true true true true true true true)
DRAFT_AXES_LIST=("" "px" "mx" "py" "my" "pz" "mz" "pz,mz" "px,mx,py,my,pz,mz")
H8_DM_PX_LIST=(8 7 7 8 8 8 8 8 8)
H8_DM_PY_LIST=(4 4 4 4 4 7 7 7 4)
H8_DM_PZ_LIST=(7 8 8 7 7 4 4 4 7)

if [ "${CASE_ID}" -lt 0 ] || [ "${CASE_ID}" -ge "${#RUN_NAMES[@]}" ]; then
  echo "Invalid CASE_ID=${CASE_ID}; expected 0..8." >&2
  exit 2
fi

RUN_NAME="${RUN_NAMES[$CASE_ID]}"
OPT_DRAFT_CLOSURE="${DRAFT_FLAGS[$CASE_ID]}"
OPT_DRAFT_AXES="${DRAFT_AXES_LIST[$CASE_ID]}"

NELX="${NELX:-100}"
NELY="${NELY:-100}"
NELZ="${NELZ:-100}"
NX="${NX:-$((NELX + 1))}"
NY="${NY:-$((NELY + 1))}"
NZ="${NZ:-$((NELZ + 1))}"

BENCHMARK_CASE="${BENCHMARK_CASE:-cube_center}"
DOMAIN_HEIGHT="${DOMAIN_HEIGHT:-0.08}"
VOLFRAC="${VOLFRAC:-0.20}"
OPT_MAX_ITER="${OPT_MAX_ITER:-500}"
OPT_FILTER_RADIUS="${OPT_FILTER_RADIUS:-5}"
OPT_MOVE="${OPT_MOVE:-0.005}"
OPT_PROJECTED_VOLUME_CORRECTION="${OPT_PROJECTED_VOLUME_CORRECTION:-true}"
OPT_DRAFT_ETA="${OPT_DRAFT_ETA:-0.5}"
OPT_MATLAB_Z_PROJECTION="${OPT_MATLAB_Z_PROJECTION:-false}"
OPT_USE_MMA="${OPT_USE_MMA:-true}"
OPT_HEAVISIDE="${OPT_HEAVISIDE:-true}"
OPT_HEAVISIDE_ETA="${OPT_HEAVISIDE_ETA:-0.50}"
OPT_HEAVISIDE_BETA_INITIAL="${OPT_HEAVISIDE_BETA_INITIAL:-1.0}"
OPT_HEAVISIDE_BETA_MAX="${OPT_HEAVISIDE_BETA_MAX:-16.0}"
OPT_HEAVISIDE_BETA_INTERVAL="${OPT_HEAVISIDE_BETA_INTERVAL:-50}"
OPT_RHO_MIN="${OPT_RHO_MIN:-1e-3}"
PENAL="${PENAL:-3.0}"
EMIN="${EMIN:-1e-9}"
VOID_DENSITY="${VOID_DENSITY:-0.02}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
OPT_LOAD="${OPT_LOAD:-1}"
KSP_RTOL="${KSP_RTOL:-1e-8}"
KSP_MAX_IT="${KSP_MAX_IT:-1000}"
KSP_TYPE="${KSP_TYPE:-fgmres}"
H8_PC_TYPE="${H8_PC_TYPE:-aux_elastic_hypre}"
H8_AUX_REBUILD_INTERVAL="${H8_AUX_REBUILD_INTERVAL:-1}"
H8_REUSE_INITIAL_GUESS="${H8_REUSE_INITIAL_GUESS:-true}"
OPT_STOP_ON_KSP_DIVERGENCE="${OPT_STOP_ON_KSP_DIVERGENCE:-true}"
OPT_WRITE_FINAL_VTK="${OPT_WRITE_FINAL_VTK:-true}"
OPT_VTK_MAX_POINTS="${OPT_VTK_MAX_POINTS:-5000000}"
OPT_WRITE_CHECKPOINT="${OPT_WRITE_CHECKPOINT:-false}"
OPT_CHECKPOINT_INTERVAL="${OPT_CHECKPOINT_INTERVAL:-0}"
PETSC_EXTRA_OPTIONS="${PETSC_EXTRA_OPTIONS:--ksp_gmres_restart 200 -ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_hypre_type boomeramg -pc_hypre_boomeramg_strong_threshold 0.7 -pc_hypre_boomeramg_nodal_coarsen 1 -pc_hypre_boomeramg_vec_interp_variant 3 -pc_hypre_boomeramg_interp_type ext+i -pc_hypre_boomeramg_coarsen_type PMIS -pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi -pc_hypre_boomeramg_max_iter 2}"
CXXSTD="${CXXSTD:-c++11}"

JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-manual}}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
NODES_ALLOC="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-8}}"
TASKS_PER_NODE_ALLOC="${SLURM_NTASKS_PER_NODE:-28}"
TOTAL_TASKS="${SLURM_NPROCS:-${SLURM_NTASKS:-$((NODES_ALLOC * TASKS_PER_NODE_ALLOC))}}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_LAUNCHER_ARGS="${MPI_LAUNCHER_ARGS:-}"

H8_DM_PX="${H8_DM_PX:-${H8_DM_PX_LIST[$CASE_ID]}}"
H8_DM_PY="${H8_DM_PY:-${H8_DM_PY_LIST[$CASE_ID]}}"
H8_DM_PZ="${H8_DM_PZ:-${H8_DM_PZ_LIST[$CASE_ID]}}"
if [ $((H8_DM_PX * H8_DM_PY * H8_DM_PZ)) -ne "${TOTAL_TASKS}" ]; then
  echo "H8_DM_PX*H8_DM_PY*H8_DM_PZ must equal TOTAL_TASKS." >&2
  echo "  H8_DM=${H8_DM_PX}x${H8_DM_PY}x${H8_DM_PZ}, TOTAL_TASKS=${TOTAL_TASKS}" >&2
  exit 2
fi

cd "${SUBMIT_DIR}"
mkdir -p bin result logs
CASE_DIR="${OUTPUT_DIR:-result/h8_${BENCHMARK_CASE}_mma9_${NELX}_${NELY}_${NELZ}_${RUN_NAME}_${JOB_ID}}"
PREFIX="${OUTPUT_PREFIX:-${CASE_DIR}/${RUN_NAME}}"
mkdir -p "${CASE_DIR}"

echo "Start time: $(date)"
echo "Submit dir: ${SUBMIT_DIR}"
echo "Run host: $(hostname)"
echo "Job id: ${JOB_ID}"
echo "Array case: ${CASE_ID} ${RUN_NAME}"
echo "benchmark_case=${BENCHMARK_CASE}"
echo "draft_closure=${OPT_DRAFT_CLOSURE}"
echo "draft_axes=${OPT_DRAFT_AXES}"
echo "elements=${NELX}x${NELY}x${NELZ}"
echo "nodes=${NX}x${NY}x${NZ}"
echo "ranks=${TOTAL_TASKS}, dm=${H8_DM_PX}x${H8_DM_PY}x${H8_DM_PZ}"
echo "KSP_RTOL=${KSP_RTOL}"
echo "output_prefix=${PREFIX}"

make -B all CXXSTD="${CXXSTD}" PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}"

if [ "${MPI_LAUNCHER}" = "srun" ]; then
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -n ${TOTAL_TASKS}"
else
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -np ${TOTAL_TASKS}"
fi

${LAUNCH_CMD} ./bin/control_arm_cpp \
  -mode optimize \
  -operator h8_matrix_free \
  -control_arm_mask false \
  -benchmark_case "${BENCHMARK_CASE}" \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -domain_height "${DOMAIN_HEIGHT}" \
  -void_density "${VOID_DENSITY}" \
  -young_modulus "${YOUNG_MODULUS}" \
  -penal "${PENAL}" \
  -emin "${EMIN}" \
  -volfrac "${VOLFRAC}" \
  -opt_max_iter "${OPT_MAX_ITER}" \
  -opt_move "${OPT_MOVE}" \
  -opt_projected_volume_correction "${OPT_PROJECTED_VOLUME_CORRECTION}" \
  -opt_filter_radius "${OPT_FILTER_RADIUS}" \
  -opt_draft_closure "${OPT_DRAFT_CLOSURE}" \
  -opt_draft_axes "${OPT_DRAFT_AXES}" \
  -opt_draft_eta "${OPT_DRAFT_ETA}" \
  -opt_matlab_z_projection "${OPT_MATLAB_Z_PROJECTION}" \
  -opt_use_mma "${OPT_USE_MMA}" \
  -opt_rho_min "${OPT_RHO_MIN}" \
  -opt_heaviside_projection "${OPT_HEAVISIDE}" \
  -opt_heaviside_eta "${OPT_HEAVISIDE_ETA}" \
  -opt_heaviside_beta_initial "${OPT_HEAVISIDE_BETA_INITIAL}" \
  -opt_heaviside_beta_max "${OPT_HEAVISIDE_BETA_MAX}" \
  -opt_heaviside_beta_interval "${OPT_HEAVISIDE_BETA_INTERVAL}" \
  -opt_load "${OPT_LOAD}" \
  -opt_ksp_rtol "${KSP_RTOL}" \
  -opt_ksp_max_it "${KSP_MAX_IT}" \
  -opt_write_final_vtk "${OPT_WRITE_FINAL_VTK}" \
  -opt_vtk_max_points "${OPT_VTK_MAX_POINTS}" \
  -opt_write_checkpoint "${OPT_WRITE_CHECKPOINT}" \
  -opt_checkpoint_interval "${OPT_CHECKPOINT_INTERVAL}" \
  -opt_checkpoint_prefix "${PREFIX}_checkpoint" \
  -output_prefix "${PREFIX}" \
  -ksp_type "${KSP_TYPE}" \
  -h8_pc_type "${H8_PC_TYPE}" \
  -h8_aux_rebuild_interval "${H8_AUX_REBUILD_INTERVAL}" \
  -h8_reuse_initial_guess "${H8_REUSE_INITIAL_GUESS}" \
  -opt_stop_on_ksp_divergence "${OPT_STOP_ON_KSP_DIVERGENCE}" \
  -ksp_converged_reason \
  -h8_dm_px "${H8_DM_PX}" \
  -h8_dm_py "${H8_DM_PY}" \
  -h8_dm_pz "${H8_DM_PZ}" \
  ${PETSC_EXTRA_OPTIONS} \
  -log_view ":${PREFIX}_petsc.log"

echo "Finished at: $(date)"
