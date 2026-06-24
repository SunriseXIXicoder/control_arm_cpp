#!/bin/bash
# H8 小网格拔模方向验证：5 个 array 任务分别对应 +x、+y、+z/+x、+z/-z、+x/+y/+z。
# 默认使用无 mask 矩形悬臂梁，适合先看约束是否改变密度形态。
# 提交方式：sbatch submit_h8_draft_small_5cases.sbatch
# 默认网格为 81x31x21 节点，约 1.58e5 位移自由度；OPT_LOAD=1e5 将柔度放大到约 1e2 量级。
# 默认跑 500 步，Heaviside beta 每 50 步翻倍到 16；同一 beta 阶段内拒绝柔度上升。
# 常用覆盖示例：NX=121 NY=41 NZ=31 OPT_MAX_ITER=500 sbatch submit_h8_draft_small_5cases.sbatch
#SBATCH -J h8_draft5
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --array=0-4
#SBATCH --output=result/%A_%a.%x.out
#SBATCH --error=result/%A_%a.%x.err

set -euo pipefail
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "Job failed at line ${LINENO:-unknown} with exit code ${code}" >&2; fi' EXIT

echo "Batch script entered shell: $(date)"
echo "Script path: $0"
echo "Bash version: ${BASH_VERSION:-unknown}"

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

PETSC_DIR="${PETSC_DIR:-/data/home/dlut_ycx/petsc/petsc-v3.19.3}"
PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-opt-hypre}"
export PETSC_DIR PETSC_ARCH
if [ ! -f "${PETSC_DIR}/lib/petsc/conf/variables" ]; then
  echo "PETSc configuration not found:" >&2
  echo "  PETSC_DIR=${PETSC_DIR}" >&2
  echo "  PETSC_ARCH=${PETSC_ARCH}" >&2
  exit 2
fi
export PATH="${PETSC_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -s unlimited || true
ulimit -l unlimited || true

CASE_ID="${SLURM_ARRAY_TASK_ID:-0}"
DRAFT_AXES_LIST=("+x" "+y" "+z,+x" "+z,-z" "+x,+y,+z")
RUN_NAMES=("plus_x" "plus_y" "plus_z_plus_x" "split_z" "plus_x_plus_y_plus_z")
if [ "${CASE_ID}" -lt 0 ] || [ "${CASE_ID}" -ge "${#DRAFT_AXES_LIST[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${CASE_ID}; expected 0-4." >&2
  exit 2
fi
DRAFT_AXES="${DRAFT_AXES_LIST[$CASE_ID]}"
RUN_NAME="${RUN_NAMES[$CASE_ID]}"

NX="${NX:-81}"
NY="${NY:-31}"
NZ="${NZ:-21}"
DOMAIN_HEIGHT="${DOMAIN_HEIGHT:-0.08}"
BENCHMARK_CASE="${BENCHMARK_CASE:-cantilever}"
VOLFRAC="${VOLFRAC:-0.30}"
OPT_MAX_ITER="${OPT_MAX_ITER:-500}"
OPT_FILTER_RADIUS="${OPT_FILTER_RADIUS:-1.5}"
OPT_DRAFT_ETA="${OPT_DRAFT_ETA:-0.5}"
OPT_LOAD="${OPT_LOAD:-1e5}"
OPT_MOVE="${OPT_MOVE:-0.025}"
OPT_MOVE_MIN="${OPT_MOVE_MIN:-0.004}"
OPT_MOVE_SHRINK="${OPT_MOVE_SHRINK:-0.5}"
OPT_MOVE_GROWTH="${OPT_MOVE_GROWTH:-1.05}"
OPT_MAX_COMPLIANCE_INCREASE="${OPT_MAX_COMPLIANCE_INCREASE:-0.0}"
OPT_HEAVISIDE="${OPT_HEAVISIDE:-true}"
OPT_HEAVISIDE_ETA="${OPT_HEAVISIDE_ETA:-0.50}"
OPT_HEAVISIDE_BETA_INITIAL="${OPT_HEAVISIDE_BETA_INITIAL:-1.0}"
OPT_HEAVISIDE_BETA_MAX="${OPT_HEAVISIDE_BETA_MAX:-16.0}"
OPT_HEAVISIDE_BETA_INTERVAL="${OPT_HEAVISIDE_BETA_INTERVAL:-50}"
# H8 小算例使用平滑的闭包后体积修正：修正下一步 raw 体积分数，不做 trial 级闭包后二分。
OPT_PROJECTED_VOLUME_CORRECTION="${OPT_PROJECTED_VOLUME_CORRECTION:-true}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
PENAL="${PENAL:-3.0}"
EMIN="${EMIN:-1e-6}"
VOID_DENSITY="${VOID_DENSITY:-0.02}"
OPT_RHO_MIN="${OPT_RHO_MIN:-1e-3}"
KSP_RTOL="${KSP_RTOL:-1e-6}"
KSP_MAX_IT="${KSP_MAX_IT:-1000}"
KSP_TYPE="${KSP_TYPE:-fgmres}"
# 0 表示不写中间 checkpoint；代码仍会在最终步写 final checkpoint。
OPT_CHECKPOINT_INTERVAL="${OPT_CHECKPOINT_INTERVAL:-0}"
OPT_WRITE_FINAL_VTK="${OPT_WRITE_FINAL_VTK:-true}"
H8_PC_TYPE="${H8_PC_TYPE:-aux_hypre}"
H8_AUX_REBUILD_INTERVAL="${H8_AUX_REBUILD_INTERVAL:-1}"
H8_REUSE_INITIAL_GUESS="${H8_REUSE_INITIAL_GUESS:-true}"
OPT_STOP_ON_KSP_DIVERGENCE="${OPT_STOP_ON_KSP_DIVERGENCE:-true}"
PETSC_EXTRA_OPTIONS="${PETSC_EXTRA_OPTIONS:--ksp_gmres_restart 200 -ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_hypre_type boomeramg -pc_hypre_boomeramg_strong_threshold 0.5}"
CXXSTD="${CXXSTD:-c++11}"
JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-manual}}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PREFIX="${OUTPUT_PREFIX:-result/h8_draft_small_${RUN_NAME}_${NX}_${NY}_${NZ}_${JOB_ID}}"
TOTAL_TASKS="${SLURM_NPROCS:-${SLURM_NTASKS:-8}}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_LAUNCHER_ARGS="${MPI_LAUNCHER_ARGS:-}"
export I_MPI_HYDRA_BOOTSTRAP="${I_MPI_HYDRA_BOOTSTRAP:-slurm}"

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
echo "Array case: ${CASE_ID} ${RUN_NAME}"
echo "Draft axes: ${DRAFT_AXES}"
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
echo "PETSC_ARCH=${PETSC_ARCH}"
command -v mpicxx || true
mpicxx --version 2>&1 | head -n 3 || true
make -B all CXXSTD="${CXXSTD}" PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}"

echo "Running H8 draft small verification case"
echo "  nx=${NX} ny=${NY} nz=${NZ}"
echo "  dof=$((NX * NY * NZ * 3))"
echo "  benchmark_case=${BENCHMARK_CASE}"
echo "  opt_max_iter=${OPT_MAX_ITER}"
echo "  volfrac=${VOLFRAC}"
echo "  filter_radius=${OPT_FILTER_RADIUS}"
echo "  draft_axes=${DRAFT_AXES}"
echo "  opt_load=${OPT_LOAD}"
echo "  heaviside=${OPT_HEAVISIDE} beta0=${OPT_HEAVISIDE_BETA_INITIAL} beta_max=${OPT_HEAVISIDE_BETA_MAX} beta_interval=${OPT_HEAVISIDE_BETA_INTERVAL}"
echo "  h8_pc_type=${H8_PC_TYPE}"
echo "  output_prefix=${PREFIX}"

if [ "${MPI_LAUNCHER}" = "srun" ]; then
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -n ${TOTAL_TASKS}"
else
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -np ${TOTAL_TASKS}"
fi
echo "  launch_command=${LAUNCH_CMD} ./bin/control_arm_cpp ..."

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
  -opt_move_min "${OPT_MOVE_MIN}" \
  -opt_move_shrink "${OPT_MOVE_SHRINK}" \
  -opt_move_growth "${OPT_MOVE_GROWTH}" \
  -opt_max_compliance_increase "${OPT_MAX_COMPLIANCE_INCREASE}" \
  -opt_stability_guard true \
  -opt_projected_volume_correction "${OPT_PROJECTED_VOLUME_CORRECTION}" \
  -opt_filter_radius "${OPT_FILTER_RADIUS}" \
  -opt_draft_closure true \
  -opt_draft_axes "${DRAFT_AXES}" \
  -opt_draft_eta "${OPT_DRAFT_ETA}" \
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
  -opt_write_checkpoint true \
  -opt_checkpoint_interval "${OPT_CHECKPOINT_INTERVAL}" \
  -opt_checkpoint_prefix "${PREFIX}_checkpoint" \
  -output_prefix "${PREFIX}" \
  -ksp_type "${KSP_TYPE}" \
  -h8_pc_type "${H8_PC_TYPE}" \
  -h8_aux_rebuild_interval "${H8_AUX_REBUILD_INTERVAL}" \
  -h8_reuse_initial_guess "${H8_REUSE_INITIAL_GUESS}" \
  -opt_stop_on_ksp_divergence "${OPT_STOP_ON_KSP_DIVERGENCE}" \
  -ksp_converged_reason \
  ${H8_DM_ARGS} \
  ${PETSC_EXTRA_OPTIONS} \
  -log_view ":${PREFIX}_petsc.log"

echo "Finished at: $(date)"
