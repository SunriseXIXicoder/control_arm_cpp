#!/bin/bash
# 40x20x20 H8 悬臂梁六个单向拔模算例：
#   array 0..5 分别对应 +x, -x, +y, -y, +z, -z。
# 默认每个方向独占 1 个节点、28 个 MPI rank；6 个方向可并发提交。
#
# 提交方式：
#   sbatch submit_h8_cantilever_6axis_40_20_20.sbatch
#
# 常用覆盖：
#   sbatch --export=ALL,OPT_MAX_ITER=250,VOLFRAC=0.25 submit_h8_cantilever_6axis_40_20_20.sbatch
#   sbatch --array=4 --export=ALL,H8_PC_TYPE=aux_elastic_lu submit_h8_cantilever_6axis_40_20_20.sbatch

#SBATCH -J h8_cant6
#SBATCH -N 1
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --exclusive
#SBATCH --array=0-5%6
#SBATCH --output=result/%A_%a.%x.out
#SBATCH --error=result/%A_%a.%x.err

set -euo pipefail
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "任务在第 ${LINENO:-unknown} 行失败，退出码 ${code}" >&2; fi' EXIT

echo "进入批处理脚本: $(date)"
echo "脚本路径: $0"
echo "Bash: ${BASH_VERSION:-unknown}"

if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
  [ -f /usr/share/Modules/init/bash ] && . /usr/share/Modules/init/bash || true
fi
if command -v module >/dev/null 2>&1; then
  module load gcc/11.3.0 2>/dev/null || true
  module load intel/2020u4 2>/dev/null || true
else
  echo "Warning: module 命令不可用，沿用当前 PATH/LD_LIBRARY_PATH" >&2
fi

PETSC_DIR="${PETSC_DIR:-/data/home/dlut_ycx/petsc/petsc-v3.19.3}"
PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-opt-hypre}"
export PETSC_DIR PETSC_ARCH
if [ ! -f "${PETSC_DIR}/lib/petsc/conf/variables" ]; then
  echo "找不到 PETSc 配置文件:" >&2
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

CASE_ID="${CASE_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
DRAFT_AXES_LIST=("px" "mx" "py" "my" "pz" "mz")
DRAFT_LABELS=("+x" "-x" "+y" "-y" "+z" "-z")
RUN_NAMES=("plus_x" "minus_x" "plus_y" "minus_y" "plus_z" "minus_z")
if [ "${CASE_ID}" -lt 0 ] || [ "${CASE_ID}" -ge "${#DRAFT_AXES_LIST[@]}" ]; then
  echo "无效 CASE_ID=${CASE_ID}; 应为 0..5。" >&2
  exit 2
fi
DRAFT_AXES="${DRAFT_AXES_LIST[$CASE_ID]}"
DRAFT_LABEL="${DRAFT_LABELS[$CASE_ID]}"
RUN_NAME="${RUN_NAMES[$CASE_ID]}"

# 这里的 NEL* 是单元数；程序参数 NX/NY/NZ 是节点数。
NELX="${NELX:-40}"
NELY="${NELY:-20}"
NELZ="${NELZ:-20}"
NX="${NX:-$((NELX + 1))}"
NY="${NY:-$((NELY + 1))}"
NZ="${NZ:-$((NELZ + 1))}"

BENCHMARK_CASE="${BENCHMARK_CASE:-tip_center}"
DOMAIN_HEIGHT="${DOMAIN_HEIGHT:-0.08}"
VOLFRAC="${VOLFRAC:-0.20}"
OPT_MAX_ITER="${OPT_MAX_ITER:-500}"
OPT_FILTER_RADIUS="${OPT_FILTER_RADIUS:-5}"
OPT_MOVE="${OPT_MOVE:-0.005}"
OPT_PROJECTED_VOLUME_CORRECTION="${OPT_PROJECTED_VOLUME_CORRECTION:-true}"
OPT_DRAFT_ETA="${OPT_DRAFT_ETA:-0.5}"
OPT_MATLAB_Z_PROJECTION="${OPT_MATLAB_Z_PROJECTION:-false}"
OPT_USE_MMA="${OPT_USE_MMA:-false}"
OPT_HEAVISIDE="${OPT_HEAVISIDE:-true}"
OPT_HEAVISIDE_ETA="${OPT_HEAVISIDE_ETA:-0.50}"
OPT_HEAVISIDE_BETA_INITIAL="${OPT_HEAVISIDE_BETA_INITIAL:-1.0}"
OPT_HEAVISIDE_BETA_MAX="${OPT_HEAVISIDE_BETA_MAX:-16.0}"
OPT_HEAVISIDE_BETA_INTERVAL="${OPT_HEAVISIDE_BETA_INTERVAL:-50}"
OPT_RHO_MIN="${OPT_RHO_MIN:-1e-3}"
PENAL="${PENAL:-1}"
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
OPT_CHECKPOINT_INTERVAL="${OPT_CHECKPOINT_INTERVAL:-0}"
PETSC_EXTRA_OPTIONS="${PETSC_EXTRA_OPTIONS:--ksp_gmres_restart 200 -ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_hypre_type boomeramg -pc_hypre_boomeramg_strong_threshold 0.5}"
CXXSTD="${CXXSTD:-c++11}"

JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-manual}}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
NODES_ALLOC="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
TASKS_PER_NODE_ALLOC="${SLURM_NTASKS_PER_NODE:-${SLURM_CPUS_ON_NODE:-28}}"
TOTAL_TASKS="${SLURM_NPROCS:-${SLURM_NTASKS:-$((NODES_ALLOC * TASKS_PER_NODE_ALLOC))}}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_LAUNCHER_ARGS="${MPI_LAUNCHER_ARGS:-}"
export I_MPI_HYDRA_BOOTSTRAP="${I_MPI_HYDRA_BOOTSTRAP:-slurm}"

# 40x20x20 单元、28 rank 时用 7x2x2，局部块约为 6x10x10 单元；
# 比自动把短边切太碎更适合半径 5 的滤波/闭合通信。
H8_DM_PX="${H8_DM_PX:-7}"
H8_DM_PY="${H8_DM_PY:-2}"
H8_DM_PZ="${H8_DM_PZ:-2}"
if [ $((H8_DM_PX * H8_DM_PY * H8_DM_PZ)) -ne "${TOTAL_TASKS}" ]; then
  echo "H8_DM_PX*H8_DM_PY*H8_DM_PZ 必须等于 TOTAL_TASKS。" >&2
  echo "  H8_DM=${H8_DM_PX}x${H8_DM_PY}x${H8_DM_PZ}, TOTAL_TASKS=${TOTAL_TASKS}" >&2
  echo "  若覆盖 --ntasks-per-node，请同步覆盖 H8_DM_PX/H8_DM_PY/H8_DM_PZ。" >&2
  exit 2
fi
H8_DM_ARGS="-h8_dm_px ${H8_DM_PX} -h8_dm_py ${H8_DM_PY} -h8_dm_pz ${H8_DM_PZ}"

cd "${SUBMIT_DIR}"
mkdir -p bin result logs
CASE_DIR="${OUTPUT_DIR:-result/h8_${BENCHMARK_CASE}_6axis_${NELX}_${NELY}_${NELZ}_${RUN_NAME}_${JOB_ID}}"
PREFIX="${OUTPUT_PREFIX:-${CASE_DIR}/${RUN_NAME}}"
mkdir -p "${CASE_DIR}"

if [ "${H8_TEE_CASE_LOG:-true}" = "true" ]; then
  exec > >(tee "${CASE_DIR}/slurm.out") 2> >(tee "${CASE_DIR}/slurm.err" >&2)
fi

echo "开始时间: $(date)"
echo "提交目录: ${SUBMIT_DIR}"
echo "运行节点: $(hostname)"
echo "Job id: ${JOB_ID}"
echo "Array case: ${CASE_ID} ${RUN_NAME}"
echo "拔模方向: ${DRAFT_LABEL} (${DRAFT_AXES})"
echo "输出目录: ${CASE_DIR}"
echo "Slurm nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"

if [ -f scripts/h8_slurm_guard.sh ]; then
  . scripts/h8_slurm_guard.sh
  h8_print_slurm_diagnostics
  h8_guard_slurm_memory_accounting
  h8_guard_required_hypre_pc
  h8_guard_petsc_hypre
else
  echo "Warning: scripts/h8_slurm_guard.sh 不存在，跳过 Slurm/PETSc guard" >&2
fi

echo "PETSC_DIR=${PETSC_DIR}"
echo "PETSC_ARCH=${PETSC_ARCH}"
command -v mpicxx || true
mpicxx --version 2>&1 | head -n 3 || true
make -B all CXXSTD="${CXXSTD}" PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}"

echo "运行 40x20x20 H8 悬臂梁单向拔模算例"
echo "  elements=${NELX}x${NELY}x${NELZ}"
echo "  nodes=${NX}x${NY}x${NZ}"
echo "  dof=$((NX * NY * NZ * 3))"
echo "  ranks=${TOTAL_TASKS}, dm=${H8_DM_PX}x${H8_DM_PY}x${H8_DM_PZ}"
echo "  benchmark_case=${BENCHMARK_CASE}"
echo "  opt_max_iter=${OPT_MAX_ITER}"
echo "  volfrac=${VOLFRAC}"
echo "  move=${OPT_MOVE}"
echo "  filter_radius=${OPT_FILTER_RADIUS}"
echo "  draft_axes=${DRAFT_LABEL} (${DRAFT_AXES})"
echo "  matlab_z_projection=${OPT_MATLAB_Z_PROJECTION}"
echo "  h8_pc_type=${H8_PC_TYPE}"
echo "  output_prefix=${PREFIX}"

if [ "${MPI_LAUNCHER}" = "srun" ]; then
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -n ${TOTAL_TASKS}"
else
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -np ${TOTAL_TASKS}"
fi
echo "启动命令: ${LAUNCH_CMD} ./bin/control_arm_cpp ..."

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
  -opt_draft_closure true \
  -opt_draft_axes "${DRAFT_AXES}" \
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

echo "结束时间: $(date)"
