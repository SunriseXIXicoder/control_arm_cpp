#!/bin/bash
#SBATCH -J ca_ems_ann
# For the 10M-DOF, sub_n=5 ANN optimizer, 48 nodes is the recommended
# strong-preconditioner default.  It gives a balanced 24x14x4-style process
# grid on 28-core nodes while leaving enough local DOFs for AMG coarsening.
# Override at submission time with: sbatch -N <nodes> ...
#SBATCH -N 48
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=result/%j.%x.out
#SBATCH --error=result/%j.%x.err

set -euo pipefail
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "Job failed at line ${LINENO:-unknown} with exit code ${code}" >&2; fi' EXIT
echo "Batch script entered shell: $(date)"
echo "Script path: $0"

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

PETSC_DIR="${PETSC_DIR:-/data/home/dlut_ycx/petsc/petsc-v3.19.3}"
PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-opt-hypre}"
export PETSC_DIR PETSC_ARCH
if [ ! -f "${PETSC_DIR}/lib/petsc/conf/variables" ]; then
  echo "PETSc configuration not found:" >&2
  echo "  PETSC_DIR=${PETSC_DIR}" >&2
  echo "  PETSC_ARCH=${PETSC_ARCH}" >&2
  echo "Expected file: ${PETSC_DIR}/lib/petsc/conf/variables" >&2
  exit 2
fi
if [ ! -d "${PETSC_DIR}/${PETSC_ARCH}" ]; then
  echo "PETSc arch directory not found:" >&2
  echo "  PETSC_DIR=${PETSC_DIR}" >&2
  echo "  PETSC_ARCH=${PETSC_ARCH}" >&2
  echo "Expected directory: ${PETSC_DIR}/${PETSC_ARCH}" >&2
  exit 2
fi
export PATH="${PETSC_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export I_MPI_HYDRA_BOOTSTRAP="${I_MPI_HYDRA_BOOTSTRAP:-slurm}"
ulimit -s unlimited || true
ulimit -l unlimited || true

# If NELX/NELY/NELZ are provided, they are MATLAB-style coarse element counts.
# Otherwise NX/NY/NZ are PETSc nodal counts.  DOF = 3*NX*NY*NZ.
if [ -n "${NELX:-}" ] || [ -n "${NELY:-}" ] || [ -n "${NELZ:-}" ]; then
  if [ -z "${NELX:-}" ] || [ -z "${NELY:-}" ] || [ -z "${NELZ:-}" ]; then
    echo "Set NELX, NELY, and NELZ together." >&2
    exit 2
  fi
  NX=$((NELX + 1))
  NY=$((NELY + 1))
  NZ=$((NELZ + 1))
else
  NX="${NX:-325}"
  NY="${NY:-186}"
  NZ="${NZ:-56}"
fi

MODE="${MODE:-optimize}"
DOMAIN_HEIGHT="${DOMAIN_HEIGHT:-0.08}"
EMS_SUB_N="${EMS_SUB_N:-5}"
ANN_DIR="${ANN_DIR:-../input_${EMS_SUB_N}}"
LOAD_CASE="${LOAD_CASE:-0}"
CONTROL_ARM_BC="${CONTROL_ARM_BC:-true}"
INCLUDE_SPRING_LOAD="${INCLUDE_SPRING_LOAD:-true}"
VOLFRAC="${VOLFRAC:-0.30}"
FILTER_RADIUS="${FILTER_RADIUS:-1.5}"
EMS_FILTER_MODE="${EMS_FILTER_MODE:-fine}"
AB_TRIANGLE_RETRACT="${AB_TRIANGLE_RETRACT:-0.22}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
PENAL="${PENAL:-3.0}"
EMIN="${EMIN:-1e-6}"
VOID_DENSITY="${VOID_DENSITY:-0.02}"
OPT_RHO_MIN="${OPT_RHO_MIN:-1e-3}"
OPT_MAX_ITER="${OPT_MAX_ITER:-250}"
OPT_MOVE="${OPT_MOVE:-0.025}"
OPT_MOVE_MIN="${OPT_MOVE_MIN:-0.004}"
OPT_MOVE_SHRINK="${OPT_MOVE_SHRINK:-0.5}"
OPT_MOVE_GROWTH="${OPT_MOVE_GROWTH:-1.05}"
OPT_MAX_COMPLIANCE_INCREASE="${OPT_MAX_COMPLIANCE_INCREASE:-0.08}"
OPT_DRAFT_CLOSURE="${OPT_DRAFT_CLOSURE:-true}"
OPT_DRAFT_AXES="${OPT_DRAFT_AXES:-+z}"
OPT_DRAFT_ETA="${OPT_DRAFT_ETA:-0.50}"
EMS_DRAFT_CLOSURE_MODE="${EMS_DRAFT_CLOSURE_MODE:-legacy_z}"
OPT_HEAVISIDE="${OPT_HEAVISIDE:-true}"
OPT_HEAVISIDE_ETA="${OPT_HEAVISIDE_ETA:-0.50}"
OPT_HEAVISIDE_BETA_INITIAL="${OPT_HEAVISIDE_BETA_INITIAL:-1.0}"
# Keep the black-white continuation explicit but gentle.  The 8525166
# production run was stable through beta=4 and became fragile when beta jumped
# to 8 at step 301, so do not advance beta faster than 100-step stages unless
# the preconditioner has been revalidated.
OPT_HEAVISIDE_BETA_MAX="${OPT_HEAVISIDE_BETA_MAX:-16.0}"
OPT_HEAVISIDE_BETA_INTERVAL="${OPT_HEAVISIDE_BETA_INTERVAL:-100}"
CACHE_K="${CACHE_K:-true}"
CACHE_GIB_LIMIT="${CACHE_GIB_LIMIT:-8}"
EMS_PC_TYPE="${EMS_PC_TYPE:-aux_ann_gamg}"
GAMG_PROFILE="${GAMG_PROFILE:-strong}"
HYPRE_PROFILE="${HYPRE_PROFILE:-strong}"
KSP_GMRES_RESTART="${KSP_GMRES_RESTART:-200}"
case "${EMS_PC_TYPE}" in
  block_jacobi)
    KSP_TYPE="${KSP_TYPE:-cg}"
    PC_TYPE="${PC_TYPE:-shell}"
    KSP_EXTRA="${KSP_EXTRA:-}"
    ;;
  aux_ann_gamg|aux_elastic_gamg|aux_gamg)
    KSP_TYPE="${KSP_TYPE:-fgmres}"
    PC_TYPE="${PC_TYPE:-gamg}"
    if [ -z "${KSP_EXTRA+x}" ]; then
      case "${GAMG_PROFILE}" in
        lowmem)
          KSP_EXTRA="-ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_gamg_type agg"
          ;;
        strong)
          KSP_EXTRA="-ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_gamg_type agg -pc_gamg_threshold 0.02 -pc_gamg_square_graph 1 -pc_gamg_agg_nsmooths 2 -mg_levels_ksp_type chebyshev -mg_levels_ksp_max_it 2 -mg_levels_pc_type jacobi"
          ;;
        xstrong)
          KSP_EXTRA="-ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_gamg_type agg -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale 0.0 -pc_gamg_square_graph 1 -pc_gamg_aggressive_coarsening 1 -pc_gamg_agg_nsmooths 3 -pc_gamg_esteig_ksp_max_it 10 -mg_levels_ksp_type chebyshev -mg_levels_ksp_max_it 4 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.1 -mg_levels_pc_type jacobi"
          ;;
        *)
          echo "Unknown GAMG_PROFILE=${GAMG_PROFILE}; use lowmem, strong, or xstrong." >&2
          exit 2
          ;;
      esac
    fi
    ;;
  *)
    KSP_TYPE="${KSP_TYPE:-fgmres}"
    PC_TYPE="${PC_TYPE:-hypre}"
    if [ -z "${KSP_EXTRA+x}" ]; then
      if [ "${EMS_PC_TYPE}" = "aux_ann_hypre" ] || [ "${EMS_PC_TYPE}" = "aux_elastic_hypre" ]; then
        case "${HYPRE_PROFILE}" in
          lowmem)
            KSP_EXTRA="-ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_hypre_type boomeramg -pc_hypre_boomeramg_strong_threshold 0.5 -pc_hypre_boomeramg_nodal_coarsen 1 -pc_hypre_boomeramg_vec_interp_variant 3"
            ;;
          strong)
            KSP_EXTRA="-ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_hypre_type boomeramg -pc_hypre_boomeramg_strong_threshold 0.6 -pc_hypre_boomeramg_nodal_coarsen 1 -pc_hypre_boomeramg_vec_interp_variant 3 -pc_hypre_boomeramg_interp_type ext+i -pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi"
            ;;
          xstrong)
            KSP_EXTRA="-ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_hypre_type boomeramg -pc_hypre_boomeramg_strong_threshold 0.7 -pc_hypre_boomeramg_nodal_coarsen 1 -pc_hypre_boomeramg_vec_interp_variant 3 -pc_hypre_boomeramg_interp_type ext+i -pc_hypre_boomeramg_coarsen_type PMIS -pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi -pc_hypre_boomeramg_max_iter 2"
            ;;
          *)
            echo "Unknown HYPRE_PROFILE=${HYPRE_PROFILE}; use lowmem, strong, or xstrong." >&2
            exit 2
            ;;
        esac
      else
        KSP_EXTRA="-ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_hypre_type boomeramg -pc_hypre_boomeramg_strong_threshold 0.6"
      fi
    fi
    ;;
esac
case "${EMS_PC_TYPE}" in
  *hypre*)
    if ! grep -q 'PETSC_HAVE_HYPRE' "${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h" 2>/dev/null; then
      echo "EMS_PC_TYPE=${EMS_PC_TYPE} requires a PETSc arch built with HYPRE." >&2
      echo "Current PETSC_DIR=${PETSC_DIR}" >&2
      echo "Current PETSC_ARCH=${PETSC_ARCH}" >&2
      echo "Use PETSC_ARCH=arch-linux-c-opt-hypre or rebuild PETSc with --download-hypre." >&2
      exit 2
    fi
    ;;
esac
KSP_MAX_IT="${KSP_MAX_IT:-1200}"
KSP_RTOL="${KSP_RTOL:-1e-3}"
KSP_ATOL="${KSP_ATOL:-1e-12}"
ALLOW_HUGE_EMS="${ALLOW_HUGE_EMS:-false}"
MAX_DEFAULT_EMS_DOF="${MAX_DEFAULT_EMS_DOF:-20000000}"
WRITE_FINAL_VTK="${WRITE_FINAL_VTK:-false}"
OPT_VTK_MAX_POINTS="${OPT_VTK_MAX_POINTS:-5000000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
CXXSTD="${CXXSTD:-c++11}"
JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-manual}}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_LABEL="${RUN_LABEL:-ems_ann_10m_sub5_auxanngamg}"
PREFIX="${OUTPUT_PREFIX:-result/${RUN_LABEL}_${NX}_${NY}_${NZ}_${JOB_ID}}"
NODES_ALLOC="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
TASKS_PER_NODE_ALLOC="${SLURM_NTASKS_PER_NODE:-${SLURM_CPUS_ON_NODE:-28}}"
TOTAL_TASKS="${SLURM_NPROCS:-${SLURM_NTASKS:-$((NODES_ALLOC * TASKS_PER_NODE_ALLOC))}}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_LAUNCHER_ARGS="${MPI_LAUNCHER_ARGS:-}"
DENSITY_FILE="${DENSITY_FILE:-}"
MASK_FILE="${MASK_FILE:-}"
POSTSOLVE_CHECKPOINT="${POSTSOLVE_CHECKPOINT:-${CHECKPOINT_FILE:-}}"
POSTSOLVE_PREFIX="${POSTSOLVE_PREFIX:-}"

DOF=$((NX * NY * NZ * 3))
COARSE_ELEMS=$(((NX - 1) * (NY - 1) * (NZ - 1)))
FINE_DENSITY_CELLS=$((COARSE_ELEMS * EMS_SUB_N * EMS_SUB_N * EMS_SUB_N))
if [ "${DOF}" -gt "${MAX_DEFAULT_EMS_DOF}" ] && [ "${ALLOW_HUGE_EMS}" != "true" ]; then
  echo "Refusing huge EMsFEM ANN run without ALLOW_HUGE_EMS=true." >&2
  echo "  nx=${NX} ny=${NY} nz=${NZ} dof=${DOF}" >&2
  echo "  ems_sub_n=${EMS_SUB_N} fine_density_cells=${FINE_DENSITY_CELLS}" >&2
  echo "This script defaults to the 10M-DOF case. Set NX/NY/NZ intentionally and ALLOW_HUGE_EMS=true for larger scaling runs." >&2
  exit 2
fi

EMS_DM_ARGS=""
if [ -n "${EMS_DM_PX:-}" ] || [ -n "${EMS_DM_PY:-}" ] || [ -n "${EMS_DM_PZ:-}" ]; then
  if [ -z "${EMS_DM_PX:-}" ] || [ -z "${EMS_DM_PY:-}" ] || [ -z "${EMS_DM_PZ:-}" ]; then
    echo "Set EMS_DM_PX, EMS_DM_PY, and EMS_DM_PZ together." >&2
    exit 2
  fi
  EMS_DM_ARGS="-ems_dm_px ${EMS_DM_PX} -ems_dm_py ${EMS_DM_PY} -ems_dm_pz ${EMS_DM_PZ}"
fi
if [ "${KSP_TYPE}" = "gmres" ] || [ "${KSP_TYPE}" = "fgmres" ]; then
  KSP_EXTRA="${KSP_EXTRA} -ksp_gmres_restart ${KSP_GMRES_RESTART}"
fi

strip_petsc_info_suffix() {
  case "$1" in
    *.petscbin.info) printf '%s\n' "${1%.info}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

checkpoint_prefix_from_file() {
  local path
  path="$(strip_petsc_info_suffix "$1")"
  case "${path}" in
    *_final_rho_phys.petscbin) printf '%s\n' "${path%_final_rho_phys.petscbin}" ;;
    *_final_rho_design.petscbin) printf '%s\n' "${path%_final_rho_design.petscbin}" ;;
    *_final_mask.petscbin) printf '%s\n' "${path%_final_mask.petscbin}" ;;
    *) printf '%s\n' "${path}" ;;
  esac
}

resolve_result_path() {
  local path="$1"
  if [ -f "${path}" ]; then
    printf '%s\n' "${path}"
  elif [[ "${path}" != */* ]] && [ -f "result/${path}" ]; then
    printf '%s\n' "result/${path}"
  else
    printf '%s\n' "${path}"
  fi
}

POSTSOLVE_ARGS=""
if [ "${MODE}" = "ems_ann_postsolve" ]; then
  if [ -n "${POSTSOLVE_CHECKPOINT}" ] && [ -z "${POSTSOLVE_PREFIX}" ]; then
    POSTSOLVE_PREFIX="$(checkpoint_prefix_from_file "${POSTSOLVE_CHECKPOINT}")"
  fi
  if [ -n "${POSTSOLVE_PREFIX}" ]; then
    DENSITY_FILE="${DENSITY_FILE:-${POSTSOLVE_PREFIX}_final_rho_phys.petscbin}"
    MASK_FILE="${MASK_FILE:-${POSTSOLVE_PREFIX}_final_mask.petscbin}"
  fi
  if [ -n "${DENSITY_FILE}" ]; then
    DENSITY_FILE="$(strip_petsc_info_suffix "${DENSITY_FILE}")"
  fi
  if [ -n "${MASK_FILE}" ]; then
    MASK_FILE="$(strip_petsc_info_suffix "${MASK_FILE}")"
  fi
  if [ -z "${DENSITY_FILE}" ]; then
    echo "MODE=ems_ann_postsolve requires DENSITY_FILE=<*_rho_phys.petscbin> or POSTSOLVE_CHECKPOINT=<final checkpoint file>." >&2
    exit 2
  fi
fi

cd "$SUBMIT_DIR"
mkdir -p bin result logs
if [ "${MODE}" = "ems_ann_postsolve" ]; then
  DENSITY_FILE="$(resolve_result_path "${DENSITY_FILE}")"
  if [ -n "${MASK_FILE}" ]; then
    MASK_FILE="$(resolve_result_path "${MASK_FILE}")"
  fi
  if [ ! -f "${DENSITY_FILE}" ]; then
    echo "EMsFEM ANN postsolve density file not found: ${DENSITY_FILE}" >&2
    echo "If you copied a .petscbin.info name, use the matching .petscbin data file; the script strips .info automatically." >&2
    exit 2
  fi
  if [ -n "${MASK_FILE}" ] && [ ! -f "${MASK_FILE}" ]; then
    echo "EMsFEM ANN postsolve mask file not found: ${MASK_FILE}" >&2
    echo "If you copied a .petscbin.info name, use the matching .petscbin data file; the script strips .info automatically." >&2
    exit 2
  fi
  POSTSOLVE_ARGS="-density_file ${DENSITY_FILE}"
  if [ -n "${MASK_FILE}" ]; then
    POSTSOLVE_ARGS="${POSTSOLVE_ARGS} -mask_file ${MASK_FILE}"
  fi
fi
echo "Start time: $(date)"
echo "Submit dir: ${SUBMIT_DIR}"
echo "Run host: $(hostname)"
echo "Job id: ${JOB_ID}"
echo "Slurm nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "PETSC_DIR=${PETSC_DIR}"
echo "PETSC_ARCH=${PETSC_ARCH}"
echo "Compiler wrapper:"
command -v mpicxx || true
mpicxx --version 2>&1 | head -n 3 || true
make -B all CXXSTD="${CXXSTD}" PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}"

echo "Running EMsFEM ANN control-arm ${MODE}"
echo "  nodes=${SLURM_JOB_NUM_NODES:-unknown}"
echo "  tasks=${TOTAL_TASKS}"
echo "  nx=${NX} ny=${NY} nz=${NZ}"
echo "  dof=${DOF}"
echo "  ems_sub_n=${EMS_SUB_N}"
echo "  fine_density_cells=${FINE_DENSITY_CELLS}"
echo "  ann_dir=${ANN_DIR}"
echo "  load_case=${LOAD_CASE} (0 means weighted 3-case load set)"
echo "  control_arm_bc=${CONTROL_ARM_BC}"
echo "  include_spring_load=${INCLUDE_SPRING_LOAD}"
echo "  volfrac=${VOLFRAC}"
echo "  ab_triangle_retract=${AB_TRIANGLE_RETRACT}"
echo "  filter_mode=${EMS_FILTER_MODE} filter_radius=${FILTER_RADIUS}"
echo "  draft_closure=${OPT_DRAFT_CLOSURE} mode=${EMS_DRAFT_CLOSURE_MODE} axes=${OPT_DRAFT_AXES} eta=${OPT_DRAFT_ETA}"
echo "  heaviside=${OPT_HEAVISIDE} beta0=${OPT_HEAVISIDE_BETA_INITIAL} beta_max=${OPT_HEAVISIDE_BETA_MAX} beta_interval=${OPT_HEAVISIDE_BETA_INTERVAL}"
echo "  ems_pc_type=${EMS_PC_TYPE}"
echo "  gamg_profile=${GAMG_PROFILE}"
echo "  hypre_profile=${HYPRE_PROFILE}"
echo "  ksp_type=${KSP_TYPE} pc_type=${PC_TYPE} ksp_rtol=${KSP_RTOL} ksp_max_it=${KSP_MAX_IT}"
echo "  output_prefix=${PREFIX}"
echo "  mpi_launcher=${MPI_LAUNCHER}"
if [ "${MODE}" = "ems_ann_postsolve" ]; then
  echo "  density_file=${DENSITY_FILE}"
  echo "  mask_file=${MASK_FILE:-none}"
fi

if [ "${MPI_LAUNCHER}" = "srun" ]; then
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -n ${TOTAL_TASKS}"
else
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -np ${TOTAL_TASKS}"
fi
echo "  launch_command=${LAUNCH_CMD} ./bin/control_arm_cpp ..."

${LAUNCH_CMD} ./bin/control_arm_cpp \
  -mode "${MODE}" \
  -operator emsfem_ann \
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
  -opt_filter_radius "${FILTER_RADIUS}" \
  -opt_draft_closure "${OPT_DRAFT_CLOSURE}" \
  -opt_draft_axes "${OPT_DRAFT_AXES}" \
  -opt_draft_eta "${OPT_DRAFT_ETA}" \
  -opt_rho_min "${OPT_RHO_MIN}" \
  -opt_heaviside_projection "${OPT_HEAVISIDE}" \
  -opt_heaviside_eta "${OPT_HEAVISIDE_ETA}" \
  -opt_heaviside_beta_initial "${OPT_HEAVISIDE_BETA_INITIAL}" \
  -opt_heaviside_beta_max "${OPT_HEAVISIDE_BETA_MAX}" \
  -opt_heaviside_beta_interval "${OPT_HEAVISIDE_BETA_INTERVAL}" \
  -opt_load 1.0 \
  -opt_ksp_rtol "${KSP_RTOL}" \
  -opt_ksp_max_it "${KSP_MAX_IT}" \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  -opt_checkpoint_prefix "${PREFIX}_checkpoint" \
  -opt_write_final_vtk "${WRITE_FINAL_VTK}" \
  -opt_vtk_max_points "${OPT_VTK_MAX_POINTS}" \
  -opt_vtk_file "${PREFIX}_final.vtk" \
  -output_prefix "${PREFIX}" \
  -ems_ann_dir "${ANN_DIR}" \
  -ems_sub_n "${EMS_SUB_N}" \
  -ems_cache_element_matrices "${CACHE_K}" \
  -ems_cache_gib_limit "${CACHE_GIB_LIMIT}" \
  -ems_pc_type "${EMS_PC_TYPE}" \
  -ems_ann_filter_mode "${EMS_FILTER_MODE}" \
  -ems_ann_draft_closure_mode "${EMS_DRAFT_CLOSURE_MODE}" \
  -control_arm_bc "${CONTROL_ARM_BC}" \
  -load_case "${LOAD_CASE}" \
  -include_spring_load "${INCLUDE_SPRING_LOAD}" \
  -write_structured_vtk false \
  -write_solution false \
  ${POSTSOLVE_ARGS} \
  -ksp_type "${KSP_TYPE}" \
  -pc_type "${PC_TYPE}" \
  -ksp_rtol "${KSP_RTOL}" \
  -ksp_atol "${KSP_ATOL}" \
  -ksp_max_it "${KSP_MAX_IT}" \
  -ksp_converged_reason \
  ${EMS_DM_ARGS} \
  ${KSP_EXTRA} \
  -log_view ":${PREFIX}_petsc.log"

echo "Finished at: $(date)"
