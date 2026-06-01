#!/bin/sh
#SBATCH -J ca_cpp_scale
#SBATCH -N 4
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=result/%j.cpp.out
#SBATCH --error=result/%j.cpp.err

set -eu

# Examples:
#   sbatch --export=ALL,MODE=geometry,NX=160,NY=80,NZ=48 submit_control_arm_cpp_scale.sbatch
#   sbatch --nodes=4 --ntasks-per-node=28 --export=ALL,MODE=solve,NX=700,NY=400,NZ=120 submit_control_arm_cpp_scale.sbatch

source /data/app/intel2022.3.1/setvars.sh > /dev/null 2>&1 || true

export PETSC_DIR="${PETSC_DIR:-/data/app/petsc}"
export PETSC_ARCH="${PETSC_ARCH:-}"
export PATH="${PETSC_DIR}/bin:${PATH}"
if [ -n "${PETSC_ARCH}" ]; then
  export LD_LIBRARY_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -s unlimited
ulimit -l unlimited || true

MODE="${MODE:-solve}"
NX="${NX:-700}"
NY="${NY:-400}"
NZ="${NZ:-120}"
OPERATOR="${OPERATOR:-low_order}"
PREFIX="${OUTPUT_PREFIX:-result/cpp_${MODE}_${NX}_${NY}_${NZ}_${SLURM_JOB_ID}}"
WRITE_STRUCTURED_VTK="${WRITE_STRUCTURED_VTK:-false}"
WRITE_SOLID_VTK="${WRITE_SOLID_VTK:-false}"
MAX_VTK_CELLS="${MAX_VTK_CELLS:-300000}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
USER_PC_TYPE="${PC_TYPE:-}"
USER_KSP_MAX_IT="${KSP_MAX_IT:-}"
MG_ARGS="-pc_gamg_type agg -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi"
if [ "${OPERATOR}" = "h8_matrix_free" ]; then
  PC_TYPE="${USER_PC_TYPE:-jacobi}"
  MG_ARGS=""
else
  PC_TYPE="${USER_PC_TYPE:-gamg}"
fi
OPT_MAX_ITER="${OPT_MAX_ITER:-20}"
VOLFRAC="${VOLFRAC:-0.30}"
AB_TRIANGLE_RETRACT="${AB_TRIANGLE_RETRACT:-0.22}"
OPT_FILTER_RADIUS="${OPT_FILTER_RADIUS:-1.5}"
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
OPT_Z_DRAFT_CLOSURE="${OPT_Z_DRAFT_CLOSURE:-true}"
OPT_Z_DRAFT_ETA="${OPT_Z_DRAFT_ETA:-0.5}"
PENAL="${PENAL:-3.0}"
EMIN="${EMIN:-1e-6}"
VOID_DENSITY="${VOID_DENSITY:-0.02}"
OPT_WRITE_CHECKPOINT="${OPT_WRITE_CHECKPOINT:-true}"
OPT_CHECKPOINT_INTERVAL="${OPT_CHECKPOINT_INTERVAL:-0}"
OPT_CHECKPOINT_PREFIX="${OPT_CHECKPOINT_PREFIX:-${PREFIX}_checkpoint}"
OPT_ARGS=""
KSP_MAX_IT="${USER_KSP_MAX_IT:-500}"
if [ "${MODE}" = "optimize" ]; then
  PC_TYPE="${USER_PC_TYPE:-jacobi}"
  MG_ARGS=""
  KSP_MAX_IT="${USER_KSP_MAX_IT:-3000}"
  OPT_ARGS="-volfrac ${VOLFRAC} -opt_max_iter ${OPT_MAX_ITER} -opt_move ${OPT_MOVE} -opt_move_min ${OPT_MOVE_MIN} -opt_move_shrink ${OPT_MOVE_SHRINK} -opt_move_growth ${OPT_MOVE_GROWTH} -opt_max_compliance_increase ${OPT_MAX_COMPLIANCE_INCREASE} -opt_stability_guard true -opt_filter_radius ${OPT_FILTER_RADIUS} -opt_z_draft_closure ${OPT_Z_DRAFT_CLOSURE} -opt_z_draft_eta ${OPT_Z_DRAFT_ETA} -opt_rho_min 1e-3 -opt_heaviside_projection ${OPT_HEAVISIDE} -opt_heaviside_eta ${OPT_HEAVISIDE_ETA} -opt_heaviside_beta_initial ${OPT_HEAVISIDE_BETA_INITIAL} -opt_heaviside_beta_max ${OPT_HEAVISIDE_BETA_MAX} -opt_heaviside_beta_interval ${OPT_HEAVISIDE_BETA_INTERVAL} -opt_ksp_max_it ${KSP_MAX_IT} -opt_write_final_vtk false -opt_write_checkpoint ${OPT_WRITE_CHECKPOINT} -opt_checkpoint_interval ${OPT_CHECKPOINT_INTERVAL} -opt_checkpoint_prefix ${OPT_CHECKPOINT_PREFIX}"
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p bin result
make all

VTK_ARGS=""
if [ "${WRITE_STRUCTURED_VTK}" = "true" ] || [ "${WRITE_STRUCTURED_VTK}" = "1" ]; then
  VTK_ARGS="${VTK_ARGS} -write_structured_vtk true -structured_vtk_file ${PREFIX}.vtk"
fi
if [ "${WRITE_SOLID_VTK}" = "true" ] || [ "${WRITE_SOLID_VTK}" = "1" ]; then
  VTK_ARGS="${VTK_ARGS} -write_solid_vtk true -solid_vtk_file ${PREFIX}_solid.vtk -max_vtk_cells ${MAX_VTK_CELLS}"
fi

echo "Running C++/PETSc control-arm case"
echo "  mode=${MODE}"
echo "  nodes=${SLURM_JOB_NUM_NODES:-unknown}"
echo "  tasks=${SLURM_NTASKS:-unknown}"
echo "  nx=${NX} ny=${NY} nz=${NZ}"
echo "  operator=${OPERATOR}"
echo "  pc_type=${PC_TYPE}"
echo "  opt_max_iter=${OPT_MAX_ITER}"
echo "  opt_filter_radius=${OPT_FILTER_RADIUS}"
echo "  ab_triangle_retract=${AB_TRIANGLE_RETRACT}"
echo "  heaviside=${OPT_HEAVISIDE} beta0=${OPT_HEAVISIDE_BETA_INITIAL} beta_max=${OPT_HEAVISIDE_BETA_MAX} beta_interval=${OPT_HEAVISIDE_BETA_INTERVAL}"
echo "  opt_write_checkpoint=${OPT_WRITE_CHECKPOINT}"
echo "  output_prefix=${PREFIX}"

# shellcheck disable=SC2086
srun -n "$SLURM_NTASKS" ./bin/control_arm_cpp \
  -mode "${MODE}" \
  -operator "${OPERATOR}" \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density "${VOID_DENSITY}" \
  -young_modulus "${YOUNG_MODULUS}" \
  -penal "${PENAL}" \
  -emin "${EMIN}" \
  -mask_threshold 0.5 \
  -ab_triangle_retract "${AB_TRIANGLE_RETRACT}" \
  -load 1.0 \
  -output_prefix "${PREFIX}" \
  ${OPT_ARGS} \
  ${VTK_ARGS} \
  -ksp_type cg \
  -pc_type "${PC_TYPE}" \
  ${MG_ARGS} \
  -ksp_rtol 1e-6 \
  -ksp_max_it "${KSP_MAX_IT}" \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log"
