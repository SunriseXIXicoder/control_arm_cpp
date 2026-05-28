#!/bin/sh
set -eu

# MATLAB-style coarse element counts. PETSc receives nodal counts.
NP="${NP:-2}"
NELX="${NELX:-40}"
NELY="${NELY:-20}"
NELZ="${NELZ:-8}"
NX=$((NELX + 1))
NY=$((NELY + 1))
NZ=$((NELZ + 1))

ITERS="${ITERS:-50}"
VOLFRAC="${VOLFRAC:-0.30}"
PREFIX="${PREFIX:-result/ems_ann_opt_${NELX}_${NELY}_${NELZ}_np${NP}}"
ANN_DIR="${ANN_DIR:-../input_5}"
# 0 means the previous three control-arm load cases with AHP weights.
LOAD_CASE="${LOAD_CASE:-0}"
FILTER_RADIUS="${FILTER_RADIUS:-1.5}"
AB_TRIANGLE_RETRACT="${AB_TRIANGLE_RETRACT:-0.22}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
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
KSP_RTOL="${KSP_RTOL:-1e-4}"
KSP_ATOL="${KSP_ATOL:-1e-12}"
KSP_MAX_IT="${KSP_MAX_IT:-5000}"
KSP_TYPE="${KSP_TYPE:-cg}"
PC_TYPE="${PC_TYPE:-shell}"
WRITE_VTK="${WRITE_VTK:-true}"
KSP_EXTRA="${KSP_EXTRA:-}"
EMS_DM_ARGS=""
if [ -n "${EMS_DM_PX:-}" ] || [ -n "${EMS_DM_PY:-}" ] || [ -n "${EMS_DM_PZ:-}" ]; then
  if [ -z "${EMS_DM_PX:-}" ] || [ -z "${EMS_DM_PY:-}" ] || [ -z "${EMS_DM_PZ:-}" ]; then
    echo "Set EMS_DM_PX, EMS_DM_PY, and EMS_DM_PZ together." >&2
    exit 2
  fi
  EMS_DM_ARGS="-ems_dm_px ${EMS_DM_PX} -ems_dm_py ${EMS_DM_PY} -ems_dm_pz ${EMS_DM_PZ}"
fi
if [ "${KSP_TYPE}" = "gmres" ]; then
  KSP_EXTRA="${KSP_EXTRA} -ksp_gmres_restart ${KSP_GMRES_RESTART:-200}"
fi

mkdir -p result
make wsl

mpirun -np "${NP}" ./bin/control_arm_cpp \
  -mode optimize \
  -operator emsfem_ann \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density 0.02 \
  -young_modulus "${YOUNG_MODULUS}" \
  -mask_threshold 0.5 \
  -ab_triangle_retract "${AB_TRIANGLE_RETRACT}" \
  -volfrac "${VOLFRAC}" \
  -opt_max_iter "${ITERS}" \
  -opt_move "${OPT_MOVE}" \
  -opt_move_min "${OPT_MOVE_MIN}" \
  -opt_move_shrink "${OPT_MOVE_SHRINK}" \
  -opt_move_growth "${OPT_MOVE_GROWTH}" \
  -opt_max_compliance_increase "${OPT_MAX_COMPLIANCE_INCREASE}" \
  -opt_stability_guard true \
  -opt_filter_radius "${FILTER_RADIUS}" \
  -opt_rho_min 1e-3 \
  -opt_heaviside_projection "${OPT_HEAVISIDE}" \
  -opt_heaviside_eta "${OPT_HEAVISIDE_ETA}" \
  -opt_heaviside_beta_initial "${OPT_HEAVISIDE_BETA_INITIAL}" \
  -opt_heaviside_beta_max "${OPT_HEAVISIDE_BETA_MAX}" \
  -opt_heaviside_beta_interval "${OPT_HEAVISIDE_BETA_INTERVAL}" \
  -opt_load 1.0 \
  -opt_ksp_rtol "${KSP_RTOL}" \
  -opt_ksp_max_it "${KSP_MAX_IT}" \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval 10 \
  -opt_checkpoint_prefix "${PREFIX}_checkpoint" \
  -opt_write_final_vtk "${WRITE_VTK}" \
  -opt_vtk_file "${PREFIX}_final.vtk" \
  -output_prefix "${PREFIX}" \
  -ems_ann_dir "${ANN_DIR}" \
  -ems_cache_element_matrices true \
  -ems_cache_gib_limit 2 \
  -control_arm_bc true \
  -load_case "${LOAD_CASE}" \
  -include_spring_load true \
  -ksp_type "${KSP_TYPE}" \
  -pc_type "${PC_TYPE}" \
  -ksp_rtol "${KSP_RTOL}" \
  -ksp_atol "${KSP_ATOL}" \
  -ksp_max_it "${KSP_MAX_IT}" \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log" \
  ${EMS_DM_ARGS} \
  ${KSP_EXTRA} < /dev/null
