#!/bin/sh
set -eu

NP="${NP:-4}"
NX="${NX:-14}"
NY="${NY:-8}"
NZ="${NZ:-12}"
ITERS="${ITERS:-4}"
VOLFRAC="${VOLFRAC:-0.35}"
H8_PC_TYPE="${H8_PC_TYPE:-aux_elastic_hypre}"
H8_AUX_REBUILD_INTERVAL="${H8_AUX_REBUILD_INTERVAL:-1}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
PENAL="${PENAL:-3.0}"
EMIN="${EMIN:-1e-6}"
VOID_DENSITY="${VOID_DENSITY:-0.02}"
KSP_RTOL="${KSP_RTOL:-1e-6}"
KSP_MAX_IT="${KSP_MAX_IT:-3000}"
KSP_TYPE="${KSP_TYPE:-fgmres}"
PETSC_EXTRA_OPTIONS="${PETSC_EXTRA_OPTIONS:--ksp_gmres_restart 80 -ksp_norm_type unpreconditioned -ksp_monitor_true_residual -pc_hypre_type boomeramg}"
OPT_MOVE="${OPT_MOVE:-0.06}"
OPT_RHO_MIN="${OPT_RHO_MIN:-1e-3}"
OPT_STOP_ON_KSP_DIVERGENCE="${OPT_STOP_ON_KSP_DIVERGENCE:-true}"
OPT_Z_DRAFT_CLOSURE="${OPT_Z_DRAFT_CLOSURE:-true}"
OPT_Z_DRAFT_ETA="${OPT_Z_DRAFT_ETA:-0.5}"
H8_LOAD_CASE="${H8_LOAD_CASE:-0}"
H8_INCLUDE_SPRING_LOAD="${H8_INCLUDE_SPRING_LOAD:-true}"
PREFIX="${PREFIX:-result/cpp_h8_opt_${NX}_${NY}_${NZ}_np${NP}}"

mkdir -p result
make wsl

mpirun -np "${NP}" ./bin/control_arm_cpp \
  -mode optimize \
  -operator h8_matrix_free \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density "${VOID_DENSITY}" \
  -young_modulus "${YOUNG_MODULUS}" \
  -penal "${PENAL}" \
  -emin "${EMIN}" \
  -mask_threshold 0.5 \
  -volfrac "${VOLFRAC}" \
  -opt_max_iter "${ITERS}" \
  -opt_move "${OPT_MOVE}" \
  -opt_filter_radius 1.5 \
  -opt_z_draft_closure "${OPT_Z_DRAFT_CLOSURE}" \
  -opt_z_draft_eta "${OPT_Z_DRAFT_ETA}" \
  -opt_rho_min "${OPT_RHO_MIN}" \
  -opt_load 1.0 \
  -h8_load_case "${H8_LOAD_CASE}" \
  -h8_include_spring_load "${H8_INCLUDE_SPRING_LOAD}" \
  -opt_ksp_rtol "${KSP_RTOL}" \
  -opt_ksp_max_it "${KSP_MAX_IT}" \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval 2 \
  -opt_checkpoint_prefix "${PREFIX}_checkpoint" \
  -opt_write_final_vtk true \
  -opt_vtk_file "${PREFIX}_final.vtk" \
  -output_prefix "${PREFIX}" \
  -ksp_type "${KSP_TYPE}" \
  -h8_pc_type "${H8_PC_TYPE}" \
  -h8_aux_rebuild_interval "${H8_AUX_REBUILD_INTERVAL}" \
  -opt_stop_on_ksp_divergence "${OPT_STOP_ON_KSP_DIVERGENCE}" \
  -ksp_converged_reason \
  ${PETSC_EXTRA_OPTIONS} \
  -log_view ":${PREFIX}_petsc.log" < /dev/null
