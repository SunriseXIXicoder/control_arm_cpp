#!/bin/sh
set -eu

# NELX/NELY/NELZ follow the MATLAB coarse-element convention.
# The PETSc solver uses nodal counts, so it runs (NELX+1)x(NELY+1)x(NELZ+1).
NP="${NP:-2}"
NELX="${NELX:-40}"
NELY="${NELY:-20}"
NELZ="${NELZ:-8}"
NX=$((NELX + 1))
NY=$((NELY + 1))
NZ=$((NELZ + 1))
DOMAIN_HEIGHT="${DOMAIN_HEIGHT:-0.08}"
PREFIX="${PREFIX:-result/ems_ann_control_bc_${NELX}_${NELY}_${NELZ}_np${NP}}"
EMS_SUB_N="${EMS_SUB_N:-5}"
ANN_DIR="${ANN_DIR:-../input_${EMS_SUB_N}}"
LOAD_CASE="${LOAD_CASE:-2}"
EMIN="${EMIN:-1e-3}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
KSP_MAX_IT="${KSP_MAX_IT:-1000}"
KSP_ATOL="${KSP_ATOL:-1e-12}"
WRITE_VTK="${WRITE_VTK:-true}"

mkdir -p result
make wsl

mpirun -np "${NP}" ./bin/control_arm_cpp \
  -mode solve \
  -operator emsfem_ann \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -domain_height "${DOMAIN_HEIGHT}" \
  -control_arm_mask true \
  -void_density 0.0 \
  -young_modulus "${YOUNG_MODULUS}" \
  -emin "${EMIN}" \
  -ems_ann_dir "${ANN_DIR}" \
  -ems_sub_n "${EMS_SUB_N}" \
  -ems_cache_element_matrices true \
  -ems_cache_gib_limit 2 \
  -control_arm_bc true \
  -load_case "${LOAD_CASE}" \
  -include_spring_load true \
  -load 1.0 \
  -write_structured_vtk "${WRITE_VTK}" \
  -structured_vtk_file "${PREFIX}.vtk" \
  -output_prefix "${PREFIX}" \
  -ksp_type gmres \
  -ksp_gmres_restart 200 \
  -pc_type jacobi \
  -ksp_rtol 1e-5 \
  -ksp_atol "${KSP_ATOL}" \
  -ksp_max_it "${KSP_MAX_IT}" \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log" < /dev/null
