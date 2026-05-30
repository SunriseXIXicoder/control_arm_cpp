#!/bin/sh
set -eu

NP="${NP:-1}"
NX="${NX:-6}"
NY="${NY:-4}"
NZ="${NZ:-3}"
PREFIX="${PREFIX:-result/ems_ann_smoke_${NX}_${NY}_${NZ}_np${NP}}"
EMS_SUB_N="${EMS_SUB_N:-5}"
ANN_DIR="${ANN_DIR:-../input_${EMS_SUB_N}}"

mkdir -p result
make wsl

mpirun -np "${NP}" ./bin/control_arm_cpp \
  -mode solve \
  -operator emsfem_ann \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density 0.02 \
  -ems_ann_dir "${ANN_DIR}" \
  -ems_sub_n "${EMS_SUB_N}" \
  -ems_cache_element_matrices true \
  -ems_cache_gib_limit 1 \
  -load 1.0 \
  -write_structured_vtk true \
  -structured_vtk_file "${PREFIX}.vtk" \
  -output_prefix "${PREFIX}" \
  -ksp_type cg \
  -pc_type jacobi \
  -ksp_rtol 1e-5 \
  -ksp_max_it 200 \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log" < /dev/null
