#!/bin/sh
set -eu

NP="${NP:-4}"
NX="${NX:-20}"
NY="${NY:-10}"
NZ="${NZ:-6}"
PREFIX="${PREFIX:-result/cpp_h8_${NX}_${NY}_${NZ}_np${NP}}"

mkdir -p result
make wsl

mpirun -np "${NP}" ./bin/control_arm_cpp \
  -mode solve \
  -operator h8_matrix_free \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density 0.02 \
  -load 1.0 \
  -output_prefix "${PREFIX}" \
  -write_structured_vtk true \
  -structured_vtk_file "${PREFIX}.vtk" \
  -write_solid_vtk true \
  -solid_vtk_file "${PREFIX}_solid.vtk" \
  -max_vtk_cells 300000 \
  -ksp_type cg \
  -pc_type jacobi \
  -ksp_rtol 1e-6 \
  -ksp_max_it 2000 \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log" < /dev/null
