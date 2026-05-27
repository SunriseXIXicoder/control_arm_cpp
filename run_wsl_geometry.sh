#!/bin/sh
set -eu

NP="${NP:-4}"
NX="${NX:-80}"
NY="${NY:-40}"
NZ="${NZ:-24}"
PREFIX="${PREFIX:-result/cpp_geometry_${NX}_${NY}_${NZ}_np${NP}}"

mkdir -p result
make wsl

mpirun -np "${NP}" ./bin/control_arm_cpp \
  -mode geometry \
  -operator low_order \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density 0.02 \
  -mask_threshold 0.5 \
  -output_prefix "${PREFIX}" \
  -write_structured_vtk true \
  -structured_vtk_file "${PREFIX}.vtk" \
  -write_solid_vtk true \
  -solid_vtk_file "${PREFIX}_solid.vtk" \
  -write_density_binary true \
  -density_binary_file "${PREFIX}_rho.petscbin" \
  -max_vtk_cells 300000 \
  -log_view ":${PREFIX}_petsc.log" < /dev/null
