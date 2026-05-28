#!/bin/sh
set -eu

NP="${NP:-4}"
NX="${NX:-40}"
NY="${NY:-20}"
NZ="${NZ:-12}"
DRAFT_DIRS="${DRAFT_DIRS:-+x,-x,+y,-y,+z,-z}"
PREFIX="${PREFIX:-result/cpp_density_${NX}_${NY}_${NZ}_np${NP}}"

mkdir -p result
make wsl

mpirun -np "${NP}" ./bin/control_arm_cpp \
  -mode density \
  -operator low_order \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density 0.02 \
  -filter_radius 1.5 \
  -draft_dirs "${DRAFT_DIRS}" \
  -draft_combine max \
  -draft_radius 1 \
  -draft_pnorm 8 \
  -draft_beta 8 \
  -draft_eta 0.5 \
  -output_prefix "${PREFIX}" \
  -write_density_vtk true \
  -density_vtk_file "${PREFIX}_density.vtk" \
  -write_density_binary true \
  -density_binary_file "${PREFIX}_rho_projected.petscbin" \
  -log_view ":${PREFIX}_petsc.log" < /dev/null
