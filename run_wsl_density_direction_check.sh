#!/bin/sh
set -eu

NP="${NP:-4}"
NX="${NX:-30}"
NY="${NY:-16}"
NZ="${NZ:-10}"

mkdir -p result
make wsl

for DIRS in "+x,-x" "+y,-y" "+z,-z" "+x,-x,+y,-y,+z,-z"; do
  TAG=$(printf "%s" "$DIRS" | tr '+-' 'pm' | tr ',' '_')
  PREFIX="result/density_dirs_${TAG}_${NX}_${NY}_${NZ}_np${NP}"
  mpirun -np "${NP}" ./bin/control_arm_cpp \
    -mode density \
    -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
    -control_arm_mask true \
    -void_density 0.02 \
    -filter_radius 1.5 \
    -draft_dirs "${DIRS}" \
    -draft_combine max \
    -draft_radius 1 \
    -draft_pnorm 8 \
    -draft_beta 8 \
    -draft_eta 0.5 \
    -output_prefix "${PREFIX}" \
    -write_density_vtk false \
    -write_density_binary false \
    -log_view ":${PREFIX}_petsc.log" < /dev/null
done
