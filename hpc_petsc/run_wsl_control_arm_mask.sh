#!/bin/sh
set -eu

NP="${NP:-4}"
NX="${NX:-80}"
NY="${NY:-40}"
NZ="${NZ:-24}"
PREFIX="${PREFIX:-result/control_arm_mask_${NX}_${NY}_${NZ}_np${NP}}"

mkdir -p bin result
make wsl

echo "Running WSL control-arm mask diagnostic: ${NX} x ${NY} x ${NZ}, NP=${NP}"
mpirun -np "${NP}" ./bin/control_arm_petsc \
  -operator low_order \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density 0.02 \
  -load 1.0 \
  -output_prefix "${PREFIX}" \
  -write_vtk true \
  -vtk_file "${PREFIX}.vtk" \
  -write_mask_vtk true \
  -mask_vtk_file "${PREFIX}_solid.vtk" \
  -ksp_type cg \
  -pc_type gamg \
  -pc_gamg_type agg \
  -mg_levels_ksp_type chebyshev \
  -mg_levels_pc_type jacobi \
  -ksp_rtol 1e-6 \
  -ksp_max_it 500 \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log" \
  < /dev/null

echo "Summary: ${PREFIX}_summary.txt"
echo "Structured VTK: ${PREFIX}.vtk"
echo "Solid mask VTK: ${PREFIX}_solid.vtk"
