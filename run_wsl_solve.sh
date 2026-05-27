#!/bin/sh
set -eu

NP="${NP:-4}"
NX="${NX:-80}"
NY="${NY:-40}"
NZ="${NZ:-24}"
PREFIX="${PREFIX:-result/cpp_solve_${NX}_${NY}_${NZ}_np${NP}}"

mkdir -p result
make wsl

mpirun -np "${NP}" ./bin/control_arm_cpp \
  -mode solve \
  -operator low_order \
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
  -pc_type gamg \
  -pc_gamg_type agg \
  -mg_levels_ksp_type chebyshev \
  -mg_levels_pc_type jacobi \
  -ksp_rtol 1e-6 \
  -ksp_max_it 500 \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log" < /dev/null
