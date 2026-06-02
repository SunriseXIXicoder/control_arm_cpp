#!/bin/sh
set -eu

NP="${NP:-4}"
NX="${NX:-80}"
NY="${NY:-40}"
NZ="${NZ:-24}"
PREFIX="${PREFIX:-result/wsl_smoke_${NX}_${NY}_${NZ}_np${NP}}"
WRITE_VTK="${WRITE_VTK:-true}"
VTK_ARGS=""
if [ "${WRITE_VTK}" = "true" ] || [ "${WRITE_VTK}" = "1" ]; then
  VTK_ARGS="-write_vtk true -vtk_file ${PREFIX}.vtk"
fi

mkdir -p bin result
make wsl

echo "Running WSL PETSc smoke test: ${NX} x ${NY} x ${NZ}, NP=${NP}"
# shellcheck disable=SC2086
mpirun -np "${NP}" ./bin/control_arm_petsc \
  -operator low_order \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -load 1.0 \
  -output_prefix "${PREFIX}" \
  ${VTK_ARGS} \
  -ksp_type cg \
  -pc_type gamg \
  -pc_gamg_type agg \
  -mg_levels_ksp_type chebyshev \
  -mg_levels_pc_type jacobi \
  -ksp_rtol 1e-6 \
  -ksp_max_it 500 \
  -ksp_converged_reason \
  -ksp_monitor_true_residual \
  -log_view ":${PREFIX}_petsc.log" \
  < /dev/null

echo "Summary: ${PREFIX}_summary.txt"
echo "PETSc log: ${PREFIX}_petsc.log"
if [ "${WRITE_VTK}" = "true" ] || [ "${WRITE_VTK}" = "1" ]; then
  echo "VTK: ${PREFIX}.vtk"
fi
