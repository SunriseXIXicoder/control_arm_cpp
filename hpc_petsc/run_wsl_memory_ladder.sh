#!/bin/sh
set -eu

NP="${NP:-4}"
CASES="${CASES:-80 40 24;120 60 32;160 80 40}"
WRITE_VTK="${WRITE_VTK:-false}"

mkdir -p bin result
make wsl

echo "Running WSL memory ladder with NP=${NP}"
echo "${CASES}" | tr ';' '\n' | while read -r NX NY NZ; do
  if [ -z "${NX:-}" ]; then
    continue
  fi
  DOF=$((3 * NX * NY * NZ))
  PREFIX="result/wsl_ladder_${NX}_${NY}_${NZ}_np${NP}"
  VTK_ARGS=""
  if [ "${WRITE_VTK}" = "true" ] || [ "${WRITE_VTK}" = "1" ]; then
    VTK_ARGS="-write_vtk true -vtk_file ${PREFIX}.vtk"
  fi
  echo "Case ${NX} x ${NY} x ${NZ}, DOF=${DOF}"
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
    -log_view ":${PREFIX}_petsc.log" \
    < /dev/null
done

echo "Ladder summaries:"
ls -1 result/wsl_ladder_*_summary.txt 2>/dev/null || true
