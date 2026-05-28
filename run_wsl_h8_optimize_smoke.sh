#!/bin/sh
set -eu

NP="${NP:-4}"
NX="${NX:-14}"
NY="${NY:-8}"
NZ="${NZ:-12}"
ITERS="${ITERS:-4}"
VOLFRAC="${VOLFRAC:-0.35}"
PREFIX="${PREFIX:-result/cpp_h8_opt_${NX}_${NY}_${NZ}_np${NP}}"

mkdir -p result
make wsl

mpirun -np "${NP}" ./bin/control_arm_cpp \
  -mode optimize \
  -operator h8_matrix_free \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density 0.02 \
  -mask_threshold 0.5 \
  -volfrac "${VOLFRAC}" \
  -opt_max_iter "${ITERS}" \
  -opt_move 0.06 \
  -opt_filter_radius 1.5 \
  -opt_rho_min 1e-3 \
  -opt_load 1.0 \
  -opt_ksp_rtol 1e-6 \
  -opt_ksp_max_it 3000 \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval 2 \
  -opt_checkpoint_prefix "${PREFIX}_checkpoint" \
  -opt_write_final_vtk true \
  -opt_vtk_file "${PREFIX}_final.vtk" \
  -output_prefix "${PREFIX}" \
  -ksp_type cg \
  -pc_type jacobi \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log" < /dev/null
