#!/bin/sh
set -eu

NX="${NX:-12}"
NY="${NY:-7}"
NZ="${NZ:-8}"
ITERS="${ITERS:-2}"
VOLFRAC="${VOLFRAC:-0.35}"
export NX NY NZ ITERS VOLFRAC

mkdir -p result
make wsl

for NP in 1 2; do
  PREFIX="result/h8_opt_consistency_${NX}_${NY}_${NZ}_np${NP}"
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
    -opt_write_final_vtk false \
    -output_prefix "${PREFIX}" \
    -ksp_type cg \
    -pc_type jacobi \
    -log_view ":${PREFIX}_petsc.log" < /dev/null
done

python3 - <<'PY'
import os
from pathlib import Path

nx = os.environ["NX"]
ny = os.environ["NY"]
nz = os.environ["NZ"]
base = Path("result")

def read_value(path, key):
    for line in path.read_text().splitlines():
        if line.startswith(key + "="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(f"Missing {key} in {path}")

np1 = base / f"h8_opt_consistency_{nx}_{ny}_{nz}_np1_opt_summary.txt"
np2 = base / f"h8_opt_consistency_{nx}_{ny}_{nz}_np2_opt_summary.txt"
c1 = read_value(np1, "final_compliance")
c2 = read_value(np2, "final_compliance")
v1 = read_value(np1, "final_volume")
v2 = read_value(np2, "final_volume")
crel = abs(c1 - c2) / max(abs(c1), 1.0)
vrel = abs(v1 - v2) / max(abs(v1), 1.0)
print(f"np1 final_compliance={c1:.12e}, final_volume={v1:.12e}")
print(f"np2 final_compliance={c2:.12e}, final_volume={v2:.12e}")
print(f"compliance_relative_difference={crel:.12e}")
print(f"volume_relative_difference={vrel:.12e}")
if crel > 1e-7 or vrel > 1e-8:
    raise SystemExit("H8 optimization rank consistency check failed")
PY
