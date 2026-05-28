#!/bin/sh
set -eu

NX="${NX:-12}"
NY="${NY:-6}"
NZ="${NZ:-4}"
RTOL="${RTOL:-1e-8}"
export NX NY NZ

mkdir -p result
make wsl

for NP in 1 4; do
  PREFIX="result/h8_uniform_${NX}_${NY}_${NZ}_np${NP}"
  mpirun -np "${NP}" ./bin/control_arm_cpp \
    -mode solve \
    -operator h8_matrix_free \
    -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
    -control_arm_mask false \
    -load 1.0 \
    -output_prefix "${PREFIX}" \
    -ksp_type cg \
    -pc_type jacobi \
    -ksp_rtol "${RTOL}" \
    -ksp_max_it 2000 \
    -ksp_converged_reason \
    -log_view ":${PREFIX}_petsc.log" < /dev/null
done

python3 - <<'PY'
import os
from pathlib import Path

def read_value(path, key):
    for line in Path(path).read_text().splitlines():
        if line.startswith(key + "="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(f"Missing {key} in {path}")

base = Path("result")
stem = f"h8_uniform_{os.environ['NX']}_{os.environ['NY']}_{os.environ['NZ']}"
np1 = base / f"{stem}_np1_solve_summary.txt"
np4 = base / f"{stem}_np4_solve_summary.txt"
c1 = read_value(np1, "compliance")
c4 = read_value(np4, "compliance")
rel = abs(c1 - c4) / max(abs(c1), 1.0)
print(f"np1 compliance={c1:.12e}")
print(f"np4 compliance={c4:.12e}")
print(f"relative_difference={rel:.12e}")
if rel > 1e-10:
    raise SystemExit("H8 matrix-free rank consistency check failed")
PY
