#!/bin/sh
set -eu

NX="${NX:-20}"
NY="${NY:-12}"
NZ="${NZ:-12}"
DRAFT_DIRS="${DRAFT_DIRS:-+z,-z}"

mkdir -p result
make wsl

for NP in 1 4; do
  PREFIX="result/density_consistency_${NX}_${NY}_${NZ}_np${NP}"
  mpirun -np "${NP}" ./bin/control_arm_cpp \
    -mode density \
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
    -write_density_vtk false \
    -write_density_binary false \
    -log_view ":${PREFIX}_petsc.log" < /dev/null
done

python3 - <<'PY'
import os
from pathlib import Path

nx = os.environ.get("NX", "20")
ny = os.environ.get("NY", "12")
nz = os.environ.get("NZ", "12")
base = Path("result")

def read_value(path, key):
    for line in path.read_text().splitlines():
        if line.startswith(key + "="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(f"Missing {key} in {path}")

np1 = base / f"density_consistency_{nx}_{ny}_{nz}_np1_density_summary.txt"
np4 = base / f"density_consistency_{nx}_{ny}_{nz}_np4_density_summary.txt"
v1 = read_value(np1, "avg_projected")
v4 = read_value(np4, "avg_projected")
rel = abs(v1 - v4) / max(abs(v1), 1.0)
print(f"np1 avg_projected={v1:.12e}")
print(f"np4 avg_projected={v4:.12e}")
print(f"relative_difference={rel:.12e}")
if rel > 1e-10:
    raise SystemExit("Density pipeline rank consistency check failed")
PY
