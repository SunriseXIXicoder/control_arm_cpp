#!/bin/bash
#SBATCH -J ca_ems_vtk
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=result/%j.%x.out
#SBATCH --error=result/%j.%x.err

set -euo pipefail
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "Job failed at line ${LINENO:-unknown} with exit code ${code}" >&2; fi' EXIT

NX="${NX:-325}"
NY="${NY:-186}"
NZ="${NZ:-56}"
EMS_SUB_N="${EMS_SUB_N:-5}"
DOMAIN_HEIGHT="${DOMAIN_HEIGHT:-0.08}"
STRIDE="${STRIDE:-1}"
MAX_CELLS="${MAX_CELLS:-0}"

if [ -z "${DENSITY_FILE:-}" ]; then
  echo "Set DENSITY_FILE to the EMsFEM ANN *_rho_phys.petscbin checkpoint." >&2
  exit 2
fi

JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-manual}}"
VTK_FILE="${VTK_FILE:-result/ems_ann_full_fine_${NX}_${NY}_${NZ}_${JOB_ID}.vtk}"
MASK_ARGS=()
if [ -n "${MASK_FILE:-}" ]; then
  MASK_ARGS=(--mask "${MASK_FILE}")
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p result logs

echo "Writing full EMsFEM ANN fine-grid VTK"
echo "  density=${DENSITY_FILE}"
echo "  mask=${MASK_FILE:-none}"
echo "  vtk=${VTK_FILE}"
echo "  nx=${NX} ny=${NY} nz=${NZ} sub_n=${EMS_SUB_N}"
echo "  stride=${STRIDE} max_cells=${MAX_CELLS}"
echo "  start=$(date)"

python3 tools/convert_ems_ann_checkpoint_vtk.py \
  --density "${DENSITY_FILE}" \
  "${MASK_ARGS[@]}" \
  --out "${VTK_FILE}" \
  --nx "${NX}" --ny "${NY}" --nz "${NZ}" \
  --sub-n "${EMS_SUB_N}" \
  --height "${DOMAIN_HEIGHT}" \
  --stride "${STRIDE}" \
  --max-cells "${MAX_CELLS}"

echo "Finished at: $(date)"
