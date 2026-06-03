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
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${POSTSOLVE_CHECKPOINT:-}}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-}"

strip_petsc_info_suffix() {
  case "$1" in
    *.petscbin.info) printf '%s\n' "${1%.info}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

checkpoint_prefix_from_file() {
  local path
  path="$(strip_petsc_info_suffix "$1")"
  case "${path}" in
    *_final_rho_phys.petscbin) printf '%s\n' "${path%_final_rho_phys.petscbin}" ;;
    *_final_rho_design.petscbin) printf '%s\n' "${path%_final_rho_design.petscbin}" ;;
    *_final_mask.petscbin) printf '%s\n' "${path%_final_mask.petscbin}" ;;
    *_final_u.petscbin) printf '%s\n' "${path%_final_u.petscbin}" ;;
    *_final_u_case_[0-9][0-9][0-9].petscbin) printf '%s\n' "${path%_final_u_case_[0-9][0-9][0-9].petscbin}" ;;
    *) printf '%s\n' "${path}" ;;
  esac
}

resolve_result_path() {
  local path="$1"
  if [ -f "${path}" ]; then
    printf '%s\n' "${path}"
  elif [[ "${path}" != */* ]] && [ -f "result/${path}" ]; then
    printf '%s\n' "result/${path}"
  else
    printf '%s\n' "${path}"
  fi
}

if [ -n "${CHECKPOINT_FILE}" ] && [ -z "${CHECKPOINT_PREFIX}" ]; then
  CHECKPOINT_PREFIX="$(checkpoint_prefix_from_file "${CHECKPOINT_FILE}")"
fi
if [ -n "${CHECKPOINT_PREFIX}" ]; then
  DENSITY_FILE="${DENSITY_FILE:-${CHECKPOINT_PREFIX}_final_rho_phys.petscbin}"
  MASK_FILE="${MASK_FILE:-${CHECKPOINT_PREFIX}_final_mask.petscbin}"
fi
if [ -n "${DENSITY_FILE:-}" ]; then
  DENSITY_FILE="$(strip_petsc_info_suffix "${DENSITY_FILE}")"
fi
if [ -n "${MASK_FILE:-}" ]; then
  MASK_FILE="$(strip_petsc_info_suffix "${MASK_FILE}")"
fi
if [ -n "${DISPLACEMENT_FILE:-}" ]; then
  DISPLACEMENT_FILE="$(strip_petsc_info_suffix "${DISPLACEMENT_FILE}")"
fi

if [ -z "${DENSITY_FILE:-}" ]; then
  echo "Set DENSITY_FILE to the EMsFEM ANN *_rho_phys.petscbin checkpoint, or CHECKPOINT_FILE/CHECKPOINT_PREFIX for auto inference." >&2
  exit 2
fi

JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-manual}}"
VTK_FILE="${VTK_FILE:-result/ems_ann_full_fine_${NX}_${NY}_${NZ}_${JOB_ID}.vtk}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p result logs
DENSITY_FILE="$(resolve_result_path "${DENSITY_FILE}")"
if [ -n "${MASK_FILE:-}" ]; then
  MASK_FILE="$(resolve_result_path "${MASK_FILE}")"
fi
if [ -n "${DISPLACEMENT_FILE:-}" ]; then
  DISPLACEMENT_FILE="$(resolve_result_path "${DISPLACEMENT_FILE}")"
fi
if [ ! -f "${DENSITY_FILE}" ]; then
  echo "Density file not found: ${DENSITY_FILE}" >&2
  echo "If you copied a .petscbin.info name, the script strips .info and expects the matching .petscbin data file." >&2
  exit 2
fi
if [ -n "${MASK_FILE:-}" ] && [ ! -f "${MASK_FILE}" ]; then
  echo "Mask file not found: ${MASK_FILE}" >&2
  echo "If you copied a .petscbin.info name, the script strips .info and expects the matching .petscbin data file." >&2
  exit 2
fi
if [ -z "${DISPLACEMENT_FILE:-}" ]; then
  CANDIDATE_U="${DENSITY_FILE%_rho_phys.petscbin}_u.petscbin"
  if [ "${CANDIDATE_U}" != "${DENSITY_FILE}_u.petscbin" ] &&
     [ -f "${CANDIDATE_U}" ]; then
    DISPLACEMENT_FILE="${CANDIDATE_U}"
  fi
fi
if [ -n "${DISPLACEMENT_FILE:-}" ] && [ ! -f "${DISPLACEMENT_FILE}" ]; then
  echo "Displacement file not found: ${DISPLACEMENT_FILE}" >&2
  exit 2
fi
MASK_ARGS=()
if [ -n "${MASK_FILE:-}" ]; then
  MASK_ARGS=(--mask "${MASK_FILE}")
fi
DISPLACEMENT_ARGS=()
if [ -n "${DISPLACEMENT_FILE:-}" ]; then
  DISPLACEMENT_ARGS=(--displacement "${DISPLACEMENT_FILE}")
fi

echo "Writing full EMsFEM ANN fine-grid VTK"
echo "  density=${DENSITY_FILE}"
echo "  mask=${MASK_FILE:-none}"
echo "  displacement=${DISPLACEMENT_FILE:-none}"
echo "  vtk=${VTK_FILE}"
echo "  nx=${NX} ny=${NY} nz=${NZ} sub_n=${EMS_SUB_N}"
echo "  stride=${STRIDE} max_cells=${MAX_CELLS}"
echo "  start=$(date)"

python3 tools/convert_ems_ann_checkpoint_vtk.py \
  --density "${DENSITY_FILE}" \
  "${MASK_ARGS[@]}" \
  "${DISPLACEMENT_ARGS[@]}" \
  --out "${VTK_FILE}" \
  --nx "${NX}" --ny "${NY}" --nz "${NZ}" \
  --sub-n "${EMS_SUB_N}" \
  --height "${DOMAIN_HEIGHT}" \
  --stride "${STRIDE}" \
  --max-cells "${MAX_CELLS}"

echo "Finished at: $(date)"
