#!/bin/bash
#SBATCH -J ca_h8_postvtk
#SBATCH -N 8
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
trap 'echo "Job failed at line ${LINENO} with exit code $?" >&2' ERR

module load gcc/11.3.0 >/dev/null 2>&1 || true
module load intel/2020u4 >/dev/null 2>&1 || true

export PETSC_DIR="${PETSC_DIR:-/data/home/dlutmxy/petsc/petsc-v3.19.3}"
export PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-debug}"
export LD_LIBRARY_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"

NX="${NX:-700}"
NY="${NY:-400}"
NZ="${NZ:-120}"
DOMAIN_HEIGHT="${DOMAIN_HEIGHT:-0.08}"
POST_STRIDE="${POST_STRIDE:-5}"
POST_MAX_SAMPLES="${POST_MAX_SAMPLES:-2000000}"
OPT_FILTER_RADIUS="${OPT_FILTER_RADIUS:-1.5}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
CXXSTD="${CXXSTD:-c++11}"

if [ -z "${SOURCE_PREFIX:-}" ]; then
  echo "Set SOURCE_PREFIX, for example:" >&2
  echo "  sbatch --export=ALL,SOURCE_PREFIX=result/h8_100m_1step_700_400_120_8518873 submit_h8_postprocess_100m.sbatch" >&2
  exit 2
fi

DENSITY_FILE="${DENSITY_FILE:-${SOURCE_PREFIX}_checkpoint_final_rho_phys.petscbin}"
MASK_FILE="${MASK_FILE:-${SOURCE_PREFIX}_checkpoint_final_mask.petscbin}"
VTK_FILE="${VTK_FILE:-${SOURCE_PREFIX}_rho_phys_stride${POST_STRIDE}.vtk}"
PREFIX="${PREFIX:-${SOURCE_PREFIX}_post_stride${POST_STRIDE}}"

mkdir -p bin result logs

echo "Postprocess H8 PETSc checkpoint"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Run host: $(hostname)"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "PETSC_DIR=${PETSC_DIR}"
echo "PETSC_ARCH=${PETSC_ARCH}"
echo "density=${DENSITY_FILE}"
echo "mask=${MASK_FILE}"
echo "vtk=${VTK_FILE}"
echo "domain_height=${DOMAIN_HEIGHT} m"

test -f "${DENSITY_FILE}"
test -f "${MASK_FILE}"

make -B all CXXSTD="${CXXSTD}" PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}"

if [ "${MPI_LAUNCHER}" = "srun" ]; then
  LAUNCH_CMD=(srun -n "${SLURM_NTASKS:-224}")
else
  LAUNCH_CMD=("${MPI_LAUNCHER}" -np "${SLURM_NTASKS:-224}")
fi

echo "launch_command=${LAUNCH_CMD[*]} ./bin/control_arm_cpp -mode h8_postprocess ..."
"${LAUNCH_CMD[@]}" ./bin/control_arm_cpp \
  -mode h8_postprocess \
  -operator h8_matrix_free \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -domain_height "${DOMAIN_HEIGHT}" \
  -opt_filter_radius "${OPT_FILTER_RADIUS}" \
  -density_file "${DENSITY_FILE}" \
  -mask_file "${MASK_FILE}" \
  -post_vtk_file "${VTK_FILE}" \
  -post_stride "${POST_STRIDE}" \
  -post_max_samples "${POST_MAX_SAMPLES}" \
  -output_prefix "${PREFIX}" \
  -log_view ":${PREFIX}_petsc.log"

echo "Finished H8 checkpoint postprocess"
