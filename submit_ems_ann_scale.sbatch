#!/bin/sh
#SBATCH -N 2
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=%j_ems_ann.out
#SBATCH --error=%j_ems_ann.err

set -eu

# Adjust these paths for the cluster.
export PETSC_DIR="${PETSC_DIR:-/data/app/petsc}"
export PETSC_ARCH="${PETSC_ARCH:-}"

# PETSc nodal counts.  DOF = 3*NX*NY*NZ.
NX="${NX:-700}"
NY="${NY:-400}"
NZ="${NZ:-120}"

ANN_DIR="${ANN_DIR:-input_5}"
PREFIX="${PREFIX:-result/ems_ann_${NX}_${NY}_${NZ}_${SLURM_JOB_ID}}"
LOAD_CASE="${LOAD_CASE:-2}"
CONTROL_ARM_BC="${CONTROL_ARM_BC:-true}"
EMIN="${EMIN:-1e-6}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
CACHE_K="${CACHE_K:-true}"
CACHE_GIB_LIMIT="${CACHE_GIB_LIMIT:-8}"
KSP_TYPE="${KSP_TYPE:-gmres}"
KSP_MAX_IT="${KSP_MAX_IT:-3000}"
KSP_RTOL="${KSP_RTOL:-1e-5}"
KSP_ATOL="${KSP_ATOL:-1e-8}"

source /data/app/intel2022.3.1/setvars.sh > /dev/null 2>&1 || true
ulimit -s unlimited
ulimit -l unlimited

mkdir -p result bin
make

TOTAL_TASKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

srun -n "${TOTAL_TASKS}" ./bin/control_arm_cpp \
  -mode solve \
  -operator emsfem_ann \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density 0.0 \
  -young_modulus "${YOUNG_MODULUS}" \
  -emin "${EMIN}" \
  -ems_ann_dir "${ANN_DIR}" \
  -ems_cache_element_matrices "${CACHE_K}" \
  -ems_cache_gib_limit "${CACHE_GIB_LIMIT}" \
  -control_arm_bc "${CONTROL_ARM_BC}" \
  -load_case "${LOAD_CASE}" \
  -include_spring_load true \
  -load 1.0 \
  -write_structured_vtk false \
  -write_solution false \
  -output_prefix "${PREFIX}" \
  -ksp_type "${KSP_TYPE}" \
  -ksp_gmres_restart 200 \
  -pc_type jacobi \
  -ksp_rtol "${KSP_RTOL}" \
  -ksp_atol "${KSP_ATOL}" \
  -ksp_max_it "${KSP_MAX_IT}" \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log"
