#!/bin/bash
#SBATCH -J ca_h8_preflight
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=result/%j.%x.out
#SBATCH --error=result/%j.%x.err

set -euxo pipefail

echo "Preflight entered shell: $(date)"
echo "WorkDir=${SLURM_SUBMIT_DIR:-$PWD}"
echo "Host=$(hostname)"
echo "Job=${SLURM_JOB_ID:-manual}"
echo "Nodes=${SLURM_JOB_NODELIST:-unknown}"
echo "Tasks=${SLURM_NTASKS:-unset}"
echo "Bash=${BASH_VERSION:-unknown}"

if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
  [ -f /usr/share/Modules/init/bash ] && . /usr/share/Modules/init/bash || true
fi
if command -v module >/dev/null 2>&1; then
  module load gcc/11.3.0 2>/dev/null || true
  module load intel/2020u4 2>/dev/null || true
fi

PETSC_DIR="${PETSC_DIR:-/data/home/dlutmxy/petsc/petsc-v3.19.3}"
PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-opt-hypre}"
export PETSC_DIR PETSC_ARCH

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p bin result logs
if [ -f scripts/h8_slurm_guard.sh ]; then
  . scripts/h8_slurm_guard.sh
  h8_print_slurm_diagnostics
  h8_guard_slurm_memory_accounting
else
  echo "Warning: scripts/h8_slurm_guard.sh not found; Slurm memory guard skipped" >&2
fi

echo "Checking script bytes"
file submit_h8_opt_100m.sbatch || true
head -1 submit_h8_opt_100m.sbatch | od -An -tx1
bash -n submit_h8_opt_100m.sbatch

echo "Checking PETSc"
test -f "${PETSC_DIR}/lib/petsc/conf/variables"
test -d "${PETSC_DIR}/${PETSC_ARCH}/include"
test -d "${PETSC_DIR}/${PETSC_ARCH}/lib"
echo "PETSC_DIR=${PETSC_DIR}"
echo "PETSC_ARCH=${PETSC_ARCH}"

echo "Checking compilers"
command -v mpicxx
mpicxx --version 2>&1 | head -n 3 || true
make -B all CXXSTD="${CXXSTD:-c++11}" PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}"
test -x ./bin/control_arm_cpp

echo "Checking MPI launcher"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
export I_MPI_HYDRA_BOOTSTRAP="${I_MPI_HYDRA_BOOTSTRAP:-slurm}"
if [ "${MPI_LAUNCHER}" = "srun" ]; then
  srun -n "${SLURM_NTASKS:-2}" hostname
else
  mpirun -np "${SLURM_NTASKS:-2}" hostname
fi

echo "Checking executable startup"
./bin/control_arm_cpp -help >/dev/null 2>&1 || true

echo "Preflight finished: $(date)"
