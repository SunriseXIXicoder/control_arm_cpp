#!/bin/bash
#SBATCH -J ca_mpi_check
#SBATCH -N 2
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=%j.mpi_check.out
#SBATCH --error=%j.mpi_check.err

set -eo pipefail
echo "MPI launcher check entered shell: $(date)"

if [ -f /data/app/intel2022.3.1/setvars.sh ]; then
  . /data/app/intel2022.3.1/setvars.sh > /dev/null 2>&1 || true
fi

TOTAL_TASKS="${SLURM_NPROCS:-${SLURM_NTASKS:-56}}"
MPI_LAUNCHER="${MPI_LAUNCHER:-srun}"
MPI_LAUNCHER_ARGS="${MPI_LAUNCHER_ARGS:-}"

echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-unknown}"
echo "Nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "SLURM_NPROCS=${SLURM_NPROCS:-unset}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "TOTAL_TASKS=${TOTAL_TASKS}"
echo "MPI_LAUNCHER=${MPI_LAUNCHER}"
echo "MPI_LAUNCHER_ARGS=${MPI_LAUNCHER_ARGS}"
echo "which srun: $(command -v srun || true)"
echo "which mpirun: $(command -v mpirun || true)"
echo "which mpiexec: $(command -v mpiexec || true)"
echo "which mpicxx: $(command -v mpicxx || true)"
mpicxx --version 2>&1 | head -n 3 || true

if [ "${MPI_LAUNCHER}" = "srun" ]; then
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -n ${TOTAL_TASKS}"
else
  LAUNCH_CMD="${MPI_LAUNCHER} ${MPI_LAUNCHER_ARGS} -np ${TOTAL_TASKS}"
fi

echo "Running: ${LAUNCH_CMD} hostname"
${LAUNCH_CMD} hostname | sort | uniq -c
echo "MPI launcher check finished: $(date)"
