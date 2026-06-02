#!/bin/sh
#SBATCH -J ca_petsc_100m
#SBATCH -N 4
#SBATCH -n 112
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=result/%j.100m.out
#SBATCH --error=result/%j.100m.err

set -eu

source /data/app/intel2022.3.1/setvars.sh > /dev/null 2>&1 || true

# Modify these two lines to match the PETSc installation on the cluster.
# A prefix install often needs only PETSC_DIR. A source-tree build may also
# need PETSC_ARCH, for example arch-linux-c-opt.
export PETSC_DIR="${PETSC_DIR:-/data/app/petsc}"
export PETSC_ARCH="${PETSC_ARCH:-}"
export PATH="${PETSC_DIR}/bin:${PATH}"
if [ -n "${PETSC_ARCH}" ]; then
  export LD_LIBRARY_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib:${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${PETSC_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -s unlimited
ulimit -l unlimited || true

cd "$SLURM_SUBMIT_DIR"
mkdir -p bin result
make all
PREFIX="result/petsc_100m_700_400_120_${SLURM_JOB_ID}"
WRITE_VTK="${WRITE_VTK:-false}"
VTK_ARGS=""
if [ "${WRITE_VTK}" = "true" ] || [ "${WRITE_VTK}" = "1" ]; then
  VTK_ARGS="-write_vtk true -vtk_file ${PREFIX}.vtk"
fi

# 3 * 700 * 400 * 120 = 100,800,000 unknowns.
# Start from submit_petsc_debug.sbatch first, then this job.
# shellcheck disable=SC2086
srun -n "$SLURM_NTASKS" ./bin/control_arm_petsc \
  -operator low_order \
  -nx 700 -ny 400 -nz 120 \
  -load 1.0 \
  -output_prefix "${PREFIX}" \
  ${VTK_ARGS} \
  -ksp_type cg \
  -pc_type gamg \
  -pc_gamg_type agg \
  -mg_levels_ksp_type chebyshev \
  -mg_levels_pc_type jacobi \
  -ksp_rtol 1e-6 \
  -ksp_max_it 500 \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log"
