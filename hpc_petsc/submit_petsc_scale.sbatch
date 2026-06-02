#!/bin/sh
#SBATCH -J ca_petsc_scale
#SBATCH -N 4
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=result/%j.scale.out
#SBATCH --error=result/%j.scale.err

set -eu

# Submit examples:
#   sbatch --export=ALL,NX=150,NY=80,NZ=30 submit_petsc_scale.sbatch
#   sbatch --nodes=4 --ntasks-per-node=28 --export=ALL,NX=700,NY=400,NZ=120 submit_petsc_scale.sbatch

source /data/app/intel2022.3.1/setvars.sh > /dev/null 2>&1 || true

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

NX="${NX:-700}"
NY="${NY:-400}"
NZ="${NZ:-120}"
OPERATOR="${OPERATOR:-low_order}"
PREFIX="${OUTPUT_PREFIX:-result/petsc_${NX}_${NY}_${NZ}_${SLURM_JOB_ID}}"
WRITE_VTK="${WRITE_VTK:-false}"
VTK_ARGS=""
if [ "${WRITE_VTK}" = "true" ] || [ "${WRITE_VTK}" = "1" ]; then
  VTK_ARGS="-write_vtk true -vtk_file ${PREFIX}.vtk"
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p bin result
make all

echo "Running PETSc scale case"
echo "  nodes=${SLURM_JOB_NUM_NODES:-unknown}"
echo "  tasks=${SLURM_NTASKS:-unknown}"
echo "  nx=${NX} ny=${NY} nz=${NZ}"
echo "  operator=${OPERATOR}"
echo "  output_prefix=${PREFIX}"
echo "  write_vtk=${WRITE_VTK}"

# shellcheck disable=SC2086
srun -n "$SLURM_NTASKS" ./bin/control_arm_petsc \
  -operator "${OPERATOR}" \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
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
