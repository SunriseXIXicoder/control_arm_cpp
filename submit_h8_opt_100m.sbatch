#!/bin/sh
#SBATCH -J ca_h8_100m
#SBATCH -N 2
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=result/%j.h8_100m.out
#SBATCH --error=result/%j.h8_100m.err

set -eu

# 700 * 400 * 120 * 3 = 100,800,000 displacement DOF.
# With filter_radius=1.5, use at most floor((NZ-1)/2) ranks for the current
# z-slab implementation.  The default 2 nodes * 28 tasks = 56 ranks is valid.

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
OPT_MAX_ITER="${OPT_MAX_ITER:-5}"
VOLFRAC="${VOLFRAC:-0.35}"
OPT_FILTER_RADIUS="${OPT_FILTER_RADIUS:-1.5}"
YOUNG_MODULUS="${YOUNG_MODULUS:-2.1e11}"
KSP_MAX_IT="${KSP_MAX_IT:-4000}"
PREFIX="${OUTPUT_PREFIX:-result/h8_100m_${NX}_${NY}_${NZ}_${SLURM_JOB_ID}}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p bin result
make all

echo "Running 100M-DOF H8 matrix-free optimization"
echo "  nodes=${SLURM_JOB_NUM_NODES:-unknown}"
echo "  tasks=${SLURM_NTASKS:-unknown}"
echo "  nx=${NX} ny=${NY} nz=${NZ}"
echo "  dof=$((NX * NY * NZ * 3))"
echo "  opt_max_iter=${OPT_MAX_ITER}"
echo "  opt_filter_radius=${OPT_FILTER_RADIUS}"
echo "  output_prefix=${PREFIX}"

srun -n "$SLURM_NTASKS" ./bin/control_arm_cpp \
  -mode optimize \
  -operator h8_matrix_free \
  -nx "${NX}" -ny "${NY}" -nz "${NZ}" \
  -control_arm_mask true \
  -void_density 0.02 \
  -young_modulus "${YOUNG_MODULUS}" \
  -mask_threshold 0.5 \
  -volfrac "${VOLFRAC}" \
  -opt_max_iter "${OPT_MAX_ITER}" \
  -opt_move 0.08 \
  -opt_filter_radius "${OPT_FILTER_RADIUS}" \
  -opt_rho_min 1e-3 \
  -opt_load 1.0 \
  -opt_ksp_rtol 1e-6 \
  -opt_ksp_max_it "${KSP_MAX_IT}" \
  -opt_write_final_vtk false \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval 1 \
  -opt_checkpoint_prefix "${PREFIX}_checkpoint" \
  -output_prefix "${PREFIX}" \
  -ksp_type cg \
  -pc_type jacobi \
  -ksp_converged_reason \
  -log_view ":${PREFIX}_petsc.log"
