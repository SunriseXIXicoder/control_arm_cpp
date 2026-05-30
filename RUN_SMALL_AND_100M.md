# Run Small Tests and 100M-DOF Jobs

## Physical Scale

The C++/PETSc model uses SI units by default.  With
`YOUNG_MODULUS=2.1e11`, force is in N, length is in m, displacement is in m,
and compliance is in J.

The default physical height is now `DOMAIN_HEIGHT=0.08 m`.  The x/y dimensions
preserve the grid aspect ratio:

- `700x400x120` nodes: about `0.470 m x 0.268 m x 0.080 m`
- `41x21x9` nodes, matching a `40x20x8` coarse grid: about
  `0.400 m x 0.200 m x 0.080 m`

Override it with `DOMAIN_HEIGHT=...` in scripts or `-domain_height ...` on the
executable command line.

## WSL Small H8 Optimization Test

Use this first.  It runs the same H8 matrix-free optimizer path as the large
job, but at a small size and writes VTK plus PETSc binary checkpoints.

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
make wsl
NP=4 NX=14 NY=8 NZ=12 ITERS=4 sh run_wsl_h8_optimize_smoke.sh
```

Expected key outputs:

- `result/cpp_h8_opt_14_8_12_np4_history.csv`
- `result/cpp_h8_opt_14_8_12_np4_objective_volume.csv`
- `result/cpp_h8_opt_14_8_12_np4_objective_volume.svg`
- `result/cpp_h8_opt_14_8_12_np4_opt_summary.txt`
- `result/cpp_h8_opt_14_8_12_np4_final.vtk`
- `result/cpp_h8_opt_14_8_12_np4_checkpoint_final_rho_design.petscbin`
- `result/cpp_h8_opt_14_8_12_np4_checkpoint_final_rho_phys.petscbin`

## WSL Consistency Check

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
NX=12 NY=7 NZ=8 ITERS=2 sh run_wsl_h8_optimize_consistency.sh
```

This compares 1-rank and 2-rank runs.  The compliance and volume differences
should be near machine precision.

## WSL ANN + EMsFEM Solve Tests

First run the tiny smoke test.  This verifies JSON ANN loading, 5x5x5
multiscale stiffness recovery, local element-matrix caching, PETSc MatShell,
and VTK output:

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
sh run_wsl_ems_ann_smoke.sh
```

Then run the MATLAB-style `40x20x8` coarse-grid control-arm boundary smoke
test.  The script converts coarse elements to PETSc nodal counts and therefore
passes `41x21x9` internally:

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
sh run_wsl_ems_ann_control_arm_40_20_8.sh
```

The ANN/EMsFEM path supports both solve and optimize:

```sh
-mode solve -operator emsfem_ann
-mode optimize -operator emsfem_ann
```

Run a complete ANN/EMsFEM topology optimization on the same `40x20x8` coarse
grid:

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
ITERS=50 sh run_wsl_ems_ann_optimize_40_20_8.sh
```

The optimization script defaults to `LOAD_CASE=0`, meaning all three old
control-arm load cases are solved separately and combined with the MATLAB AHP
weights.  Use `LOAD_CASE=2` to reproduce only the deep-pothole case.

For a longer production-style local run:

```sh
ITERS=500 WRITE_VTK=false sh run_wsl_ems_ann_optimize_40_20_8.sh
```

It does not store `all_shape_functions`; by default it caches only local
24x24 coarse element matrices per MPI rank.  The optimizer currently uses a
frozen-ANN-shape energy sensitivity approximation to avoid global shape
function/Jacobian storage.  See `ANN_EMSFEM_NOTES.md` for memory details and
current limitations.

## 100M-DOF Slurm Job

Copy or keep the whole `control_arm_cpp` folder on the cluster, then submit:

```sh
cd /path/to/control_arm_cpp
sbatch submit_h8_opt_100m.sbatch
```

Default 100M case:

- `NX=700`
- `NY=400`
- `NZ=120`
- DOF: `100,800,000`
- physical size with default `DOMAIN_HEIGHT=0.08`: about
  `0.470 m x 0.268 m x 0.080 m`
- ranks: default sbatch uses `8 nodes * 28 tasks = 224 ranks`
- operator: H8 matrix-free
- default iterations: `OPT_MAX_ITER=5`
- density filter radius: `OPT_FILTER_RADIUS=1.5`
- H8 preconditioner: `H8_PC_TYPE=block_jacobi`
- no VTK output
- distributed PETSc binary checkpoints enabled

Useful overrides:

```sh
sbatch --export=ALL,OPT_MAX_ITER=10,VOLFRAC=0.35 submit_h8_opt_100m.sbatch
sbatch --nodes=2 --ntasks-per-node=28 --export=ALL,NX=350,NY=200,NZ=80,OPT_MAX_ITER=3 submit_h8_opt_100m.sbatch
sbatch --nodes=8 --ntasks-per-node=28 --export=ALL,NX=700,NY=400,NZ=120,OPT_MAX_ITER=20,VOLFRAC=0.35 submit_h8_opt_100m.sbatch
sbatch --export=ALL,H8_PC_TYPE=jacobi,OPT_MAX_ITER=1 submit_h8_opt_100m.sbatch
```

`H8_PC_TYPE=block_jacobi` uses a matrix-free node-wise 3x3 Jacobi
preconditioner.  It stores only one inverse 3x3 block per displacement node and
keeps the global stiffness matrix implicit.  Use `H8_PC_TYPE=jacobi` only for
old scalar-Jacobi comparison, or `H8_PC_TYPE=petsc` if you want to drive PETSc
with raw `-pc_type` options yourself.

The H8 optimizer uses an x/y/z DMDA process grid, so higher MPI rank counts are
not limited by z slabs only.  With `OPT_FILTER_RADIUS=1.5`, each local element
block must still be at least two elements thick in x, y, and z.  The process
grid is chosen automatically and printed at startup, for example
`H8 optimizer DMDA process grid: px=... py=... pz=...`.  If needed, override it
with `H8_DM_PX/H8_DM_PY/H8_DM_PZ` in the sbatch environment, or with
`-h8_dm_px`, `-h8_dm_py`, and `-h8_dm_pz` on the executable command line.  The
product must equal the MPI rank count.

## H8 KSP Convergence Diagnostics

`ksp_it` is the PETSc Krylov iteration count for the linear elasticity solve
`K u = f`.  If `ksp_it` equals `KSP_MAX_IT` and `ksp_reason_name` is
`KSP_DIVERGED_ITS`, the solve hit the iteration limit and did not converge.
This is not caused by the absolute Young's modulus alone: scaling all stiffness
entries by the same `E` changes displacement magnitude but not the conditioning.
The main causes are mesh refinement, high SIMP contrast, low-density/void
regions, and the current matrix-free H8 preconditioner.

The H8 optimizer defaults to `OPT_STOP_ON_KSP_DIVERGENCE=true`.  When a solve
does not converge, the density update for that failed iteration is skipped and
the latest checkpoint is still written.  This prevents a 250-step run from
continuing with unreliable displacements and sensitivities.

Use a softer startup continuation when diagnosing a new large run:

```sh
sbatch --nodes=8 --ntasks-per-node=28 \
  --export=ALL,RUN_LABEL=h8_10m_kspdiag,NX=325,NY=186,NZ=56,OPT_MAX_ITER=5,\
VOLFRAC=0.35,OPT_MOVE=0.02,KSP_RTOL=1e-3,KSP_MAX_IT=1500,\
PENAL=1.5,EMIN=1e-4,VOID_DENSITY=0.05,OPT_RHO_MIN=1e-3,\
MPI_LAUNCHER=srun \
  submit_h8_opt_100m.sbatch
```

If this converges, tighten in stages:

```sh
# Stage 2
PENAL=2.0 EMIN=1e-5 KSP_RTOL=3e-4

# Stage 3
PENAL=3.0 EMIN=1e-6 KSP_RTOL=1e-4
```

Adding more nodes reduces memory per rank and may reduce wall time per
iteration, but it does not fix a weak preconditioner by itself.  For production
100M-DOF optimization, the next major solver upgrade should be matrix-free
geometric multigrid or an assembled coarse/preconditioning operator; plain
block Jacobi is mostly a memory-safe baseline.

The EMsFEM/ANN optimizer also uses x/y/z fine-density decomposition now.  Its
z-direction draft closure communicates only within ranks that share the same
x-y subblock, so `NP=8` on the `40*20*8` WSL test no longer fails because of
the fine filter radius.  Manual overrides use
`EMS_DM_PX/EMS_DM_PY/EMS_DM_PZ` in the WSL script or `-ems_dm_px`,
`-ems_dm_py`, and `-ems_dm_pz` on the executable command line.

## Large-Job Outputs

For a job prefix such as `result/h8_100m_700_400_120_JOBID`, expect:

- `*_history.csv`
- `*_objective_volume.csv`
- `*_objective_volume.svg`
- `*_opt_summary.txt`
- `*_petsc.log`
- `*_checkpoint_iter_000001_rho_design.petscbin`
- `*_checkpoint_iter_000001_rho_phys.petscbin`
- `*_checkpoint_final_rho_design.petscbin`
- `*_checkpoint_final_rho_phys.petscbin`
- `*_checkpoint_final_mask.petscbin`

`*_history.csv` records per-iteration timings:

- `filter_s`
- `linear_solve_s`
- `sensitivity_s`
- `update_s`
- `checkpoint_s`
- `iter_total_s`

`*_objective_volume.csv` records a compact `iter,objective,volume,volume_target`
table.  `*_objective_volume.svg` plots objective and volume in the same figure.
Both are refreshed after every completed iteration, so early termination keeps
the last completed iteration's CSV row and plot.

`*_opt_summary.txt` records `total_wall_time_s` and peak PETSc memory.

Do not enable full VTK on the 100M run.  Convert the distributed checkpoint to
a downsampled visualization file after the optimization finishes:

```sh
sbatch --export=ALL,SOURCE_PREFIX=result/h8_100m_1step_700_400_120_8518873 \
  submit_h8_postprocess_100m.sbatch
```

Useful postprocess overrides:

```sh
sbatch --export=ALL,SOURCE_PREFIX=result/h8_100m_1step_700_400_120_8518873,POST_STRIDE=4 \
  submit_h8_postprocess_100m.sbatch
sbatch --export=ALL,SOURCE_PREFIX=result/h8_100m_1step_700_400_120_8518873,DENSITY_FILE=result/h8_100m_1step_700_400_120_8518873_checkpoint_final_rho_design.petscbin \
  submit_h8_postprocess_100m.sbatch
```

Default `POST_STRIDE=5` turns the 700x400x120 node job into roughly 140x80x24
visualization cells.  Open the generated `*_rho_phys_stride5.vtk` in ParaView
and threshold `rho_masked` or `rho`.

If you need full-resolution density plus displacement, submit the parallel VTK
export job instead:

```sh
sbatch --export=ALL,SOURCE_PREFIX=result/h8_100m_1step_700_400_120_8518873 \
  submit_h8_full_vtk_100m.sbatch
```

This mode does not sample the grid.  It reloads `*_checkpoint_final_rho_phys`,
solves one displacement field, and writes:

- `*_full_density_displacement.pvti`
- `*_full_density_displacement_pieces/*_rank000000.vti`, one piece per MPI rank

Open the `.pvti` file in ParaView.  Expect several GB of ASCII VTK output for
the 100M-DOF case, so use this only when you really need the full field.

The executable also guards VTK output with `-opt_vtk_max_points` so accidental
large VTK writes fail early instead of exhausting rank-0 memory.
