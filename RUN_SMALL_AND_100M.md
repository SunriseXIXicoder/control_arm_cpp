# Run Small Tests and 100M-DOF Jobs

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
- ranks: default sbatch uses `8 nodes * 28 tasks = 224 ranks`
- operator: H8 matrix-free
- default iterations: `OPT_MAX_ITER=5`
- density filter radius: `OPT_FILTER_RADIUS=1.5`
- no VTK output
- distributed PETSc binary checkpoints enabled

Useful overrides:

```sh
sbatch --export=ALL,OPT_MAX_ITER=10,VOLFRAC=0.35 submit_h8_opt_100m.sbatch
sbatch --nodes=2 --ntasks-per-node=28 --export=ALL,NX=350,NY=200,NZ=80,OPT_MAX_ITER=3 submit_h8_opt_100m.sbatch
sbatch --nodes=8 --ntasks-per-node=28 --export=ALL,NX=700,NY=400,NZ=120,OPT_MAX_ITER=20,VOLFRAC=0.35 submit_h8_opt_100m.sbatch
```

The H8 optimizer uses an x/y/z DMDA process grid, so higher MPI rank counts are
not limited by z slabs only.  With `OPT_FILTER_RADIUS=1.5`, each local element
block must still be at least two elements thick in x, y, and z.  The process
grid is chosen automatically and printed at startup, for example
`H8 optimizer DMDA process grid: px=... py=... pz=...`.  If needed, override it
with `H8_DM_PX/H8_DM_PY/H8_DM_PZ` in the sbatch environment, or with
`-h8_dm_px`, `-h8_dm_py`, and `-h8_dm_pz` on the executable command line.  The
product must equal the MPI rank count.

The EMsFEM/ANN optimizer also uses x/y/z fine-density decomposition now.  Its
z-direction draft closure communicates only within ranks that share the same
x-y subblock, so `NP=8` on the `40*20*8` WSL test no longer fails because of
the fine filter radius.  Manual overrides use
`EMS_DM_PX/EMS_DM_PY/EMS_DM_PZ` in the WSL script or `-ems_dm_px`,
`-ems_dm_py`, and `-ems_dm_pz` on the executable command line.

## Large-Job Outputs

For a job prefix such as `result/h8_100m_700_400_120_JOBID`, expect:

- `*_history.csv`
- `*_opt_summary.txt`
- `*_petsc.log`
- `*_checkpoint_iter_000001_rho_design.petscbin`
- `*_checkpoint_iter_000001_rho_phys.petscbin`
- `*_checkpoint_final_rho_design.petscbin`
- `*_checkpoint_final_rho_phys.petscbin`
- `*_checkpoint_final_mask.petscbin`

Do not enable VTK on the 100M run.  Use PETSc binary/HDF5-style postprocessing
or run a smaller visualization case.

The executable also guards VTK output with `-opt_vtk_max_points` so accidental
large VTK writes fail early instead of exhausting rank-0 memory.
