# Linux MATLAB and C++/PETSc Migration Route

This folder is intentionally separate from the MATLAB code and from the earlier
`hpc_petsc` benchmark.  It is the first C++/PETSc migration base for the
control-arm topology-optimization workflow.

## Current Status

Implemented now:

- C++17 PETSc executable: `bin/control_arm_cpp`
- `geometry` mode:
  - analytic control-arm mask
  - distributed PETSc `Vec` for nodal density
  - volume fraction report
  - optional PETSc binary density checkpoint
  - legacy VTK density and solid-cell VTK export
- `solve` mode:
  - distributed PETSc low-order SPD stiffness benchmark
  - control-arm mask density in the stiffness weights
  - GAMG/CG solver path
  - summary report, PETSc `-log_view`, optional VTK output
- `h8_matrix_free` solve path:
  - PETSc `MatShell`
  - DMDA distributed 3D grid with ghost nodes
  - real 8-node hexahedral linear-elasticity element from 2x2x2 Gauss integration
  - homogeneous fixed left face and right-face z-load
  - Jacobi diagonal supplied by `MATOP_GET_DIAGONAL`
- `density` mode:
  - cell-centered DMDA density field
  - distributed cone filter
  - axis-direction machining/draft projection for `+x,-x,+y,-y,+z,-z`
  - p-norm and smooth-Heaviside projection parameters matching the MATLAB
    projection style
  - `-draft_combine max|min|product`; `max` is the default for projecting one
    existing density field, while `product` is the MATLAB-style combination for
    independent directional cutting variables
  - VTK with `rho_initial`, `rho_filtered`, and `rho_projected` cell data
- `optimize` mode:
  - first distributed topology-optimization loop
  - matrix-free low-order stiffness operator and H8 matrix-free operator
  - PETSc `Vec` storage for `rho`, `dc`, and design mask
  - low-order nodal sensitivities and H8 element-energy sensitivities
  - optional H8 density filter and adjoint sensitivity filter through
    `-opt_filter_radius`
  - H8 filtered optimization constrains physical volume through `dv_design`
    rather than only constraining raw design density
  - distributed PETSc binary checkpoints for large runs
  - OC volume update
  - CSV objective history and final VTK
- WSL scripts:
  - `run_wsl_geometry.sh`
  - `run_wsl_solve.sh`
  - `run_wsl_h8_smoke.sh`
  - `run_wsl_h8_consistency.sh`
  - `run_wsl_density.sh`
  - `run_wsl_density_direction_check.sh`
  - `run_wsl_density_consistency.sh`
  - `run_wsl_optimize_smoke.sh`
  - `run_wsl_optimize_consistency.sh`
  - `run_wsl_h8_optimize_smoke.sh`
  - `run_wsl_h8_optimize_consistency.sh`
- Slurm script:
  - `submit_control_arm_cpp_scale.sbatch`
  - `submit_h8_opt_100m.sbatch`

Explicitly not implemented yet:

- ANN JSON material model
- MMA update loop
- density filter adjoint in the optimizer loop

The H8 path is now a real matrix-free finite-element operator, but it is still
a solver-kernel validation stage.  It is not yet the full topology optimizer.
For large H8 production runs, the next important step is replacing Jacobi with
a scalable geometric multigrid or auxiliary assembled preconditioner.

## WSL Usage

From Ubuntu/WSL:

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
make wsl
sh run_wsl_geometry.sh
sh run_wsl_solve.sh
sh run_wsl_h8_smoke.sh
sh run_wsl_h8_consistency.sh
sh run_wsl_density.sh
sh run_wsl_density_direction_check.sh
sh run_wsl_density_consistency.sh
sh run_wsl_optimize_smoke.sh
sh run_wsl_optimize_consistency.sh
sh run_wsl_h8_optimize_smoke.sh
sh run_wsl_h8_optimize_consistency.sh
```

Useful overrides:

```sh
NP=4 NX=80 NY=40 NZ=24 sh run_wsl_geometry.sh
NP=4 NX=80 NY=40 NZ=24 sh run_wsl_solve.sh
NP=4 NX=20 NY=10 NZ=6 sh run_wsl_h8_smoke.sh
NX=12 NY=6 NZ=4 sh run_wsl_h8_consistency.sh
NP=4 NX=40 NY=20 NZ=12 sh run_wsl_density.sh
NX=20 NY=12 NZ=12 sh run_wsl_density_consistency.sh
NP=4 NX=24 NY=12 NZ=8 ITERS=8 sh run_wsl_optimize_smoke.sh
NX=16 NY=8 NZ=6 ITERS=3 sh run_wsl_optimize_consistency.sh
NP=4 NX=14 NY=8 NZ=12 ITERS=4 sh run_wsl_h8_optimize_smoke.sh
NX=12 NY=7 NZ=8 ITERS=2 sh run_wsl_h8_optimize_consistency.sh
```

For the concise small/large run recipe, see
`RUN_SMALL_AND_100M.md`.

Important output files:

- `result/cpp_geometry_80_40_24_np4_geometry_summary.txt`
- `result/cpp_geometry_80_40_24_np4_solid.vtk`
- `result/cpp_geometry_80_40_24_np4_rho.petscbin`
- `result/cpp_solve_80_40_24_np4_solve_summary.txt`
- `result/cpp_solve_80_40_24_np4_petsc.log`
- `result/cpp_solve_80_40_24_np4_solid.vtk`
- `result/cpp_h8_20_10_6_np4_solve_summary.txt`
- `result/cpp_h8_20_10_6_np4_petsc.log`
- `result/cpp_h8_20_10_6_np4.vtk`
- `result/h8_uniform_12_6_4_np1_solve_summary.txt`
- `result/h8_uniform_12_6_4_np4_solve_summary.txt`
- `result/cpp_density_40_20_12_np4_density_summary.txt`
- `result/cpp_density_40_20_12_np4_density.vtk`
- `result/cpp_opt_24_12_8_np4_history.csv`
- `result/cpp_opt_24_12_8_np4_opt_summary.txt`
- `result/cpp_opt_24_12_8_np4_final.vtk`
- `result/opt_consistency_16_8_6_np1_opt_summary.txt`
- `result/opt_consistency_16_8_6_np4_opt_summary.txt`
- `result/cpp_h8_opt_14_8_12_np4_history.csv`
- `result/cpp_h8_opt_14_8_12_np4_opt_summary.txt`
- `result/cpp_h8_opt_14_8_12_np4_final.vtk`

For ParaView, open the `*_solid.vtk` file when checking the control-arm
outline.  The structured `*.vtk` file is mainly for scalar-field inspection and
can look like a rectangular box if the wrong scalar or threshold is selected.

The `density` mode is a kernel validation path, not a finished optimizer.  It
uses the current density field as a diagnostic source for filtering and draft
projection.  If the surrounding void is interpreted as a cutting seed in many
directions, the projected density can become much lower than the input density.
That is expected for this diagnostic mode; the production optimizer stage must
introduce independent distributed design variables for directional cuts or MMA
density updates.

Density mode currently uses a z-slab decomposition so every rank owns a full
`x-y` slab range.  Use enough `nz` layers for the chosen filter/draft radius;
otherwise reduce `NP` or the radius.

## Slurm Usage

Small geometry check:

```sh
sbatch --export=ALL,MODE=geometry,NX=160,NY=80,NZ=48,WRITE_SOLID_VTK=true \
  submit_control_arm_cpp_scale.sbatch
```

Low-order distributed solve:

```sh
sbatch --nodes=4 --ntasks-per-node=28 \
  --export=ALL,MODE=solve,NX=700,NY=400,NZ=120 \
  submit_control_arm_cpp_scale.sbatch
```

H8 matrix-free smoke or medium run:

```sh
sbatch --nodes=1 --ntasks-per-node=28 \
  --export=ALL,MODE=solve,OPERATOR=h8_matrix_free,NX=80,NY=40,NZ=24 \
  submit_control_arm_cpp_scale.sbatch
```

When using `OPERATOR=h8_matrix_free`, keep `-pc_type jacobi` or provide a
matrix-free-compatible preconditioner.  Plain `PCGAMG` needs an assembled graph
and is not the correct default for the current `MatShell`.

For large jobs, keep `WRITE_STRUCTURED_VTK=false` and `WRITE_SOLID_VTK=false`.
Use PETSc binary or future PVTU/HDF5 checkpoints instead of root-rank legacy VTK.

## Why C++/PETSc Instead of Linux MATLAB for 100M DOF

Linux MATLAB is useful for small reference runs and postprocessing, but it is
not the production route for 100M+ DOF.  The memory bottleneck in the MATLAB
workflow comes from global sparse assembly, index arrays, material tensors,
global density/sensitivity arrays, and serial I/O patterns.  PETSc/MPI keeps
vectors and matrices distributed by ownership ranges, which is the path needed
for multi-node production.

Install Linux MATLAB only if you need small-script parity checks:

```sh
unzip matlab_R20XXx_Linux.zip -d matlab_R20XXx_Linux
cd matlab_R20XXx_Linux
sudo ./install
export PATH=/usr/local/MATLAB/R20XXx/bin:$PATH
matlab -batch "disp(version)"
```

For headless installation, use MathWorks silent install with an input file and
license information.  On a cluster, prefer the public MATLAB module if it
already exists.

## Migration Stages

1. MATLAB reference baselines
   - Run `20_10_4` and `40_20_8`.
   - Save `opt_data.mat`, VTK, objective history, and final density.

2. C++ geometry and density
   - This folder now implements the first version.
   - Compare `*_solid.vtk` and volume fraction against MATLAB masks.
   - Target mask volume-fraction difference: `<1e-4` after the exact MATLAB
     mask is fully translated.

3. C++ H8 matrix-free FEM
   - Implemented first PETSc `MatShell` over H8 cells.
   - Avoids explicit global H8 sparse matrix and MATLAB-style `iK/jK/sK`.
   - Compare small-grid displacement/compliance with MATLAB.
   - Target relative compliance error: `<1e-5`.

4. Distributed density, filter, draft projection, and sensitivities
   - `rho` and `rho_phys` now have an initial PETSc/DMDA distributed path.
   - Cone filtering and six-axis draft projection are implemented.
   - Distributed H8 `dc/dv` and filter adjoint are implemented in the optimizer
     path.
   - Use MPI reductions for volume and compliance.
   - Write checkpoints in PETSc binary or HDF5/XDMF.

5. C++ optimizer
   - First OC loop is implemented for the low-order matrix-free validation
     operator.
   - H8 matrix-free OC loop with element-density sensitivities is implemented.
   - H8 optimization now supports density filtering with adjoint sensitivity
     filtering and physical-volume constrained OC updates.
   - Next: port MMA or replace with a tested C++ MMA implementation.
   - Next: connect distributed filter adjoints and smooth/draft projection to
     the optimizer loop.
   - Compare 20-step trends with MATLAB, not exact floating-point traces.

6. Scale ladder
   - Run `1M -> 5M -> 10M -> 100M DOF`.
   - Save `*_summary.txt`, `*_petsc.log`, KSP iteration counts, and peak memory.

## Acceptance Checks

- No MATLAB global sparse on the production path.
- `rho/dc/dv` not stored globally by one rank.
- Peak memory per rank decreases when rank count increases.
- Same grid on 1/2/4/8 ranks gives compliance differences below `1e-6` for the
  low-order validation operator and the H8 matrix-free smoke cases.
- 100M DOF runs produce summary, PETSc log, and checkpoint without root-rank
  full-field VTK.
