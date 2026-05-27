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
- WSL scripts:
  - `run_wsl_geometry.sh`
  - `run_wsl_solve.sh`
- Slurm script:
  - `submit_control_arm_cpp_scale.sbatch`

Explicitly not implemented yet:

- true H8/EMsFEM `MatShell` operator
- ANN JSON material model
- draft-direction projection and smoothness constraint in distributed form
- MMA update loop
- distributed filter and sensitivities

The program refuses `-operator h8_matrix_free` for now so that engineering
results are not confused with the low-order validation kernel.

## WSL Usage

From Ubuntu/WSL:

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
make wsl
sh run_wsl_geometry.sh
sh run_wsl_solve.sh
```

Useful overrides:

```sh
NP=4 NX=80 NY=40 NZ=24 sh run_wsl_geometry.sh
NP=4 NX=80 NY=40 NZ=24 sh run_wsl_solve.sh
```

Important output files:

- `result/cpp_geometry_80_40_24_np4_geometry_summary.txt`
- `result/cpp_geometry_80_40_24_np4_solid.vtk`
- `result/cpp_geometry_80_40_24_np4_rho.petscbin`
- `result/cpp_solve_80_40_24_np4_solve_summary.txt`
- `result/cpp_solve_80_40_24_np4_petsc.log`
- `result/cpp_solve_80_40_24_np4_solid.vtk`

For ParaView, open the `*_solid.vtk` file when checking the control-arm
outline.  The structured `*.vtk` file is mainly for scalar-field inspection and
can look like a rectangular box if the wrong scalar or threshold is selected.

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
   - Implement PETSc `MatShell` over H8 cells.
   - Avoid explicit global H8 sparse matrix and MATLAB-style `iK/jK/sK`.
   - Compare small-grid displacement/compliance with MATLAB.
   - Target relative compliance error: `<1e-5`.

4. Distributed density, filter, draft projection, and sensitivities
   - Store `rho`, `rho_phys`, `dc`, and `dv` as PETSc `Vec`.
   - Use MPI reductions for volume and compliance.
   - Write checkpoints in PETSc binary or HDF5/XDMF.

5. C++ optimizer
   - Port MMA or replace with a tested C++ MMA implementation.
   - Compare 20-step trends with MATLAB, not exact floating-point traces.

6. Scale ladder
   - Run `1M -> 5M -> 10M -> 100M DOF`.
   - Save `*_summary.txt`, `*_petsc.log`, KSP iteration counts, and peak memory.

## Acceptance Checks

- No MATLAB global sparse on the production path.
- `rho/dc/dv` not stored globally by one rank.
- Peak memory per rank decreases when rank count increases.
- Same grid on 1/2/4/8 ranks gives compliance differences below `1e-6` for the
  low-order validation operator.
- 100M DOF runs produce summary, PETSc log, and checkpoint without root-rank
  full-field VTK.
