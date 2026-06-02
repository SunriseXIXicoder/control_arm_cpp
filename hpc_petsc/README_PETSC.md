# PETSc/MPI 100M-DOF Control-Arm-Scale Kernel

This folder is intentionally separate from the original MATLAB code and from
`hpc_parallel`.  It provides a PETSc/MPI distributed stiffness solve that can be
submitted to a cluster as a 100M+ unknown benchmark.

For the full memory-bottleneck strategy and migration rationale, read
`MEMORY_BOTTLENECK_SOLUTION.md`.

## What is included

- `src/control_arm_petsc.c`
  - PETSc C program.
  - Assembles a distributed 3-D SPD stiffness matrix.
  - Uses 3 displacement DOFs per node.
  - Applies fixed constraints on the left face and a right-face load.
  - Default size: `nx=700, ny=400, nz=120`, i.e. `100,800,000` DOFs.
- `Makefile`
  - Builds the PETSc executable.
- `submit_petsc_debug.sbatch`
  - Small cluster smoke test.
- `submit_petsc_scale.sbatch`
  - Parameterized cluster scale test. Submit with `NX/NY/NZ` overrides.
- `submit_petsc_100m.sbatch`
  - 100M-DOF submission template.
- `run_wsl_smoke.sh` and `run_wsl_memory_ladder.sh`
  - Local WSL validation scripts using `pkg-config PETSc`.
  - `run_wsl_smoke.sh` writes a small VTK file by default.
- `run_wsl_control_arm_mask.sh`
  - Local diagnostic run for the analytic control-arm mask. It writes both a
    structured VTK and a thresholded solid-cell VTK.
- `.wslconfig.example`
  - Suggested WSL memory/swap settings for local validation.
- `result/`
  - Output directory for Slurm stdout/stderr and PETSc logs.

## Build

Edit the PETSc paths in the sbatch files, or export them before running:

```sh
export PETSC_DIR=/data/app/petsc
export PETSC_ARCH=
make -C hpc_petsc all
```

On WSL Ubuntu with PETSc installed through system packages:

```sh
cd hpc_petsc
make wsl
sh run_wsl_smoke.sh
sh run_wsl_control_arm_mask.sh
sh run_wsl_memory_ladder.sh
```

The smoke run writes:

```text
result/wsl_smoke_80_40_24_np4.vtk
```

Open it in ParaView and color by `displacement_magnitude` or `density`.

For the control-arm mask diagnostic, open:

```text
result/control_arm_mask_80_40_24_np4_solid.vtk
```

This file already contains only solid mask cells, so it does not need a
Threshold filter.

For a PETSc source-tree build, `PETSC_ARCH` may be required, for example:

```sh
export PETSC_DIR=/data/app/petsc/petsc-3.20.5
export PETSC_ARCH=arch-linux-c-opt
```

## Submit

Run the debug job first:

```sh
cd hpc_petsc
sbatch submit_petsc_debug.sbatch
```

If the debug job builds and converges, submit the 100M-DOF job:

```sh
sbatch submit_petsc_100m.sbatch
```

Or submit a parameterized scale job:

```sh
sbatch --nodes=4 --ntasks-per-node=28 \
  --export=ALL,NX=700,NY=400,NZ=120 \
  submit_petsc_scale.sbatch
```

The default 100M case uses:

```text
nx = 700
ny = 400
nz = 120
DOF = 3 * nx * ny * nz = 100,800,000
MPI ranks = 4 nodes * 28 ranks/node = 112
```

## PETSc options

The sbatch uses:

```text
-ksp_type cg
-pc_type gamg
-pc_gamg_type agg
-mg_levels_ksp_type chebyshev
-mg_levels_pc_type jacobi
-ksp_rtol 1e-6
```

For difficult cases, useful alternatives are:

```text
-ksp_type fgmres -pc_type gamg
-pc_gamg_threshold 0.02
-pc_gamg_square_graph 1
-ksp_rtol 1e-5
```

## Important scope note

This is a distributed PETSc solve kernel, not yet the full MATLAB topology
optimization loop.  The current MATLAB optimizer still stores density fields,
sensitivities, filters, and history arrays in MATLAB memory.  A true 100M-DOF
optimization run needs the density update and sensitivity/filter steps to be
distributed too, or it needs a file/MPI coupling layer between MATLAB and PETSc.

The recommended path is:

1. Use this folder to validate PETSc, Slurm, memory, and solver convergence.
2. Replace this benchmark matrix assembly with the exact H8 element assembly.
3. Move density, sensitivity filtering, and OC/MMA update into distributed PETSc
   vectors, or couple MATLAB to PETSc only for much smaller design-variable
   sets.
4. Only then scale the full optimization loop to 100M+ DOFs.

## Optional control-arm-like mask

The executable has an analytic coarse mask option:

```sh
./bin/control_arm_petsc -nx 700 -ny 400 -nz 120 -control_arm_mask true
```

The default is a uniform material benchmark because it is the most stable way to
test cluster scale and solver performance first.

## Current CLI status

Implemented now:

```text
-operator low_order
-nx -ny -nz
-output_prefix <path>
-write_solution true|false
-write_vtk true|false
-vtk_file <path>
-vtk_max_points <n>
-write_mask_vtk true|false
-mask_vtk_file <path>
-mask_threshold <rho>
-mask_vtk_max_cells <n>
-control_arm_mask true|false
```

`-write_vtk` writes a legacy ASCII `STRUCTURED_POINTS` file on rank 0. It is
for small diagnostic runs only; the default `-vtk_max_points` limit is
`5000000` grid points to avoid accidentally writing a huge single file.

Reserved for the next migration stages:

```text
-operator h8_matrix_free
-density_file <path>
```

Those reserved options intentionally return a PETSc unsupported-operation error
until the real H8 matrix-free operator and distributed density vectors are added.
