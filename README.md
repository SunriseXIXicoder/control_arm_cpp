# Control Arm C++/PETSc Driver

This directory contains the C++/PETSc migration driver for distributed control-arm topology optimization.  The executable exposes PETSc command-line interfaces for geometry checks, density preprocessing, linear solves, H8 and EMsFEM ANN optimization, and checkpoint postprocessing.

## Build

Use the same compiler/MPI module stack for PETSc, compilation, and runtime.

```sh
export PETSC_DIR=/data/home/dlut_ycx/petsc/petsc-v3.19.3
export PETSC_ARCH=arch-linux-c-opt-hypre
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$PETSC_DIR/lib:${LD_LIBRARY_PATH:-}"

make -B all CXXSTD=c++11 PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH"
```

`PETSC_ARCH=arch-linux-c-opt-hypre` is recommended for production because H8 and EMsFEM ANN runs can use HYPRE BoomerAMG. PETSc GAMG is built into PETSc; HYPRE requires PETSc to be configured with `--download-hypre` or an equivalent external HYPRE installation.

## Command Format

The executable uses PETSc option syntax:

```sh
mpirun -np <ranks> ./bin/control_arm_cpp \
  -mode <interface> \
  -operator <operator> \
  -nx <int> -ny <int> -nz <int> \
  [interface options] [PETSc KSP/PC options]
```

Rules:

- Options are passed as `-key value`; booleans use `true` or `false`.
- Paths are ordinary relative or absolute paths.
- PETSc solver options such as `-ksp_type`, `-pc_type`, `-pc_gamg_*`, `-pc_hypre_*`, `-ksp_monitor_true_residual`, and `-log_view ":file.log"` can be appended to any solve or optimization command.
- `-output_prefix <path>` controls report, history, checkpoint, and default output names.

## Interfaces

| Interface | Required mode | Operator | Purpose | Main outputs |
| --- | --- | --- | --- | --- |
| Geometry inspection | `-mode geometry` | any, usually `low_order` | Builds the control-arm mask and density field, reports mask volume, and optionally writes VTK/PETSc binary density. | `<prefix>_summary.txt`, optional `.vtk`, optional `_rho.petscbin` |
| Density pipeline | `-mode density` | any | Applies density filtering and optional draft-direction aggregation before optimization diagnostics. | `<prefix>_density.vtk`, optional `_rho.petscbin` |
| Linear solve | `-mode solve` | `low_order`, `h8_matrix_free`, or `emsfem_ann` | Builds one elasticity system and solves `K u = f` for solver and boundary-condition validation. | `<prefix>_summary.txt`, optional `_u.petscbin`, optional `.vtk` |
| Optimization | `-mode optimize` | `low_order`, `h8_matrix_free`, or `emsfem_ann` | Runs topology optimization with filtering, projection, checkpoints, and selected preconditioner. | history CSV, objective-volume CSV/SVG, summary, checkpoints, optional final VTK |
| EMsFEM ANN postsolve | `-mode ems_ann_postsolve` | `emsfem_ann` | Reloads an EMsFEM ANN density checkpoint and solves displacement fields without rerunning optimization. | postsolve summary and displacement checkpoints |
| H8 initial VTK | `-mode h8_initial_vtk` | `h8_matrix_free` | Writes the initial filtered/masked H8 density for inspection. | downsampled `.vtk` |
| H8 density postprocess | `-mode h8_postprocess` | `h8_matrix_free` | Converts an H8 density checkpoint into downsampled VTK. | downsampled `.vtk` |
| H8 full VTK postprocess | `-mode h8_full_vtk` | `h8_matrix_free` | Reloads H8 density, solves displacement, and writes full-resolution parallel VTK. | `.pvti` plus rank-local `.vti` pieces |

## Operators

| Operator | Use case |
| --- | --- |
| `low_order` | Lightweight distributed PETSc baseline for geometry, solver, and optimizer smoke tests. |
| `h8_matrix_free` | H8 matrix-free elasticity and topology optimization. This is the main H8 production path. |
| `emsfem_ann` | EMsFEM ANN matrix-free operator with fine-density subcells and ANN auxiliary preconditioning. |

## Common Options

| Option | Meaning | Default |
| --- | --- | --- |
| `-nx`, `-ny`, `-nz` | Nodal grid dimensions. DOF is `3*nx*ny*nz`. | `80 40 24` |
| `-domain_height` | Physical height in meters; length and width are derived by project geometry helpers. | `0.08` |
| `-control_arm_mask` | Use the control-arm solid/void mask. | `true` |
| `-benchmark_case` | Rectangular no-mask benchmark load when `-control_arm_mask false`: `cantilever` or `torsion`. | `cantilever` |
| `-void_density` | Density assigned to void/non-design regions. | `0.02` |
| `-young_modulus` | Elastic modulus. | `2.1e11` |
| `-penal` | SIMP penalty. | `3.0` |
| `-emin` | Minimum stiffness scale. | `1e-6` |
| `-mask_threshold` | Threshold used when writing solid-mask VTK. | `0.5` |
| `-ab_triangle_retract` | Retraction ratio for the A/B triangle region in the control-arm mask. | `0.22` |
| `-output_prefix` | Base path for generated files. | `result/control_arm_cpp` |

## Geometry Interface

Format:

```sh
./bin/control_arm_cpp \
  -mode geometry \
  -nx <int> -ny <int> -nz <int> \
  -output_prefix <prefix> \
  [-write_structured_vtk true -structured_vtk_file <file.vtk>] \
  [-write_solid_vtk true -solid_vtk_file <file.vtk>] \
  [-write_density_binary true -density_binary_file <file.petscbin>]
```

Example:

```sh
mpirun -np 4 ./bin/control_arm_cpp \
  -mode geometry \
  -nx 80 -ny 40 -nz 24 \
  -output_prefix result/geom_80_40_24 \
  -write_solid_vtk true \
  -write_structured_vtk true
```

## Density Pipeline Interface

Format:

```sh
./bin/control_arm_cpp \
  -mode density \
  -nx <int> -ny <int> -nz <int> \
  -filter_radius <real> \
  -draft_radius <int> \
  -draft_pnorm <real> \
  -draft_beta <real> \
  -draft_eta <real> \
  -draft_dirs <+x,-x,+y,-y,+z,-z list> \
  -draft_combine <max|...> \
  [-write_density_vtk true -density_vtk_file <file.vtk>] \
  [-write_density_binary true -density_binary_file <file.petscbin>]
```

Example:

```sh
mpirun -np 4 ./bin/control_arm_cpp \
  -mode density \
  -nx 80 -ny 40 -nz 24 \
  -filter_radius 1.5 \
  -draft_radius 1 \
  -draft_dirs +z \
  -write_density_vtk true \
  -write_density_binary true \
  -output_prefix result/density_80_40_24
```

## Paper Verification And Benchmark Cases

### Layer 1: Draft-Constraint Verification

Use `-mode density` with `-control_arm_mask false` to generate regular rectangular-domain density fields. The VTK contains `rho_initial`, `rho_filtered`, and `rho_projected`, so ParaView can show the three columns directly.

No draft constraint:

```sh
mpirun -np 4 ./bin/control_arm_cpp \
  -mode density \
  -control_arm_mask false \
  -nx 121 -ny 41 -nz 31 \
  -filter_radius 1.5 \
  -draft_radius 0 \
  -draft_dirs +z \
  -write_density_vtk true \
  -density_vtk_file result/layer1_nodraft.vtk \
  -write_density_binary true \
  -output_prefix result/layer1_nodraft
```

Single-draw `+z`, single-draw `-z`, and split-draw `+z/-z`:

```sh
mpirun -np 4 ./bin/control_arm_cpp -mode density -control_arm_mask false \
  -nx 121 -ny 41 -nz 31 -filter_radius 1.5 \
  -draft_radius 6 -draft_pnorm 8 -draft_beta 8 -draft_eta 0.5 \
  -draft_dirs +z -draft_combine max \
  -write_density_vtk true -density_vtk_file result/layer1_plus_z.vtk \
  -write_density_binary true -output_prefix result/layer1_plus_z

mpirun -np 4 ./bin/control_arm_cpp -mode density -control_arm_mask false \
  -nx 121 -ny 41 -nz 31 -filter_radius 1.5 \
  -draft_radius 6 -draft_pnorm 8 -draft_beta 8 -draft_eta 0.5 \
  -draft_dirs -z -draft_combine max \
  -write_density_vtk true -density_vtk_file result/layer1_minus_z.vtk \
  -write_density_binary true -output_prefix result/layer1_minus_z

mpirun -np 4 ./bin/control_arm_cpp -mode density -control_arm_mask false \
  -nx 121 -ny 41 -nz 31 -filter_radius 1.5 \
  -draft_radius 6 -draft_pnorm 8 -draft_beta 8 -draft_eta 0.5 \
  -draft_dirs +z,-z -draft_combine max \
  -write_density_vtk true -density_vtk_file result/layer1_split_z.vtk \
  -write_density_binary true -output_prefix result/layer1_split_z
```

### Layer 2: Classical Rectangular Benchmarks

Set `-control_arm_mask false` to use a regular design box. The benchmark interface fixes the left end face and supports `-benchmark_case cantilever` for a tip bending load or `-benchmark_case torsion` for a free-end torque about the x-axis. The commands below use the full model; strict half/quarter symmetry needs additional component-wise symmetry-plane constraints.

H8 upper-bound baseline without draft:

```sh
mpirun -np 8 ./bin/control_arm_cpp \
  -mode optimize \
  -operator h8_matrix_free \
  -control_arm_mask false \
  -benchmark_case cantilever \
  -nx 121 -ny 41 -nz 31 \
  -volfrac 0.30 \
  -opt_max_iter 200 \
  -opt_filter_radius 1.5 \
  -opt_z_draft_closure false \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval 50 \
  -h8_pc_type aux_hypre \
  -ksp_type fgmres \
  -ksp_gmres_restart 200 \
  -output_prefix result/layer2_h8_cantilever_nodraft
```

H8 `+Z` draft, H8 `+Z` draft with Heaviside, and EMsFEM ANN `+Z` draft:

```sh
mpirun -np 8 ./bin/control_arm_cpp -mode optimize -operator h8_matrix_free \
  -control_arm_mask false -benchmark_case cantilever \
  -nx 121 -ny 41 -nz 31 -volfrac 0.30 -opt_max_iter 200 \
  -opt_filter_radius 1.5 -opt_z_draft_closure true -opt_z_draft_eta 0.5 \
  -opt_write_checkpoint true -opt_checkpoint_interval 50 \
  -h8_pc_type aux_hypre -ksp_type fgmres -ksp_gmres_restart 200 \
  -output_prefix result/layer2_h8_cantilever_zdraft

mpirun -np 8 ./bin/control_arm_cpp -mode optimize -operator h8_matrix_free \
  -control_arm_mask false -benchmark_case cantilever \
  -nx 121 -ny 41 -nz 31 -volfrac 0.30 -opt_max_iter 200 \
  -opt_filter_radius 1.5 -opt_z_draft_closure true -opt_z_draft_eta 0.5 \
  -opt_heaviside_projection true -opt_heaviside_eta 0.5 \
  -opt_heaviside_beta_initial 1 -opt_heaviside_beta_max 16 \
  -opt_heaviside_beta_interval 50 \
  -opt_write_checkpoint true -opt_checkpoint_interval 50 \
  -h8_pc_type aux_hypre -ksp_type fgmres -ksp_gmres_restart 200 \
  -output_prefix result/layer2_h8_cantilever_zdraft_heaviside

mpirun -np 8 ./bin/control_arm_cpp -mode optimize -operator emsfem_ann \
  -control_arm_mask false -control_arm_bc false -benchmark_case cantilever \
  -nx 121 -ny 41 -nz 31 -ems_sub_n 5 -ems_ann_dir ../input_5 \
  -volfrac 0.30 -opt_max_iter 200 -opt_filter_radius 1.5 \
  -opt_z_draft_closure true -opt_z_draft_eta 0.5 \
  -opt_write_checkpoint true -opt_checkpoint_interval 50 \
  -ems_pc_type aux_ann_hypre -ksp_type fgmres -ksp_gmres_restart 200 \
  -output_prefix result/layer2_ems_ann_cantilever_zdraft
```

For the torsion beam, keep the same options and change only the benchmark case and output prefix:

```sh
  -benchmark_case torsion \
  -output_prefix result/layer2_h8_torsion_zdraft
```

## Solve Interface

Format:

```sh
./bin/control_arm_cpp \
  -mode solve \
  -operator <low_order|h8_matrix_free|emsfem_ann> \
  -nx <int> -ny <int> -nz <int> \
  -load <real> \
  [-write_solution true -solution_file <file.petscbin>] \
  [-write_structured_vtk true -structured_vtk_file <file.vtk>] \
  [-write_solid_vtk true -solid_vtk_file <file.vtk>] \
  [PETSc KSP/PC options]
```

Example:

```sh
mpirun -np 4 ./bin/control_arm_cpp \
  -mode solve \
  -operator h8_matrix_free \
  -nx 80 -ny 40 -nz 24 \
  -load 1.0 \
  -ksp_type cg \
  -pc_type jacobi \
  -ksp_monitor_true_residual \
  -write_solution true \
  -write_structured_vtk true \
  -output_prefix result/h8_solve_80_40_24
```

## Optimization Interface

Format:

```sh
./bin/control_arm_cpp \
  -mode optimize \
  -operator <low_order|h8_matrix_free|emsfem_ann> \
  -nx <int> -ny <int> -nz <int> \
  -volfrac <real> \
  -opt_max_iter <int> \
  -opt_filter_radius <real> \
  -opt_move <real> \
  -opt_ksp_rtol <real> \
  -opt_ksp_max_it <int> \
  [-control_arm_mask false -benchmark_case <cantilever|torsion>] \
  [-opt_heaviside_projection true -opt_heaviside_eta <real> \
   -opt_heaviside_beta_initial <real> -opt_heaviside_beta_max <real> \
   -opt_heaviside_beta_interval <int>] \
  [-opt_z_draft_closure true -opt_z_draft_eta <real>] \
  [-opt_write_checkpoint true -opt_checkpoint_interval <int> \
   -opt_checkpoint_prefix <prefix>] \
  [-opt_write_final_vtk true -opt_vtk_file <file.vtk>] \
  [operator-specific options] [PETSc KSP/PC options]
```

Important optimization options:

| Option | Meaning |
| --- | --- |
| `-opt_move`, `-opt_move_min`, `-opt_move_shrink`, `-opt_move_growth` | Move-limit control. |
| `-opt_max_compliance_increase` | Stability guard threshold for rejecting unsafe updates. |
| `-opt_projected_volume_correction` | Correct volume after projection. |
| `-opt_rho_min` | Lower bound on design density. |
| `-opt_write_checkpoint` | Write PETSc binary density/mask checkpoints. |
| `-opt_stop_on_ksp_divergence` | Stop or skip unsafe optimization updates when KSP diverges. |
| `-benchmark_case` | `cantilever` or `torsion` when `-control_arm_mask false`. |

### H8 Options

| Option | Values | Meaning |
| --- | --- | --- |
| `-h8_pc_type` | `block_jacobi`, `jacobi`, `petsc`, `aux_gamg`, `aux_hypre`, `aux_elastic_gamg`, `aux_elastic_hypre` | Select the H8 preconditioner. Production H8 runs normally use `aux_hypre` or `aux_elastic_hypre`. |
| `-h8_dm_px`, `-h8_dm_py`, `-h8_dm_pz` | positive integers | Manually set the H8 3D process grid; product must equal MPI ranks. |
| `-h8_load_case` | integer | Select H8 control-arm load case. |
| `-h8_include_spring_load` | `true|false` | Include spring/bushing load contribution. |
| `-benchmark_case` | `cantilever`, `torsion` | Select the no-mask rectangular benchmark load. |
| `-h8_aux_rebuild_interval` | integer | Rebuild auxiliary matrix every N iterations. |
| `-h8_reuse_initial_guess` | `true|false` | Reuse previous displacement as KSP initial guess. |

Direct H8 example:

```sh
mpirun -np 28 ./bin/control_arm_cpp \
  -mode optimize \
  -operator h8_matrix_free \
  -nx 325 -ny 186 -nz 56 \
  -volfrac 0.30 \
  -opt_max_iter 5 \
  -opt_filter_radius 1.5 \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval 1 \
  -h8_pc_type aux_hypre \
  -ksp_type fgmres \
  -ksp_gmres_restart 200 \
  -pc_hypre_type boomeramg \
  -pc_hypre_boomeramg_strong_threshold 0.5 \
  -output_prefix result/h8_10m_direct
```

### EMsFEM ANN Options

| Option | Values | Meaning |
| --- | --- | --- |
| `-ems_ann_dir` | path | ANN model directory, for example `../input_5` or `../input_20`. |
| `-ems_sub_n` | integer >= 2 | Fine-density subcell count per coarse element direction. |
| `-ems_cache_element_matrices` | `true|false` | Cache EMsFEM element matrices; optimization requires `true`. |
| `-ems_cache_gib_limit` | real | Per-rank cache limit in GiB; `0` means unlimited. |
| `-control_arm_bc` | `true|false` | Use control-arm boundary conditions instead of simple test BC. |
| `-benchmark_case` | `cantilever`, `torsion` | Select the no-mask rectangular benchmark load when `-control_arm_bc false`. |
| `-load_case` | integer | `0` means weighted multi-case load set in the production scripts. |
| `-include_spring_load` | `true|false` | Include spring/bushing load contribution. |
| `-ems_pc_type` | `block_jacobi`, `jacobi`, `petsc`, `aux_gamg`, `aux_hypre`, `aux_elastic_gamg`, `aux_elastic_hypre`, `aux_ann_gamg`, `aux_ann_hypre` | Select the EMsFEM ANN preconditioner. Production default is `aux_ann_gamg`; HYPRE comparison uses `aux_ann_hypre`. |
| `-ems_dm_px`, `-ems_dm_py`, `-ems_dm_pz` | positive integers | Manually set the EMsFEM 3D process grid; product must equal MPI ranks. |

Direct EMsFEM ANN example:

```sh
mpirun -np 56 ./bin/control_arm_cpp \
  -mode optimize \
  -operator emsfem_ann \
  -nx 325 -ny 186 -nz 56 \
  -ems_sub_n 5 \
  -ems_ann_dir ../input_5 \
  -control_arm_bc true \
  -load_case 0 \
  -include_spring_load true \
  -volfrac 0.30 \
  -opt_max_iter 10 \
  -opt_filter_radius 1.5 \
  -ems_pc_type aux_ann_gamg \
  -pc_gamg_type agg \
  -pc_gamg_threshold 0.02 \
  -ksp_type fgmres \
  -ksp_gmres_restart 200 \
  -output_prefix result/ems_ann_10m_direct
```

## Postprocessing Interfaces

H8 downsampled density VTK:

```sh
mpirun -np 28 ./bin/control_arm_cpp \
  -mode h8_postprocess \
  -operator h8_matrix_free \
  -nx 700 -ny 400 -nz 120 \
  -density_file result/run_checkpoint_final_rho_phys.petscbin \
  -mask_file result/run_checkpoint_final_mask.petscbin \
  -post_vtk_file result/run_rho_phys_stride5.vtk \
  -post_stride 5 \
  -post_max_samples 2000000 \
  -output_prefix result/run_post
```

H8 full density plus displacement VTK:

```sh
mpirun -np 224 ./bin/control_arm_cpp \
  -mode h8_full_vtk \
  -operator h8_matrix_free \
  -nx 700 -ny 400 -nz 120 \
  -density_file result/run_checkpoint_final_rho_phys.petscbin \
  -mask_file result/run_checkpoint_final_mask.petscbin \
  -post_vtk_file result/run_full_density_displacement.pvti \
  -h8_pc_type block_jacobi \
  -output_prefix result/run_full_vtk
```

EMsFEM ANN postsolve:

```sh
mpirun -np 56 ./bin/control_arm_cpp \
  -mode ems_ann_postsolve \
  -operator emsfem_ann \
  -nx 325 -ny 186 -nz 56 \
  -ems_sub_n 5 \
  -ems_ann_dir ../input_5 \
  -density_file result/run_checkpoint_final_rho_phys.petscbin \
  -mask_file result/run_checkpoint_final_mask.petscbin \
  -ems_pc_type aux_ann_gamg \
  -output_prefix result/run_postsolve
```

## Slurm Wrapper Interfaces

The sbatch files wrap the command-line interfaces above and expose settings through environment variables:

```sh
sbatch --export=ALL,KEY=value,KEY=value script.sbatch
```

| Script | Purpose | Useful exports |
| --- | --- | --- |
| `submit_h8_preflight.sbatch` | Validate PETSc paths, shell syntax, compiler, MPI launcher, and executable startup. | `PETSC_DIR`, `PETSC_ARCH`, `MPI_LAUNCHER` |
| `submit_mpi_launcher_check.sbatch` | Check `srun`/`mpirun` behavior across allocated nodes. | `MPI_LAUNCHER`, `MPI_LAUNCHER_ARGS` |
| `submit_h8_opt_10m.sbatch` | H8 10M-DOF diagnostic optimization. | `NX`, `NY`, `NZ`, `OPT_MAX_ITER`, `H8_PC_TYPE`, `PETSC_EXTRA_OPTIONS` |
| `submit_h8_opt_100m.sbatch` | H8 100M-DOF production optimization. | `NX`, `NY`, `NZ`, `OPT_MAX_ITER`, `H8_PC_TYPE`, `H8_DM_PX/PY/PZ`, `RUN_LABEL` |
| `submit_h8_opt_100m_1step.sbatch` | One-step 100M-DOF H8 smoke run. | same as H8 production |
| `submit_ems_ann_scale.sbatch` | EMsFEM ANN optimization. | `NELX/NELY/NELZ` or `NX/NY/NZ`, `EMS_SUB_N`, `ANN_DIR`, `EMS_PC_TYPE`, `GAMG_PROFILE`, `HYPRE_PROFILE`, `ALLOW_HUGE_EMS` |
| `submit_h8_postprocess_100m.sbatch` | Convert H8 checkpoint to downsampled VTK. | `SOURCE_PREFIX`, `DENSITY_FILE`, `MASK_FILE`, `POST_STRIDE`, `VTK_FILE` |
| `submit_h8_full_vtk_100m.sbatch` | Write full-resolution H8 `.pvti` and `.vti` pieces. | `SOURCE_PREFIX`, `DENSITY_FILE`, `MASK_FILE`, `VTK_FILE`, `H8_PC_TYPE` |
| `submit_ems_ann_full_vtk.sbatch` | Convert EMsFEM ANN fine-density checkpoint to full VTK through the Python converter. | `CHECKPOINT_PREFIX`, `DENSITY_FILE`, `MASK_FILE`, `VTK_FILE`, `EMS_SUB_N` |

Example preflight on the new cluster home:

```sh
sbatch --export=ALL,PETSC_DIR=/data/home/dlut_ycx/petsc/petsc-v3.19.3,PETSC_ARCH=arch-linux-c-opt-hypre \
  submit_h8_preflight.sbatch
```

Example H8 production submission:

```sh
sbatch --nodes=8 --ntasks-per-node=28 \
  --export=ALL,PETSC_DIR=/data/home/dlut_ycx/petsc/petsc-v3.19.3,PETSC_ARCH=arch-linux-c-opt-hypre,H8_PC_TYPE=aux_hypre,RUN_LABEL=h8_100m \
  submit_h8_opt_100m.sbatch
```

Example EMsFEM ANN submission:

```sh
sbatch --nodes=48 --ntasks-per-node=28 \
  --export=ALL,PETSC_DIR=/data/home/dlut_ycx/petsc/petsc-v3.19.3,PETSC_ARCH=arch-linux-c-opt-hypre,EMS_SUB_N=5,ANN_DIR=../input_5,EMS_PC_TYPE=aux_ann_gamg,GAMG_PROFILE=strong \
  submit_ems_ann_scale.sbatch
```

---

# 控制臂 C++/PETSc 驱动中文说明

本目录是控制臂拓扑优化从 MATLAB 迁移到 C++/PETSc 的并行驱动。程序通过 PETSc 命令行选项开放几类接口：几何检查、密度预处理、线性求解、H8 与 EMsFEM ANN 拓扑优化，以及 checkpoint 后处理。

## 编译

编译 PETSc、编译本程序和提交作业时，应使用同一套编译器/MPI 环境。

```sh
export PETSC_DIR=/data/home/dlut_ycx/petsc/petsc-v3.19.3
export PETSC_ARCH=arch-linux-c-opt-hypre
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$PETSC_DIR/lib:${LD_LIBRARY_PATH:-}"

make -B all CXXSTD=c++11 PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH"
```

生产运行建议使用 `PETSC_ARCH=arch-linux-c-opt-hypre`。PETSc GAMG 是 PETSc 自带预条件器；HYPRE BoomerAMG 需要 PETSc 配置时带 `--download-hypre`，或者链接已有的 HYPRE。

## 命令格式

可执行文件使用 PETSc 选项格式：

```sh
mpirun -np <进程数> ./bin/control_arm_cpp \
  -mode <接口名称> \
  -operator <算子名称> \
  -nx <整数> -ny <整数> -nz <整数> \
  [接口参数] [PETSc KSP/PC 参数]
```

使用规则：

- 参数写成 `-key value`；布尔值写 `true` 或 `false`。
- 文件路径可以是相对路径或绝对路径。
- 线性求解器参数可以直接追加，例如 `-ksp_type`、`-pc_type`、`-pc_gamg_*`、`-pc_hypre_*`、`-ksp_monitor_true_residual`、`-log_view ":file.log"`。
- `-output_prefix <路径前缀>` 控制报告、历史文件、checkpoint 和默认输出文件名。

## 已开放接口

| 接口 | 对应 mode | 算子 | 功能 | 主要输出 |
| --- | --- | --- | --- | --- |
| 几何检查 | `-mode geometry` | 任意，通常用 `low_order` | 构造控制臂 mask 和密度场，统计 mask 体积分数，可写 VTK 或 PETSc 二进制密度。 | `<prefix>_summary.txt`，可选 `.vtk`、`_rho.petscbin` |
| 密度流水线 | `-mode density` | 任意 | 做密度滤波和可选拔模方向聚合，用于优化前诊断。 | `<prefix>_density.vtk`，可选 `_rho.petscbin` |
| 单次线性求解 | `-mode solve` | `low_order`、`h8_matrix_free` 或 `emsfem_ann` | 构造一次弹性系统并求解 `K u = f`，用于验证边界、载荷和求解器。 | `<prefix>_summary.txt`，可选 `_u.petscbin`、`.vtk` |
| 拓扑优化 | `-mode optimize` | `low_order`、`h8_matrix_free` 或 `emsfem_ann` | 执行拓扑优化，包含滤波、投影、checkpoint 和预条件器设置。 | history CSV、目标/体积分数 CSV/SVG、summary、checkpoint、可选最终 VTK |
| EMsFEM ANN 补求解 | `-mode ems_ann_postsolve` | `emsfem_ann` | 读取 EMsFEM ANN 密度 checkpoint，补算位移场，不重新优化。 | postsolve summary 和位移 checkpoint |
| H8 初始 VTK | `-mode h8_initial_vtk` | `h8_matrix_free` | 写出 H8 初始滤波/mask 后密度，方便检查初始设计域。 | 降采样 `.vtk` |
| H8 密度后处理 | `-mode h8_postprocess` | `h8_matrix_free` | 将 H8 密度 checkpoint 转换成降采样 VTK。 | 降采样 `.vtk` |
| H8 全量 VTK 后处理 | `-mode h8_full_vtk` | `h8_matrix_free` | 读取 H8 密度、补算位移，写全分辨率并行 VTK。 | `.pvti` 和每个 MPI rank 的 `.vti` 分片 |

## 已开放算子

| 算子 | 用途 |
| --- | --- |
| `low_order` | 轻量级分布式 PETSc 基准，适合几何、求解器和优化烟雾测试。 |
| `h8_matrix_free` | H8 matrix-free 弹性算子和拓扑优化主路径。 |
| `emsfem_ann` | 带细密度子单元和 ANN 辅助预条件矩阵的 EMsFEM ANN 算子。 |

## 通用参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `-nx`, `-ny`, `-nz` | 节点网格尺寸，总自由度为 `3*nx*ny*nz`。 | `80 40 24` |
| `-domain_height` | 物理高度，单位米。 | `0.08` |
| `-control_arm_mask` | 是否使用控制臂专用实体/空域 mask。 | `true` |
| `-benchmark_case` | `-control_arm_mask false` 时的矩形域 benchmark 载荷：`cantilever` 或 `torsion`。 | `cantilever` |
| `-void_density` | 空域或非设计域密度。 | `0.02` |
| `-young_modulus` | 弹性模量。 | `2.1e11` |
| `-penal` | SIMP 惩罚因子。 | `3.0` |
| `-emin` | 最小刚度比例。 | `1e-6` |
| `-mask_threshold` | 写 solid-mask VTK 时使用的阈值。 | `0.5` |
| `-ab_triangle_retract` | 控制臂 A/B 三角区域回缩比例。 | `0.22` |
| `-output_prefix` | 输出文件名前缀。 | `result/control_arm_cpp` |

## 几何接口

格式：

```sh
./bin/control_arm_cpp \
  -mode geometry \
  -nx <整数> -ny <整数> -nz <整数> \
  -output_prefix <输出前缀> \
  [-write_structured_vtk true -structured_vtk_file <文件.vtk>] \
  [-write_solid_vtk true -solid_vtk_file <文件.vtk>] \
  [-write_density_binary true -density_binary_file <文件.petscbin>]
```

示例：

```sh
mpirun -np 4 ./bin/control_arm_cpp \
  -mode geometry \
  -nx 80 -ny 40 -nz 24 \
  -output_prefix result/geom_80_40_24 \
  -write_solid_vtk true \
  -write_structured_vtk true
```

## 密度流水线接口

格式：

```sh
./bin/control_arm_cpp \
  -mode density \
  -nx <整数> -ny <整数> -nz <整数> \
  -filter_radius <实数> \
  -draft_radius <整数> \
  -draft_pnorm <实数> \
  -draft_beta <实数> \
  -draft_eta <实数> \
  -draft_dirs <+x,-x,+y,-y,+z,-z 列表> \
  -draft_combine <max|...> \
  [-write_density_vtk true -density_vtk_file <文件.vtk>] \
  [-write_density_binary true -density_binary_file <文件.petscbin>]
```

示例：

```sh
mpirun -np 4 ./bin/control_arm_cpp \
  -mode density \
  -nx 80 -ny 40 -nz 24 \
  -filter_radius 1.5 \
  -draft_radius 1 \
  -draft_dirs +z \
  -write_density_vtk true \
  -write_density_binary true \
  -output_prefix result/density_80_40_24
```

## 论文验证与经典 benchmark 算例

### 第一层：拔模约束验证

使用 `-mode density` 和 `-control_arm_mask false` 可以生成规则矩形设计域的密度流水线结果。输出 VTK 中包含 `rho_initial`、`rho_filtered` 和 `rho_projected`，在 ParaView 中可以直接做三列对比。

无拔模约束：

```sh
mpirun -np 4 ./bin/control_arm_cpp \
  -mode density \
  -control_arm_mask false \
  -nx 121 -ny 41 -nz 31 \
  -filter_radius 1.5 \
  -draft_radius 0 \
  -draft_dirs +z \
  -write_density_vtk true \
  -density_vtk_file result/layer1_nodraft.vtk \
  -write_density_binary true \
  -output_prefix result/layer1_nodraft
```

`+z` 单向拔模、`-z` 单向拔模和 `+z/-z` split draw：

```sh
mpirun -np 4 ./bin/control_arm_cpp -mode density -control_arm_mask false \
  -nx 121 -ny 41 -nz 31 -filter_radius 1.5 \
  -draft_radius 6 -draft_pnorm 8 -draft_beta 8 -draft_eta 0.5 \
  -draft_dirs +z -draft_combine max \
  -write_density_vtk true -density_vtk_file result/layer1_plus_z.vtk \
  -write_density_binary true -output_prefix result/layer1_plus_z

mpirun -np 4 ./bin/control_arm_cpp -mode density -control_arm_mask false \
  -nx 121 -ny 41 -nz 31 -filter_radius 1.5 \
  -draft_radius 6 -draft_pnorm 8 -draft_beta 8 -draft_eta 0.5 \
  -draft_dirs -z -draft_combine max \
  -write_density_vtk true -density_vtk_file result/layer1_minus_z.vtk \
  -write_density_binary true -output_prefix result/layer1_minus_z

mpirun -np 4 ./bin/control_arm_cpp -mode density -control_arm_mask false \
  -nx 121 -ny 41 -nz 31 -filter_radius 1.5 \
  -draft_radius 6 -draft_pnorm 8 -draft_beta 8 -draft_eta 0.5 \
  -draft_dirs +z,-z -draft_combine max \
  -write_density_vtk true -density_vtk_file result/layer1_split_z.vtk \
  -write_density_binary true -output_prefix result/layer1_split_z
```

### 第二层：经典矩形域结构 benchmark

设置 `-control_arm_mask false` 后，程序使用无 mask 的规则矩形设计域。benchmark 接口固定左端面；`-benchmark_case cantilever` 对应右端弯曲载荷，`-benchmark_case torsion` 对应右端绕 x 轴扭转载荷。下面命令使用完整模型；如果论文需要严格半模型或四分之一模型，还需要额外增加分量式对称面约束。

H8 无拔模基准：

```sh
mpirun -np 8 ./bin/control_arm_cpp \
  -mode optimize \
  -operator h8_matrix_free \
  -control_arm_mask false \
  -benchmark_case cantilever \
  -nx 121 -ny 41 -nz 31 \
  -volfrac 0.30 \
  -opt_max_iter 200 \
  -opt_filter_radius 1.5 \
  -opt_z_draft_closure false \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval 50 \
  -h8_pc_type aux_hypre \
  -ksp_type fgmres \
  -ksp_gmres_restart 200 \
  -output_prefix result/layer2_h8_cantilever_nodraft
```

H8 `+Z` 拔模、H8 `+Z` 拔模加 Heaviside，以及 EMsFEM ANN `+Z` 拔模：

```sh
mpirun -np 8 ./bin/control_arm_cpp -mode optimize -operator h8_matrix_free \
  -control_arm_mask false -benchmark_case cantilever \
  -nx 121 -ny 41 -nz 31 -volfrac 0.30 -opt_max_iter 200 \
  -opt_filter_radius 1.5 -opt_z_draft_closure true -opt_z_draft_eta 0.5 \
  -opt_write_checkpoint true -opt_checkpoint_interval 50 \
  -h8_pc_type aux_hypre -ksp_type fgmres -ksp_gmres_restart 200 \
  -output_prefix result/layer2_h8_cantilever_zdraft

mpirun -np 8 ./bin/control_arm_cpp -mode optimize -operator h8_matrix_free \
  -control_arm_mask false -benchmark_case cantilever \
  -nx 121 -ny 41 -nz 31 -volfrac 0.30 -opt_max_iter 200 \
  -opt_filter_radius 1.5 -opt_z_draft_closure true -opt_z_draft_eta 0.5 \
  -opt_heaviside_projection true -opt_heaviside_eta 0.5 \
  -opt_heaviside_beta_initial 1 -opt_heaviside_beta_max 16 \
  -opt_heaviside_beta_interval 50 \
  -opt_write_checkpoint true -opt_checkpoint_interval 50 \
  -h8_pc_type aux_hypre -ksp_type fgmres -ksp_gmres_restart 200 \
  -output_prefix result/layer2_h8_cantilever_zdraft_heaviside

mpirun -np 8 ./bin/control_arm_cpp -mode optimize -operator emsfem_ann \
  -control_arm_mask false -control_arm_bc false -benchmark_case cantilever \
  -nx 121 -ny 41 -nz 31 -ems_sub_n 5 -ems_ann_dir ../input_5 \
  -volfrac 0.30 -opt_max_iter 200 -opt_filter_radius 1.5 \
  -opt_z_draft_closure true -opt_z_draft_eta 0.5 \
  -opt_write_checkpoint true -opt_checkpoint_interval 50 \
  -ems_pc_type aux_ann_hypre -ksp_type fgmres -ksp_gmres_restart 200 \
  -output_prefix result/layer2_ems_ann_cantilever_zdraft
```

扭转梁只需要把同一套命令中的 benchmark 和输出前缀改掉：

```sh
  -benchmark_case torsion \
  -output_prefix result/layer2_h8_torsion_zdraft
```

## 求解接口

格式：

```sh
./bin/control_arm_cpp \
  -mode solve \
  -operator <low_order|h8_matrix_free|emsfem_ann> \
  -nx <整数> -ny <整数> -nz <整数> \
  -load <实数> \
  [-write_solution true -solution_file <文件.petscbin>] \
  [-write_structured_vtk true -structured_vtk_file <文件.vtk>] \
  [-write_solid_vtk true -solid_vtk_file <文件.vtk>] \
  [PETSc KSP/PC 参数]
```

示例：

```sh
mpirun -np 4 ./bin/control_arm_cpp \
  -mode solve \
  -operator h8_matrix_free \
  -nx 80 -ny 40 -nz 24 \
  -load 1.0 \
  -ksp_type cg \
  -pc_type jacobi \
  -ksp_monitor_true_residual \
  -write_solution true \
  -write_structured_vtk true \
  -output_prefix result/h8_solve_80_40_24
```

## 优化接口

格式：

```sh
./bin/control_arm_cpp \
  -mode optimize \
  -operator <low_order|h8_matrix_free|emsfem_ann> \
  -nx <整数> -ny <整数> -nz <整数> \
  -volfrac <实数> \
  -opt_max_iter <整数> \
  -opt_filter_radius <实数> \
  -opt_move <实数> \
  -opt_ksp_rtol <实数> \
  -opt_ksp_max_it <整数> \
  [-control_arm_mask false -benchmark_case <cantilever|torsion>] \
  [-opt_heaviside_projection true -opt_heaviside_eta <实数> \
   -opt_heaviside_beta_initial <实数> -opt_heaviside_beta_max <实数> \
   -opt_heaviside_beta_interval <整数>] \
  [-opt_z_draft_closure true -opt_z_draft_eta <实数>] \
  [-opt_write_checkpoint true -opt_checkpoint_interval <整数> \
   -opt_checkpoint_prefix <前缀>] \
  [-opt_write_final_vtk true -opt_vtk_file <文件.vtk>] \
  [算子专用参数] [PETSc KSP/PC 参数]
```

常用优化参数：

| 参数 | 含义 |
| --- | --- |
| `-opt_move`, `-opt_move_min`, `-opt_move_shrink`, `-opt_move_growth` | 移动限控制。 |
| `-opt_max_compliance_increase` | 稳定性保护阈值，用于拒绝不可靠更新。 |
| `-opt_projected_volume_correction` | 投影后体积分数修正。 |
| `-opt_rho_min` | 设计变量密度下界。 |
| `-opt_write_checkpoint` | 写 PETSc 二进制密度/mask checkpoint。 |
| `-opt_stop_on_ksp_divergence` | KSP 发散时停止或跳过不可靠更新。 |
| `-benchmark_case` | `-control_arm_mask false` 时选择 `cantilever` 或 `torsion`。 |

### H8 专用参数

| 参数 | 取值 | 含义 |
| --- | --- | --- |
| `-h8_pc_type` | `block_jacobi`, `jacobi`, `petsc`, `aux_gamg`, `aux_hypre`, `aux_elastic_gamg`, `aux_elastic_hypre` | 选择 H8 预条件器。生产 H8 通常用 `aux_hypre` 或 `aux_elastic_hypre`。 |
| `-h8_dm_px`, `-h8_dm_py`, `-h8_dm_pz` | 正整数 | 手动指定 H8 三维进程网格，乘积必须等于 MPI 进程数。 |
| `-h8_load_case` | 整数 | 选择 H8 控制臂载荷工况。 |
| `-h8_include_spring_load` | `true|false` | 是否包含弹簧/衬套区域载荷。 |
| `-benchmark_case` | `cantilever`, `torsion` | 选择无 mask 矩形域 benchmark 载荷。 |
| `-h8_aux_rebuild_interval` | 整数 | 每隔多少步重建辅助矩阵。 |
| `-h8_reuse_initial_guess` | `true|false` | 是否复用上一轮位移作为 KSP 初值。 |

H8 直接运行示例：

```sh
mpirun -np 28 ./bin/control_arm_cpp \
  -mode optimize \
  -operator h8_matrix_free \
  -nx 325 -ny 186 -nz 56 \
  -volfrac 0.30 \
  -opt_max_iter 5 \
  -opt_filter_radius 1.5 \
  -opt_write_checkpoint true \
  -opt_checkpoint_interval 1 \
  -h8_pc_type aux_hypre \
  -ksp_type fgmres \
  -ksp_gmres_restart 200 \
  -pc_hypre_type boomeramg \
  -pc_hypre_boomeramg_strong_threshold 0.5 \
  -output_prefix result/h8_10m_direct
```

### EMsFEM ANN 专用参数

| 参数 | 取值 | 含义 |
| --- | --- | --- |
| `-ems_ann_dir` | 路径 | ANN 模型目录，例如 `../input_5` 或 `../input_20`。 |
| `-ems_sub_n` | >= 2 的整数 | 每个粗单元每个方向的细密度子单元数量。 |
| `-ems_cache_element_matrices` | `true|false` | 是否缓存 EMsFEM 单元矩阵；优化模式要求为 `true`。 |
| `-ems_cache_gib_limit` | 实数 | 每个 rank 的缓存上限，单位 GiB；`0` 表示不限。 |
| `-control_arm_bc` | `true|false` | 是否使用控制臂专用边界条件。 |
| `-benchmark_case` | `cantilever`, `torsion` | `-control_arm_bc false` 时选择无 mask 矩形域 benchmark 载荷。 |
| `-load_case` | 整数 | 在生产脚本中，`0` 表示加权多工况载荷。 |
| `-include_spring_load` | `true|false` | 是否包含弹簧/衬套区域载荷。 |
| `-ems_pc_type` | `block_jacobi`, `jacobi`, `petsc`, `aux_gamg`, `aux_hypre`, `aux_elastic_gamg`, `aux_elastic_hypre`, `aux_ann_gamg`, `aux_ann_hypre` | 选择 EMsFEM ANN 预条件器。生产默认 `aux_ann_gamg`，HYPRE 对照可用 `aux_ann_hypre`。 |
| `-ems_dm_px`, `-ems_dm_py`, `-ems_dm_pz` | 正整数 | 手动指定 EMsFEM 三维进程网格，乘积必须等于 MPI 进程数。 |

EMsFEM ANN 直接运行示例：

```sh
mpirun -np 56 ./bin/control_arm_cpp \
  -mode optimize \
  -operator emsfem_ann \
  -nx 325 -ny 186 -nz 56 \
  -ems_sub_n 5 \
  -ems_ann_dir ../input_5 \
  -control_arm_bc true \
  -load_case 0 \
  -include_spring_load true \
  -volfrac 0.30 \
  -opt_max_iter 10 \
  -opt_filter_radius 1.5 \
  -ems_pc_type aux_ann_gamg \
  -pc_gamg_type agg \
  -pc_gamg_threshold 0.02 \
  -ksp_type fgmres \
  -ksp_gmres_restart 200 \
  -output_prefix result/ems_ann_10m_direct
```

## 后处理接口

H8 降采样密度 VTK：

```sh
mpirun -np 28 ./bin/control_arm_cpp \
  -mode h8_postprocess \
  -operator h8_matrix_free \
  -nx 700 -ny 400 -nz 120 \
  -density_file result/run_checkpoint_final_rho_phys.petscbin \
  -mask_file result/run_checkpoint_final_mask.petscbin \
  -post_vtk_file result/run_rho_phys_stride5.vtk \
  -post_stride 5 \
  -post_max_samples 2000000 \
  -output_prefix result/run_post
```

H8 全量密度和位移 VTK：

```sh
mpirun -np 224 ./bin/control_arm_cpp \
  -mode h8_full_vtk \
  -operator h8_matrix_free \
  -nx 700 -ny 400 -nz 120 \
  -density_file result/run_checkpoint_final_rho_phys.petscbin \
  -mask_file result/run_checkpoint_final_mask.petscbin \
  -post_vtk_file result/run_full_density_displacement.pvti \
  -h8_pc_type block_jacobi \
  -output_prefix result/run_full_vtk
```

EMsFEM ANN 补求解：

```sh
mpirun -np 56 ./bin/control_arm_cpp \
  -mode ems_ann_postsolve \
  -operator emsfem_ann \
  -nx 325 -ny 186 -nz 56 \
  -ems_sub_n 5 \
  -ems_ann_dir ../input_5 \
  -density_file result/run_checkpoint_final_rho_phys.petscbin \
  -mask_file result/run_checkpoint_final_mask.petscbin \
  -ems_pc_type aux_ann_gamg \
  -output_prefix result/run_postsolve
```

## Slurm 脚本接口

sbatch 脚本是对上述命令行接口的封装，主要通过环境变量传参：

```sh
sbatch --export=ALL,KEY=value,KEY=value script.sbatch
```

| 脚本 | 功能 | 常用导出变量 |
| --- | --- | --- |
| `submit_h8_preflight.sbatch` | 检查 PETSc 路径、脚本语法、编译器、MPI 启动器和程序启动。 | `PETSC_DIR`, `PETSC_ARCH`, `MPI_LAUNCHER` |
| `submit_mpi_launcher_check.sbatch` | 检查 `srun`/`mpirun` 在多节点上的行为。 | `MPI_LAUNCHER`, `MPI_LAUNCHER_ARGS` |
| `submit_h8_opt_10m.sbatch` | H8 约 10M 自由度诊断优化。 | `NX`, `NY`, `NZ`, `OPT_MAX_ITER`, `H8_PC_TYPE`, `PETSC_EXTRA_OPTIONS` |
| `submit_h8_opt_100m.sbatch` | H8 约 100M 自由度生产优化。 | `NX`, `NY`, `NZ`, `OPT_MAX_ITER`, `H8_PC_TYPE`, `H8_DM_PX/PY/PZ`, `RUN_LABEL` |
| `submit_h8_opt_100m_1step.sbatch` | H8 100M 一步烟雾测试。 | 同 H8 生产脚本 |
| `submit_ems_ann_scale.sbatch` | EMsFEM ANN 优化。 | `NELX/NELY/NELZ` 或 `NX/NY/NZ`, `EMS_SUB_N`, `ANN_DIR`, `EMS_PC_TYPE`, `GAMG_PROFILE`, `HYPRE_PROFILE`, `ALLOW_HUGE_EMS` |
| `submit_h8_postprocess_100m.sbatch` | 将 H8 checkpoint 转为降采样 VTK。 | `SOURCE_PREFIX`, `DENSITY_FILE`, `MASK_FILE`, `POST_STRIDE`, `VTK_FILE` |
| `submit_h8_full_vtk_100m.sbatch` | 写 H8 全分辨率 `.pvti` 和 `.vti` 分片。 | `SOURCE_PREFIX`, `DENSITY_FILE`, `MASK_FILE`, `VTK_FILE`, `H8_PC_TYPE` |
| `submit_ems_ann_full_vtk.sbatch` | 通过 Python 转换器将 EMsFEM ANN 细密度 checkpoint 转为全量 VTK。 | `CHECKPOINT_PREFIX`, `DENSITY_FILE`, `MASK_FILE`, `VTK_FILE`, `EMS_SUB_N` |

新超算家目录上的预检查示例：

```sh
sbatch --export=ALL,PETSC_DIR=/data/home/dlut_ycx/petsc/petsc-v3.19.3,PETSC_ARCH=arch-linux-c-opt-hypre \
  submit_h8_preflight.sbatch
```

H8 生产提交示例：

```sh
sbatch --nodes=8 --ntasks-per-node=28 \
  --export=ALL,PETSC_DIR=/data/home/dlut_ycx/petsc/petsc-v3.19.3,PETSC_ARCH=arch-linux-c-opt-hypre,H8_PC_TYPE=aux_hypre,RUN_LABEL=h8_100m \
  submit_h8_opt_100m.sbatch
```

EMsFEM ANN 提交示例：

```sh
sbatch --nodes=48 --ntasks-per-node=28 \
  --export=ALL,PETSC_DIR=/data/home/dlut_ycx/petsc/petsc-v3.19.3,PETSC_ARCH=arch-linux-c-opt-hypre,EMS_SUB_N=5,ANN_DIR=../input_5,EMS_PC_TYPE=aux_ann_gamg,GAMG_PROFILE=strong \
  submit_ems_ann_scale.sbatch
```
