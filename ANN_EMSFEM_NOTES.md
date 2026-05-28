# ANN + EMsFEM PETSc Path

This folder now has an ANN-based extended multiscale FEM solve path:

```sh
-mode solve -operator emsfem_ann
```

The implementation follows the MATLAB flow:

- read the four `input_5/model_weight_3DEMs_FEMBLC_20220821_input_5_layer*.json`
  neural networks;
- use one 5x5x5 material block per coarse element as ANN input;
- predict the internal multiscale shape-function rows;
- recover one 24x24 coarse EMsFEM element stiffness by
  `sum(B' * KE_fine * B * E_fine)`;
- solve with PETSc `MatShell`, so no global MATLAB sparse matrix is formed.

Large-scale memory rules:

- The code never stores MATLAB-style `all_shape_functions`.
- By default it stores only local 24x24 coarse element matrices on each MPI
  rank. This is controlled by `-ems_cache_element_matrices true|false`.
- Use `-ems_cache_gib_limit <GiB>` to prevent accidental over-allocation.
- For extreme runs, use `-write_structured_vtk false` and keep distributed
  PETSc logs/checkpoints instead of root-gathered VTK.

WSL smoke test:

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
sh run_wsl_ems_ann_smoke.sh
```

MATLAB-style 40x20x8 coarse-grid control-arm boundary test:

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
sh run_wsl_ems_ann_control_arm_40_20_8.sh
```

`NELX/NELY/NELZ` are coarse-element counts. PETSc uses nodal counts internally,
so the script passes `41x21x9` nodes for a MATLAB `40x20x8` coarse grid.

Complete optimization path:

```sh
cd "/mnt/c/Users/administered/Desktop/PIML拔模SIMP/control_arm_cpp"
ITERS=50 sh run_wsl_ems_ann_optimize_40_20_8.sh
```

By default `LOAD_CASE=0`, which solves the three previous control-arm load
cases separately and accumulates compliance/sensitivity with the MATLAB AHP
weights.  Set `LOAD_CASE=1`, `2`, or `3` for a single load case.

For a longer run:

```sh
ITERS=500 WRITE_VTK=false sh run_wsl_ems_ann_optimize_40_20_8.sh
```

The optimizer writes:

- `*_history.csv`
- `*_opt_summary.txt`
- `*_checkpoint_iter_*.petscbin`
- `*_checkpoint_final_rho_design.petscbin`
- `*_checkpoint_final_rho_phys.petscbin`
- optional `*_final.vtk`

Current scope:

- `emsfem_ann` is available for solve-mode verification and optimize-mode
  topology optimization.
- The optimize path uses one design density per coarse EMsFEM element and
  repeats that value over the 5x5x5 ANN input block.  This keeps the memory
  model scalable for 100M-DOF displacement solves.
- Sensitivity currently uses a frozen-ANN-shape energy approximation:
  `dK/drho ~= (dE/drho/E) K`.  This avoids storing global shape functions or
  ANN Jacobians.  Exact fine-subcell sensitivity is a later, much heavier path.
- The WSL script uses a relaxed absolute tolerance for the 40x20x8 control-arm
  boundary smoke test because Jacobi is only a placeholder preconditioner for
  the ANN multiscale operator. Cluster-scale production should use a stronger
  matrix-free preconditioner or a small-scale explicit-AIJ verification path.
