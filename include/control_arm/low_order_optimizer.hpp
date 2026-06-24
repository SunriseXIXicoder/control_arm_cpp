#pragma once

#include "control_arm/geometry.hpp"

#include <petscksp.h>

namespace control_arm {

struct OptimizerOptions {
  PetscInt max_iter = 20;
  PetscReal volfrac = 0.30;
  PetscReal move = 0.08;
  PetscReal move_min = 0.005;
  PetscReal move_shrink = 0.5;
  PetscReal move_growth = 1.05;
  PetscReal max_compliance_increase = 0.10;
  PetscBool stability_guard = PETSC_TRUE;
  PetscBool projected_volume_correction = PETSC_TRUE;
  PetscReal rho_min = 1.0e-3;
  PetscBool heaviside_projection = PETSC_FALSE;
  PetscReal heaviside_eta = 0.50;
  PetscReal heaviside_beta_initial = 1.0;
  PetscReal heaviside_beta_max = 16.0;
  PetscInt heaviside_beta_interval = 0;
  PetscBool z_draft_closure = PETSC_TRUE;
  PetscReal z_draft_eta = 0.50;
  PetscReal load = 1.0;
  PetscReal ksp_rtol = 1.0e-6;
  PetscInt ksp_max_it = 1000;
  PetscInt vtk_interval = 0;
  PetscInt vtk_max_points = 5000000;
  PetscBool write_final_vtk = PETSC_TRUE;
  PetscReal filter_radius = 0.0;
  PetscBool write_checkpoint = PETSC_FALSE;
  PetscInt checkpoint_interval = 0;
  // 无控制臂 mask 的矩形域 benchmark 工况：cantilever 为端部弯曲，torsion 为端部扭转载荷。
  char benchmark_case[32] = "cantilever";
  char checkpoint_prefix[PETSC_MAX_PATH_LEN] = "";
};

PetscErrorCode run_low_order_optimizer(const Grid &grid,
                                       const DensityOptions &density_options,
                                       const OptimizerOptions &optimizer_options,
                                       const char *output_prefix,
                                       const char *final_vtk_file);

} // namespace control_arm
