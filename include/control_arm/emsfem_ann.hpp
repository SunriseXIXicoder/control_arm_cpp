#pragma once

#include "control_arm/ann_model.hpp"
#include "control_arm/geometry.hpp"
#include "control_arm/low_order_optimizer.hpp"

#include <petscksp.h>

namespace control_arm {

struct EmSfemAnnOptions {
  char ann_dir[PETSC_MAX_PATH_LEN] = "../input_20";
  PetscInt sub_n = 20;
  PetscBool cache_element_matrices = PETSC_TRUE;
  PetscReal cache_gib_limit = 0.0;
  PetscBool control_arm_bc = PETSC_FALSE;
  PetscInt load_case = 2;
  PetscBool include_spring_load = PETSC_TRUE;
};

PetscErrorCode create_emsfem_ann_system(const Grid &grid,
                                        const DensityOptions &density_options,
                                        const EmSfemAnnOptions &ems_options,
                                        PetscReal load_scale,
                                        Mat *A,
                                        Vec *u,
                                        Vec *b);

PetscErrorCode write_emsfem_ann_solution_vtk(Mat A,
                                             Vec displacement,
                                             const char *path,
                                             const Grid &grid,
                                             const DensityOptions &options);

PetscErrorCode run_emsfem_ann_optimizer(const Grid &grid,
                                        const DensityOptions &density_options,
                                        const OptimizerOptions &optimizer_options,
                                        const EmSfemAnnOptions &ems_options,
                                        const char *output_prefix,
                                        const char *final_vtk_file);

} // namespace control_arm
