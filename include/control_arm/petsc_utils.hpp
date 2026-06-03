#pragma once

#include "control_arm/geometry.hpp"

#include <petscksp.h>

#include <vector>

namespace control_arm {

struct ObjectiveVolumePoint {
  ObjectiveVolumePoint() = default;
  ObjectiveVolumePoint(PetscInt iter_in, PetscReal objective_in,
                       PetscReal volume_in)
      : iter(iter_in), objective(objective_in), volume(volume_in) {}

  PetscInt iter = 0;
  PetscReal objective = 0.0;
  PetscReal volume = 0.0;
};

void count_prealloc_column(PetscInt col, PetscInt cstart, PetscInt cend,
                           PetscInt *dnz, PetscInt *onz);

PetscErrorCode write_geometry_report(const char *output_prefix,
                                     const Grid &grid,
                                     const DensityOptions &options,
                                     PetscMPIInt ranks,
                                     PetscReal volume_fraction,
                                     PetscInt solid_cells,
                                     PetscInt total_cells);

PetscErrorCode write_solve_report(const char *output_prefix,
                                  const Grid &grid,
                                  PetscMPIInt ranks,
                                  const char *operator_type,
                                  PetscLogDouble assembly_time,
                                  PetscLogDouble solve_time,
                                  PetscInt iterations,
                                  PetscReal residual_norm,
                                  PetscReal compliance);

PetscErrorCode write_objective_volume_history(
    const char *output_prefix,
    const std::vector<ObjectiveVolumePoint> &points,
    PetscReal volume_target);

PetscErrorCode load_vec_binary_checkpoint(Vec v, const char *path);

PetscErrorCode load_dmda_natural_binary_checkpoint(DM da, Vec v,
                                                   const char *path);

} // namespace control_arm
