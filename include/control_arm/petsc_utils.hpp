#pragma once

#include "control_arm/geometry.hpp"

#include <petscksp.h>

namespace control_arm {

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

} // namespace control_arm
