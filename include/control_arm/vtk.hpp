#pragma once

#include "control_arm/geometry.hpp"

#include <petscvec.h>

namespace control_arm {

PetscErrorCode write_structured_density_vtk(MPI_Comm comm, const char *path,
                                            const Grid &grid,
                                            const DensityOptions &options);

PetscErrorCode write_structured_solution_vtk(Vec displacement,
                                             const char *path,
                                             const Grid &grid,
                                             const DensityOptions &options);

PetscErrorCode write_solid_mask_vtk(MPI_Comm comm, const char *path,
                                    const Grid &grid,
                                    const DensityOptions &options,
                                    PetscInt max_cells);

} // namespace control_arm
