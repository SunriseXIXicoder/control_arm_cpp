#pragma once

#include "control_arm/geometry.hpp"

#include <petscdmda.h>
#include <petscksp.h>

namespace control_arm {

PetscErrorCode create_h8_matrix_free_system(const Grid &grid,
                                            const DensityOptions &options,
                                            PetscReal load,
                                            Mat *A,
                                            Vec *u,
                                            Vec *b);

PetscErrorCode write_h8_solution_vtk(Mat A, Vec displacement,
                                     const char *path,
                                     const Grid &grid,
                                     const DensityOptions &options);

} // namespace control_arm
