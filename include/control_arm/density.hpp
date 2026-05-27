#pragma once

#include "control_arm/geometry.hpp"

#include <petscvec.h>

namespace control_arm {

PetscErrorCode create_nodal_density_vec(MPI_Comm comm, const Grid &grid,
                                        const DensityOptions &options,
                                        Vec *rho);

PetscErrorCode density_vec_average(Vec rho, PetscReal *average);

PetscErrorCode write_density_binary(Vec rho, const char *path);

} // namespace control_arm
