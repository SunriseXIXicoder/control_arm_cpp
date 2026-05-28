#pragma once

#include "control_arm/geometry.hpp"
#include "control_arm/low_order_optimizer.hpp"

#include <petscksp.h>

namespace control_arm {

PetscErrorCode run_h8_optimizer(const Grid &grid,
                                const DensityOptions &density_options,
                                const OptimizerOptions &optimizer_options,
                                const char *output_prefix,
                                const char *final_vtk_file);

} // namespace control_arm
