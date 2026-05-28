#pragma once

#include "control_arm/geometry.hpp"

#include <petscdmda.h>

namespace control_arm {

struct DensityPipelineOptions {
  PetscReal filter_radius = 1.5;
  PetscInt draft_radius = 0;
  PetscReal draft_pnorm = 8.0;
  PetscReal draft_beta = 8.0;
  PetscReal draft_eta = 0.5;
  char draft_dirs[128] = "+x,-x,+y,-y,+z,-z";
  char draft_combine[32] = "max";
};

PetscErrorCode run_density_pipeline(const Grid &grid,
                                    const DensityOptions &density_options,
                                    const DensityPipelineOptions &pipeline_options,
                                    const char *output_prefix,
                                    PetscBool write_vtk,
                                    const char *vtk_file,
                                    PetscBool write_binary,
                                    const char *binary_file);

} // namespace control_arm
