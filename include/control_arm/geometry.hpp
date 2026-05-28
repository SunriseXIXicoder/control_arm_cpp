#pragma once

#include <petscsys.h>

namespace control_arm {

struct Grid {
  PetscInt nx = 80;
  PetscInt ny = 40;
  PetscInt nz = 24;
};

struct DensityOptions {
  PetscBool use_control_arm_mask = PETSC_TRUE;
  PetscReal void_density = 0.02;
  PetscReal young_modulus = 2.1e11;
  PetscReal penal = 3.0;
  PetscReal emin = 1.0e-6;
  PetscReal mask_threshold = 0.5;
  PetscReal ab_triangle_retract = 0.22;
};

PetscReal domain_length(const Grid &grid);
PetscReal domain_width(const Grid &grid);
PetscReal domain_height(const Grid &grid);

PetscReal density_at_normalized(PetscReal x, PetscReal y, PetscReal z,
                                const Grid &grid,
                                const DensityOptions &options);

PetscReal node_density(PetscInt i, PetscInt j, PetscInt k,
                       const Grid &grid,
                       const DensityOptions &options);

PetscReal cell_density(PetscInt i, PetscInt j, PetscInt k,
                       const Grid &grid,
                       const DensityOptions &options);

PetscReal edge_stiffness(PetscReal rho0, PetscReal rho1,
                         const DensityOptions &options);

PetscInt node_count(const Grid &grid);
PetscInt dof_count(const Grid &grid);
PetscInt cell_count(const Grid &grid);
PetscInt global_dof(PetscInt i, PetscInt j, PetscInt k, PetscInt c,
                    const Grid &grid);

PetscErrorCode compute_mask_volume_fraction(MPI_Comm comm, const Grid &grid,
                                            const DensityOptions &options,
                                            PetscReal *volume_fraction,
                                            PetscInt *solid_cells,
                                            PetscInt *total_cells);

} // namespace control_arm
