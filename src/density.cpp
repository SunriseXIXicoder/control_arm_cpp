#include "control_arm/density.hpp"

#include <petscviewer.h>

namespace control_arm {

PetscErrorCode create_nodal_density_vec(MPI_Comm comm, const Grid &grid,
                                        const DensityOptions &options,
                                        Vec *rho) {
  PetscCall(VecCreate(comm, rho));
  PetscCall(VecSetSizes(*rho, PETSC_DECIDE, node_count(grid)));
  PetscCall(VecSetFromOptions(*rho));

  PetscInt start = 0;
  PetscInt end = 0;
  PetscCall(VecGetOwnershipRange(*rho, &start, &end));
  for (PetscInt node = start; node < end; ++node) {
    const PetscInt k = node / (grid.nx * grid.ny);
    const PetscInt rem = node - k * grid.nx * grid.ny;
    const PetscInt j = rem / grid.nx;
    const PetscInt i = rem - j * grid.nx;
    const PetscScalar value = node_density(i, j, k, grid, options);
    PetscCall(VecSetValue(*rho, node, value, INSERT_VALUES));
  }

  PetscCall(VecAssemblyBegin(*rho));
  PetscCall(VecAssemblyEnd(*rho));
  return 0;
}

PetscErrorCode density_vec_average(Vec rho, PetscReal *average) {
  PetscScalar sum = 0.0;
  PetscInt n = 0;
  PetscCall(VecSum(rho, &sum));
  PetscCall(VecGetSize(rho, &n));
  *average = n > 0 ? PetscRealPart(sum) / static_cast<PetscReal>(n) : 0.0;
  return 0;
}

PetscErrorCode write_density_binary(Vec rho, const char *path) {
  PetscViewer viewer = nullptr;
  PetscCall(PetscViewerBinaryOpen(PetscObjectComm(reinterpret_cast<PetscObject>(rho)),
                                  path, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(rho, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PetscObjectComm(reinterpret_cast<PetscObject>(rho)),
                        "Wrote distributed density Vec: %s\n", path));
  return 0;
}

} // namespace control_arm
