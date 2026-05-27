#include "control_arm/vtk.hpp"

#include <petscsys.h>

#include <cstdio>

namespace control_arm {

PetscErrorCode write_structured_density_vtk(MPI_Comm comm, const char *path,
                                            const Grid &grid,
                                            const DensityOptions &options) {
  PetscMPIInt rank = 0;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (rank == 0) {
    FILE *fp = std::fopen(path, "w");
    PetscCheck(fp != nullptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Cannot open structured VTK file: %s", path);

    const PetscReal hx = domain_length(grid) / static_cast<PetscReal>(grid.nx - 1);
    const PetscReal hy = domain_width(grid) / static_cast<PetscReal>(grid.ny - 1);
    const PetscReal hz = domain_height(grid) / static_cast<PetscReal>(grid.nz - 1);

    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "C++ PETSc control-arm density\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    std::fprintf(fp, "DIMENSIONS %lld %lld %lld\n", static_cast<long long>(grid.nx),
                 static_cast<long long>(grid.ny), static_cast<long long>(grid.nz));
    std::fprintf(fp, "ORIGIN 0 0 0\n");
    std::fprintf(fp, "SPACING %.17g %.17g %.17g\n", static_cast<double>(hx),
                 static_cast<double>(hy), static_cast<double>(hz));
    std::fprintf(fp, "POINT_DATA %lld\n", static_cast<long long>(node_count(grid)));
    std::fprintf(fp, "SCALARS density double 1\n");
    std::fprintf(fp, "LOOKUP_TABLE default\n");
    for (PetscInt k = 0; k < grid.nz; ++k) {
      for (PetscInt j = 0; j < grid.ny; ++j) {
        for (PetscInt i = 0; i < grid.nx; ++i) {
          std::fprintf(fp, "%.17e\n",
                       static_cast<double>(node_density(i, j, k, grid, options)));
        }
      }
    }
    std::fclose(fp);
  }

  PetscCall(PetscPrintf(comm, "Wrote structured density VTK: %s\n", path));
  return 0;
}

PetscErrorCode write_structured_solution_vtk(Vec displacement,
                                             const char *path,
                                             const Grid &grid,
                                             const DensityOptions &options) {
  PetscMPIInt rank = 0;
  VecScatter scatter = nullptr;
  Vec seq = nullptr;
  MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(displacement));

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecScatterCreateToZero(displacement, &scatter, &seq));
  PetscCall(VecScatterBegin(scatter, displacement, seq, INSERT_VALUES,
                            SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, displacement, seq, INSERT_VALUES,
                          SCATTER_FORWARD));

  if (rank == 0) {
    const PetscScalar *u = nullptr;
    FILE *fp = std::fopen(path, "w");
    PetscCheck(fp != nullptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Cannot open solution VTK file: %s", path);

    const PetscReal hx = domain_length(grid) / static_cast<PetscReal>(grid.nx - 1);
    const PetscReal hy = domain_width(grid) / static_cast<PetscReal>(grid.ny - 1);
    const PetscReal hz = domain_height(grid) / static_cast<PetscReal>(grid.nz - 1);

    PetscCall(VecGetArrayRead(seq, &u));
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "C++ PETSc low_order control-arm result\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    std::fprintf(fp, "DIMENSIONS %lld %lld %lld\n", static_cast<long long>(grid.nx),
                 static_cast<long long>(grid.ny), static_cast<long long>(grid.nz));
    std::fprintf(fp, "ORIGIN 0 0 0\n");
    std::fprintf(fp, "SPACING %.17g %.17g %.17g\n", static_cast<double>(hx),
                 static_cast<double>(hy), static_cast<double>(hz));
    std::fprintf(fp, "POINT_DATA %lld\n", static_cast<long long>(node_count(grid)));

    std::fprintf(fp, "VECTORS displacement double\n");
    for (PetscInt k = 0; k < grid.nz; ++k) {
      for (PetscInt j = 0; j < grid.ny; ++j) {
        for (PetscInt i = 0; i < grid.nx; ++i) {
          const PetscInt node = i + grid.nx * (j + grid.ny * k);
          std::fprintf(fp, "%.17e %.17e %.17e\n",
                       static_cast<double>(PetscRealPart(u[3 * node + 0])),
                       static_cast<double>(PetscRealPart(u[3 * node + 1])),
                       static_cast<double>(PetscRealPart(u[3 * node + 2])));
        }
      }
    }

    std::fprintf(fp, "SCALARS displacement_magnitude double 1\n");
    std::fprintf(fp, "LOOKUP_TABLE default\n");
    for (PetscInt k = 0; k < grid.nz; ++k) {
      for (PetscInt j = 0; j < grid.ny; ++j) {
        for (PetscInt i = 0; i < grid.nx; ++i) {
          const PetscInt node = i + grid.nx * (j + grid.ny * k);
          const PetscReal ux = PetscRealPart(u[3 * node + 0]);
          const PetscReal uy = PetscRealPart(u[3 * node + 1]);
          const PetscReal uz = PetscRealPart(u[3 * node + 2]);
          std::fprintf(fp, "%.17e\n",
                       static_cast<double>(PetscSqrtReal(ux * ux + uy * uy + uz * uz)));
        }
      }
    }

    std::fprintf(fp, "SCALARS density double 1\n");
    std::fprintf(fp, "LOOKUP_TABLE default\n");
    for (PetscInt k = 0; k < grid.nz; ++k) {
      for (PetscInt j = 0; j < grid.ny; ++j) {
        for (PetscInt i = 0; i < grid.nx; ++i) {
          std::fprintf(fp, "%.17e\n",
                       static_cast<double>(node_density(i, j, k, grid, options)));
        }
      }
    }

    PetscCall(VecRestoreArrayRead(seq, &u));
    std::fclose(fp);
  }

  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&seq));
  PetscCall(PetscPrintf(comm, "Wrote structured solution VTK: %s\n", path));
  return 0;
}

PetscErrorCode write_solid_mask_vtk(MPI_Comm comm, const char *path,
                                    const Grid &grid,
                                    const DensityOptions &options,
                                    PetscInt max_cells) {
  PetscMPIInt rank = 0;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (rank == 0) {
    PetscInt count = 0;
    for (PetscInt k = 0; k < grid.nz - 1; ++k) {
      for (PetscInt j = 0; j < grid.ny - 1; ++j) {
        for (PetscInt i = 0; i < grid.nx - 1; ++i) {
          if (cell_density(i, j, k, grid, options) >= options.mask_threshold) {
            ++count;
          }
        }
      }
    }

    PetscCheck(count <= max_cells, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
               "Solid VTK has %lld cells, above -max_vtk_cells=%lld",
               static_cast<long long>(count), static_cast<long long>(max_cells));

    FILE *fp = std::fopen(path, "w");
    PetscCheck(fp != nullptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Cannot open solid VTK file: %s", path);

    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "C++ PETSc thresholded solid control-arm mask\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    std::fprintf(fp, "POINTS %lld double\n", static_cast<long long>(8 * count));

    const PetscReal DL = domain_length(grid);
    const PetscReal DW = domain_width(grid);
    const PetscReal DH = domain_height(grid);
    for (PetscInt k = 0; k < grid.nz - 1; ++k) {
      for (PetscInt j = 0; j < grid.ny - 1; ++j) {
        for (PetscInt i = 0; i < grid.nx - 1; ++i) {
          if (cell_density(i, j, k, grid, options) >= options.mask_threshold) {
            const PetscReal x0 = DL * static_cast<PetscReal>(i) /
                                 static_cast<PetscReal>(grid.nx - 1);
            const PetscReal x1 = DL * static_cast<PetscReal>(i + 1) /
                                 static_cast<PetscReal>(grid.nx - 1);
            const PetscReal y0 = DW * static_cast<PetscReal>(j) /
                                 static_cast<PetscReal>(grid.ny - 1);
            const PetscReal y1 = DW * static_cast<PetscReal>(j + 1) /
                                 static_cast<PetscReal>(grid.ny - 1);
            const PetscReal z0 = DH * static_cast<PetscReal>(k) /
                                 static_cast<PetscReal>(grid.nz - 1);
            const PetscReal z1 = DH * static_cast<PetscReal>(k + 1) /
                                 static_cast<PetscReal>(grid.nz - 1);
            std::fprintf(fp, "%.17e %.17e %.17e\n", static_cast<double>(x0),
                         static_cast<double>(y0), static_cast<double>(z0));
            std::fprintf(fp, "%.17e %.17e %.17e\n", static_cast<double>(x1),
                         static_cast<double>(y0), static_cast<double>(z0));
            std::fprintf(fp, "%.17e %.17e %.17e\n", static_cast<double>(x1),
                         static_cast<double>(y1), static_cast<double>(z0));
            std::fprintf(fp, "%.17e %.17e %.17e\n", static_cast<double>(x0),
                         static_cast<double>(y1), static_cast<double>(z0));
            std::fprintf(fp, "%.17e %.17e %.17e\n", static_cast<double>(x0),
                         static_cast<double>(y0), static_cast<double>(z1));
            std::fprintf(fp, "%.17e %.17e %.17e\n", static_cast<double>(x1),
                         static_cast<double>(y0), static_cast<double>(z1));
            std::fprintf(fp, "%.17e %.17e %.17e\n", static_cast<double>(x1),
                         static_cast<double>(y1), static_cast<double>(z1));
            std::fprintf(fp, "%.17e %.17e %.17e\n", static_cast<double>(x0),
                         static_cast<double>(y1), static_cast<double>(z1));
          }
        }
      }
    }

    std::fprintf(fp, "CELLS %lld %lld\n", static_cast<long long>(count),
                 static_cast<long long>(9 * count));
    for (PetscInt cell = 0; cell < count; ++cell) {
      const PetscInt base = 8 * cell;
      std::fprintf(fp, "8 %lld %lld %lld %lld %lld %lld %lld %lld\n",
                   static_cast<long long>(base + 0),
                   static_cast<long long>(base + 1),
                   static_cast<long long>(base + 2),
                   static_cast<long long>(base + 3),
                   static_cast<long long>(base + 4),
                   static_cast<long long>(base + 5),
                   static_cast<long long>(base + 6),
                   static_cast<long long>(base + 7));
    }

    std::fprintf(fp, "CELL_TYPES %lld\n", static_cast<long long>(count));
    for (PetscInt cell = 0; cell < count; ++cell) {
      std::fprintf(fp, "12\n");
    }

    std::fprintf(fp, "CELL_DATA %lld\n", static_cast<long long>(count));
    std::fprintf(fp, "SCALARS density double 1\n");
    std::fprintf(fp, "LOOKUP_TABLE default\n");
    for (PetscInt cell = 0; cell < count; ++cell) {
      std::fprintf(fp, "1.0\n");
    }

    std::fclose(fp);
  }

  PetscCall(PetscPrintf(comm, "Wrote solid-mask VTK: %s\n", path));
  return 0;
}

} // namespace control_arm
