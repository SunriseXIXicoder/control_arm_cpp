#include "control_arm/density.hpp"
#include "control_arm/density_pipeline.hpp"
#include "control_arm/emsfem_ann.hpp"
#include "control_arm/geometry.hpp"
#include "control_arm/h8_matrix_free.hpp"
#include "control_arm/h8_optimizer.hpp"
#include "control_arm/low_order_optimizer.hpp"
#include "control_arm/petsc_utils.hpp"
#include "control_arm/vtk.hpp"

#include <petscksp.h>

using namespace control_arm;

static char help[] =
    "C++/PETSc control-arm topology-optimization migration driver.\n"
    "\n"
    "This first C++ stage provides a distributed geometry/density Vec path and\n"
    "a low_order PETSc solve for WSL/Slurm validation.  The true H8\n"
    "matrix-free FEM, ANN material model, and MMA optimizer are explicit next\n"
    "stage interfaces and are not silently approximated here.\n"
    "\n"
    "Main options:\n"
    "  -mode geometry|density|solve|optimize|ems_ann_postsolve|h8_initial_vtk|h8_postprocess|h8_full_vtk\n"
    "  -nx -ny -nz\n"
    "  -operator low_order|h8_matrix_free|emsfem_ann\n"
    "  -output_prefix <path>\n"
    "  -control_arm_mask true|false\n"
    "  -write_solid_vtk true|false\n"
    "  -write_structured_vtk true|false\n"
    "  -write_density_binary true|false\n";

namespace {

PetscErrorCode assemble_low_order_matrix(const Grid &grid,
                                         const DensityOptions &density_options,
                                         Mat *A) {
  PetscInt ndof = dof_count(grid);
  const PetscInt nxy = grid.nx * grid.ny;
  PetscInt local_rows = PETSC_DECIDE;
  PetscInt rstart = 0;
  PetscInt rend = 0;
  PetscInt *d_nnz = nullptr;
  PetscInt *o_nnz = nullptr;

  PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &local_rows, &ndof));
  {
    PetscInt scan_end = 0;
    PetscCallMPI(MPI_Scan(&local_rows, &scan_end, 1, MPIU_INT, MPI_SUM,
                          PETSC_COMM_WORLD));
    rstart = scan_end - local_rows;
    rend = scan_end;
  }

  PetscCall(PetscMalloc2(local_rows, &d_nnz, local_rows, &o_nnz));
  for (PetscInt row = rstart; row < rend; ++row) {
    const PetscInt local = row - rstart;
    const PetscInt c = row % 3;
    const PetscInt node = row / 3;
    const PetscInt k = node / nxy;
    const PetscInt rem = node - k * nxy;
    const PetscInt j = rem / grid.nx;
    const PetscInt i = rem - j * grid.nx;
    const PetscInt di[6] = {-1, 1, 0, 0, 0, 0};
    const PetscInt dj[6] = {0, 0, -1, 1, 0, 0};
    const PetscInt dk[6] = {0, 0, 0, 0, -1, 1};

    d_nnz[local] = 0;
    o_nnz[local] = 0;
    count_prealloc_column(row, rstart, rend, &d_nnz[local], &o_nnz[local]);
    if (i == 0) {
      continue;
    }

    for (PetscInt q = 0; q < 6; ++q) {
      const PetscInt ii = i + di[q];
      const PetscInt jj = j + dj[q];
      const PetscInt kk = k + dk[q];
      if (ii >= 0 && ii < grid.nx && jj >= 0 && jj < grid.ny && kk >= 0 &&
          kk < grid.nz && ii != 0) {
        const PetscInt col = global_dof(ii, jj, kk, c, grid);
        count_prealloc_column(col, rstart, rend, &d_nnz[local], &o_nnz[local]);
      }
    }
  }

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, local_rows, local_rows, ndof, ndof,
                         0, d_nnz, 0, o_nnz, A));
  PetscCall(PetscFree2(d_nnz, o_nnz));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

  const PetscReal hx = 1.0 / static_cast<PetscReal>(grid.nx - 1);
  const PetscReal hy = 1.0 / static_cast<PetscReal>(grid.ny - 1);
  const PetscReal hz = 1.0 / static_cast<PetscReal>(grid.nz - 1);
  const PetscReal kx = 1.0 / (hx * hx);
  const PetscReal ky = 1.0 / (hy * hy);
  const PetscReal kz = 1.0 / (hz * hz);

  for (PetscInt row = rstart; row < rend; ++row) {
    const PetscInt c = row % 3;
    const PetscInt node = row / 3;
    const PetscInt k = node / nxy;
    const PetscInt rem = node - k * nxy;
    const PetscInt j = rem / grid.nx;
    const PetscInt i = rem - j * grid.nx;
    PetscInt cols[7];
    PetscScalar vals[7];
    PetscInt ncols = 0;

    if (i == 0) {
      cols[0] = row;
      vals[0] = 1.0;
      PetscCall(MatSetValues(*A, 1, &row, 1, cols, vals, INSERT_VALUES));
    } else {
      const PetscReal rho0 = node_density(i, j, k, grid, density_options);
      PetscReal diag = 0.0;
      const PetscInt di[6] = {-1, 1, 0, 0, 0, 0};
      const PetscInt dj[6] = {0, 0, -1, 1, 0, 0};
      const PetscInt dk[6] = {0, 0, 0, 0, -1, 1};
      const PetscReal weight[6] = {kx, kx, ky, ky, kz, kz};

      for (PetscInt q = 0; q < 6; ++q) {
        const PetscInt ii = i + di[q];
        const PetscInt jj = j + dj[q];
        const PetscInt kk = k + dk[q];
        if (ii >= 0 && ii < grid.nx && jj >= 0 && jj < grid.ny && kk >= 0 &&
            kk < grid.nz) {
          const PetscReal rho1 = node_density(ii, jj, kk, grid, density_options);
          const PetscReal kij = weight[q] * edge_stiffness(rho0, rho1,
                                                           density_options);
          diag += kij;
          if (ii != 0) {
            cols[ncols] = global_dof(ii, jj, kk, c, grid);
            vals[ncols] = -kij;
            ++ncols;
          }
        }
      }

      cols[ncols] = row;
      vals[ncols] = diag;
      ++ncols;
      PetscCall(MatSetValues(*A, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  return 0;
}

PetscErrorCode fill_low_order_load(const Grid &grid, PetscReal load, Vec b) {
  PetscInt rstart = 0;
  PetscInt rend = 0;
  const PetscInt nxy = grid.nx * grid.ny;

  PetscCall(VecGetOwnershipRange(b, &rstart, &rend));
  PetscCall(VecSet(b, 0.0));
  for (PetscInt row = rstart; row < rend; ++row) {
    const PetscInt c = row % 3;
    const PetscInt node = row / 3;
    const PetscInt k = node / nxy;
    const PetscInt rem = node - k * nxy;
    const PetscInt j = rem / grid.nx;
    const PetscInt i = rem - j * grid.nx;
    if (i == grid.nx - 1 && c == 2) {
      const PetscScalar value = -load / static_cast<PetscReal>(grid.ny * grid.nz);
      PetscCall(VecSetValue(b, row, value, INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  return 0;
}

PetscErrorCode run_geometry_mode(const Grid &grid,
                                 const DensityOptions &density_options,
                                 const char *output_prefix,
                                 PetscBool write_structured_vtk,
                                 PetscBool write_solid_vtk,
                                 PetscBool write_density_vec,
                                 const char *structured_vtk_file,
                                 const char *solid_vtk_file,
                                 const char *density_vec_file,
                                 PetscInt max_vtk_cells) {
  PetscMPIInt ranks = 1;
  PetscReal vf = 0.0;
  PetscReal density_average = 0.0;
  PetscInt solid_cells = 0;
  PetscInt total_cells = 0;
  Vec rho = nullptr;

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCall(create_nodal_density_vec(PETSC_COMM_WORLD, grid, density_options, &rho));
  PetscCall(density_vec_average(rho, &density_average));
  PetscCall(compute_mask_volume_fraction(PETSC_COMM_WORLD, grid, density_options,
                                         &vf, &solid_cells, &total_cells));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Geometry mode: nx=%lld ny=%lld nz=%lld, nodal_density_avg=%.6g\n",
                        static_cast<long long>(grid.nx),
                        static_cast<long long>(grid.ny),
                        static_cast<long long>(grid.nz),
                        static_cast<double>(density_average)));

  if (write_density_vec) {
    PetscCall(write_density_binary(rho, density_vec_file));
  }
  if (write_structured_vtk) {
    PetscCall(write_structured_density_vtk(PETSC_COMM_WORLD, structured_vtk_file,
                                           grid, density_options));
  }
  if (write_solid_vtk) {
    PetscCall(write_solid_mask_vtk(PETSC_COMM_WORLD, solid_vtk_file,
                                   grid, density_options, max_vtk_cells));
  }

  PetscCall(write_geometry_report(output_prefix, grid, density_options, ranks, vf,
                                  solid_cells, total_cells));
  PetscCall(VecDestroy(&rho));
  return 0;
}

PetscErrorCode run_density_mode(const Grid &grid,
                                const DensityOptions &density_options,
                                const DensityPipelineOptions &pipeline_options,
                                const char *output_prefix,
                                PetscBool write_density_vtk,
                                PetscBool write_density_binary,
                                const char *density_vtk_file,
                                const char *density_binary_file) {
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Density mode: nx=%lld ny=%lld nz=%lld filter_radius=%g draft_dirs=%s\n",
                        static_cast<long long>(grid.nx),
                        static_cast<long long>(grid.ny),
                        static_cast<long long>(grid.nz),
                        static_cast<double>(pipeline_options.filter_radius),
                        pipeline_options.draft_dirs));
  PetscCall(run_density_pipeline(grid, density_options, pipeline_options,
                                 output_prefix, write_density_vtk,
                                 density_vtk_file, write_density_binary,
                                 density_binary_file));
  return 0;
}

PetscErrorCode run_optimize_mode(const Grid &grid,
                                 const DensityOptions &density_options,
                                 const OptimizerOptions &optimizer_options,
                                 const EmSfemAnnOptions &ems_options,
                                 const char *operator_type,
                                 const char *output_prefix,
                                 const char *final_vtk_file) {
  PetscBool is_low_order = PETSC_FALSE;
  PetscBool is_h8 = PETSC_FALSE;
  PetscBool is_ems = PETSC_FALSE;
  PetscCall(PetscStrcmp(operator_type, "low_order", &is_low_order));
  PetscCall(PetscStrcmp(operator_type, "h8_matrix_free", &is_h8));
  PetscCall(PetscStrcmp(operator_type, "emsfem_ann", &is_ems));
  PetscCheck(is_low_order || is_h8 || is_ems, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "-mode optimize currently supports -operator low_order, h8_matrix_free, or emsfem_ann");
  if (is_h8) {
    PetscCall(run_h8_optimizer(grid, density_options, optimizer_options,
                               output_prefix, final_vtk_file));
  } else if (is_ems) {
    PetscCall(run_emsfem_ann_optimizer(grid, density_options, optimizer_options,
                                       ems_options, output_prefix, final_vtk_file));
  } else {
    PetscCall(run_low_order_optimizer(grid, density_options, optimizer_options,
                                      output_prefix, final_vtk_file));
  }
  return 0;
}

PetscErrorCode run_solve_mode(const Grid &grid,
                              const DensityOptions &density_options,
                              const EmSfemAnnOptions &ems_options,
                              const char *operator_type,
                              const char *output_prefix,
                              PetscReal load,
                              PetscBool write_solution,
                              PetscBool write_structured_vtk,
                              PetscBool write_solid_vtk,
                              const char *solution_file,
                              const char *structured_vtk_file,
                              const char *solid_vtk_file,
                              PetscInt max_vtk_cells) {
  PetscMPIInt ranks = 1;
  Mat A = nullptr;
  Vec b = nullptr;
  Vec u = nullptr;
  KSP ksp = nullptr;
  PC pc = nullptr;
  PetscLogDouble t0 = 0.0;
  PetscLogDouble t1 = 0.0;
  PetscLogDouble t2 = 0.0;
  PetscInt its = 0;
  PetscReal rnorm = 0.0;
  PetscScalar compliance = 0.0;
  PetscBool is_h8_matrix_free = PETSC_FALSE;
  PetscBool is_emsfem_ann = PETSC_FALSE;

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCall(PetscStrcmp(operator_type, "h8_matrix_free", &is_h8_matrix_free));
  PetscCall(PetscStrcmp(operator_type, "emsfem_ann", &is_emsfem_ann));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "C++ PETSc solve: operator=%s nx=%lld ny=%lld nz=%lld dof=%lld ranks=%d\n",
                        operator_type,
                        static_cast<long long>(grid.nx),
                        static_cast<long long>(grid.ny),
                        static_cast<long long>(grid.nz),
                        static_cast<long long>(dof_count(grid)),
                        static_cast<int>(ranks)));
  PetscCall(PetscTime(&t0));
  if (is_h8_matrix_free) {
    PetscCall(create_h8_matrix_free_system(grid, density_options, load, &A, &u, &b));
  } else if (is_emsfem_ann) {
    PetscCall(create_emsfem_ann_system(grid, density_options, ems_options, load,
                                       &A, &u, &b));
  } else {
    PetscCall(assemble_low_order_matrix(grid, density_options, &A));
    PetscCall(MatCreateVecs(A, &u, &b));
    PetscCall(VecSet(u, 0.0));
    PetscCall(fill_low_order_load(grid, load, b));
  }
  PetscCall(PetscTime(&t1));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, (is_h8_matrix_free || is_emsfem_ann) ? PCJACOBI : PCGAMG));
  PetscCall(KSPSetTolerances(ksp, 1.0e-6, PETSC_DEFAULT, PETSC_DEFAULT, 500));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Assembly time: %.6e s\n",
                        static_cast<double>(t1 - t0)));
  PetscCall(PetscTime(&t1));
  PetscCall(KSPSolve(ksp, b, u));
  PetscCall(PetscTime(&t2));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPGetResidualNorm(ksp, &rnorm));
  PetscCall(VecDot(b, u, &compliance));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Solve time: %.6e s, iterations=%lld, residual=%g, compliance=%g\n",
                        static_cast<double>(t2 - t1),
                        static_cast<long long>(its),
                        static_cast<double>(rnorm),
                        static_cast<double>(PetscRealPart(compliance))));
  PetscCall(write_solve_report(output_prefix, grid, ranks, operator_type, t1 - t0,
                               t2 - t1, its, rnorm, PetscRealPart(compliance)));

  if (write_solution) {
    PetscViewer viewer = nullptr;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, solution_file, FILE_MODE_WRITE,
                                    &viewer));
    PetscCall(VecView(u, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Wrote PETSc solution Vec: %s\n",
                          solution_file));
  }
  if (write_structured_vtk) {
    if (is_h8_matrix_free) {
      PetscCall(write_h8_solution_vtk(A, u, structured_vtk_file, grid,
                                      density_options));
    } else if (is_emsfem_ann) {
      PetscCall(write_emsfem_ann_solution_vtk(A, u, structured_vtk_file, grid,
                                              density_options));
    } else {
      PetscCall(write_structured_solution_vtk(u, structured_vtk_file, grid,
                                              density_options));
    }
  }
  if (write_solid_vtk) {
    PetscCall(write_solid_mask_vtk(PETSC_COMM_WORLD, solid_vtk_file, grid,
                                   density_options, max_vtk_cells));
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, help);
  if (ierr) {
    return ierr;
  }
  PetscCall(PetscMemorySetGetMaximumUsage());

  Grid grid;
  DensityOptions density_options;
  char mode[64] = "geometry";
  char operator_type[64] = "low_order";
  char output_prefix[PETSC_MAX_PATH_LEN] = "result/control_arm_cpp";
  char structured_vtk_file[PETSC_MAX_PATH_LEN] = "";
  char solid_vtk_file[PETSC_MAX_PATH_LEN] = "";
  char density_vec_file[PETSC_MAX_PATH_LEN] = "";
  char solution_file[PETSC_MAX_PATH_LEN] = "";
  char density_file[PETSC_MAX_PATH_LEN] = "";
  char mask_file[PETSC_MAX_PATH_LEN] = "";
  char post_vtk_file[PETSC_MAX_PATH_LEN] = "";
  PetscBool has_structured_vtk_file = PETSC_FALSE;
  PetscBool has_solid_vtk_file = PETSC_FALSE;
  PetscBool has_density_vec_file = PETSC_FALSE;
  PetscBool has_solution_file = PETSC_FALSE;
  PetscBool has_density_file = PETSC_FALSE;
  PetscBool has_mask_file = PETSC_FALSE;
  PetscBool has_post_vtk_file = PETSC_FALSE;
  PetscBool write_structured_vtk = PETSC_FALSE;
  PetscBool write_solid_vtk = PETSC_FALSE;
  PetscBool write_density_binary_flag = PETSC_FALSE;
  PetscBool write_solution = PETSC_FALSE;
  PetscBool is_geometry_mode = PETSC_FALSE;
  PetscBool is_solve_mode = PETSC_FALSE;
  PetscBool is_density_mode = PETSC_FALSE;
  PetscBool is_optimize_mode = PETSC_FALSE;
  PetscBool is_ems_ann_postsolve_mode = PETSC_FALSE;
  PetscBool is_h8_initial_vtk_mode = PETSC_FALSE;
  PetscBool is_h8_postprocess_mode = PETSC_FALSE;
  PetscBool is_h8_full_vtk_mode = PETSC_FALSE;
  PetscBool is_low_order = PETSC_FALSE;
  PetscBool is_h8_matrix_free = PETSC_FALSE;
  PetscBool is_emsfem_ann = PETSC_FALSE;
  PetscReal load = 1.0;
  PetscInt max_vtk_cells = 300000;
  PetscInt post_stride = 5;
  PetscInt post_max_samples = 2000000;
  DensityPipelineOptions pipeline_options;
  PetscBool write_density_vtk = PETSC_FALSE;
  char density_vtk_file[PETSC_MAX_PATH_LEN] = "";
  PetscBool has_density_vtk_file = PETSC_FALSE;
  OptimizerOptions optimizer_options;
  EmSfemAnnOptions ems_options;
  char opt_vtk_file[PETSC_MAX_PATH_LEN] = "";
  PetscBool has_opt_vtk_file = PETSC_FALSE;
  PetscBool has_ems_update_method = PETSC_FALSE;
  PetscBool has_ems_update_alias = PETSC_FALSE;

  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-nx", &grid.nx, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-ny", &grid.ny, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-nz", &grid.nz, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-domain_height",
                                &grid.physical_height, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-mode", mode, sizeof(mode),
                                  nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-operator", operator_type,
                                  sizeof(operator_type), nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-output_prefix", output_prefix,
                                  sizeof(output_prefix), nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-control_arm_mask",
                                &density_options.use_control_arm_mask, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-void_density",
                                &density_options.void_density, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-young_modulus",
                                &density_options.young_modulus, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-penal", &density_options.penal,
                                nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-emin", &density_options.emin,
                                nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-mask_threshold",
                                &density_options.mask_threshold, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-ab_triangle_retract",
                                &density_options.ab_triangle_retract,
                                nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-load", &load, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-write_structured_vtk",
                                &write_structured_vtk, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-structured_vtk_file",
                                  structured_vtk_file, sizeof(structured_vtk_file),
                                  &has_structured_vtk_file));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-write_solid_vtk",
                                &write_solid_vtk, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-solid_vtk_file",
                                  solid_vtk_file, sizeof(solid_vtk_file),
                                  &has_solid_vtk_file));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-write_density_binary",
                                &write_density_binary_flag, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-density_binary_file",
                                  density_vec_file, sizeof(density_vec_file),
                                  &has_density_vec_file));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-write_solution",
                                &write_solution, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-solution_file", solution_file,
                                  sizeof(solution_file), &has_solution_file));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-density_file", density_file,
                                  sizeof(density_file), &has_density_file));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-mask_file", mask_file,
                                  sizeof(mask_file), &has_mask_file));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-post_vtk_file", post_vtk_file,
                                  sizeof(post_vtk_file), &has_post_vtk_file));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-post_stride", &post_stride,
                               nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-post_max_samples",
                               &post_max_samples, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-max_vtk_cells", &max_vtk_cells,
                               nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-filter_radius",
                                &pipeline_options.filter_radius, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-draft_radius",
                               &pipeline_options.draft_radius, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-draft_pnorm",
                                &pipeline_options.draft_pnorm, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-draft_beta",
                                &pipeline_options.draft_beta, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-draft_eta",
                                &pipeline_options.draft_eta, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-draft_dirs",
                                  pipeline_options.draft_dirs,
                                  sizeof(pipeline_options.draft_dirs), nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-draft_combine",
                                  pipeline_options.draft_combine,
                                  sizeof(pipeline_options.draft_combine), nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-write_density_vtk",
                                &write_density_vtk, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-density_vtk_file",
                                  density_vtk_file, sizeof(density_vtk_file),
                                  &has_density_vtk_file));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-opt_max_iter",
                               &optimizer_options.max_iter, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-volfrac",
                                &optimizer_options.volfrac, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_move",
                                &optimizer_options.move, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_move_min",
                                &optimizer_options.move_min, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_move_shrink",
                                &optimizer_options.move_shrink, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_move_growth",
                                &optimizer_options.move_growth, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_max_compliance_increase",
                                &optimizer_options.max_compliance_increase,
                                nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_stability_guard",
                                &optimizer_options.stability_guard, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr,
                                "-opt_projected_volume_correction",
                                &optimizer_options.projected_volume_correction,
                                nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_rho_min",
                                &optimizer_options.rho_min, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_heaviside_projection",
                                &optimizer_options.heaviside_projection,
                                nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_use_mma",
                                &optimizer_options.use_mma, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_matlab_z_projection",
                                &optimizer_options.matlab_z_projection,
                                nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_heaviside_eta",
                                &optimizer_options.heaviside_eta, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr,
                                "-opt_heaviside_beta_initial",
                                &optimizer_options.heaviside_beta_initial,
                                nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_heaviside_beta_max",
                                &optimizer_options.heaviside_beta_max,
                                nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-opt_heaviside_beta_interval",
                               &optimizer_options.heaviside_beta_interval,
                               nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_z_draft_closure",
                                &optimizer_options.z_draft_closure, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_z_draft_eta",
                                &optimizer_options.z_draft_eta, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_draft_closure",
                                &optimizer_options.z_draft_closure, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_draft_eta",
                                &optimizer_options.z_draft_eta, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_draft_beta",
                                &optimizer_options.draft_beta, nullptr));
  PetscBool has_opt_draft_axis = PETSC_FALSE;
  PetscBool has_opt_draft_axes = PETSC_FALSE;
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-opt_draft_axis",
                                  optimizer_options.draft_axis,
                                  sizeof(optimizer_options.draft_axis),
                                  &has_opt_draft_axis));
  if (has_opt_draft_axis) {
    PetscCall(PetscStrncpy(optimizer_options.draft_axes,
                           optimizer_options.draft_axis,
                           sizeof(optimizer_options.draft_axes)));
  }
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-opt_draft_axes",
                                  optimizer_options.draft_axes,
                                  sizeof(optimizer_options.draft_axes),
                                  &has_opt_draft_axes));
  if (has_opt_draft_axes) {
    PetscCall(PetscStrncpy(optimizer_options.draft_axis,
                           optimizer_options.draft_axes,
                           sizeof(optimizer_options.draft_axis)));
  }
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_load",
                                 &optimizer_options.load, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_ksp_rtol",
                                &optimizer_options.ksp_rtol, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-opt_ksp_max_it",
                               &optimizer_options.ksp_max_it, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-opt_vtk_interval",
                               &optimizer_options.vtk_interval, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-opt_vtk_max_points",
                               &optimizer_options.vtk_max_points, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_write_final_vtk",
                                &optimizer_options.write_final_vtk, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-benchmark_case",
                                  optimizer_options.benchmark_case,
                                  sizeof(optimizer_options.benchmark_case),
                                  nullptr));
  PetscCall(PetscStrncpy(density_options.benchmark_case,
                         optimizer_options.benchmark_case,
                         sizeof(density_options.benchmark_case)));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-opt_filter_radius",
                                &optimizer_options.filter_radius, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_write_checkpoint",
                                &optimizer_options.write_checkpoint, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-opt_checkpoint_interval",
                               &optimizer_options.checkpoint_interval, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-opt_checkpoint_prefix",
                                  optimizer_options.checkpoint_prefix,
                                  sizeof(optimizer_options.checkpoint_prefix), nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-opt_vtk_file",
                                  opt_vtk_file, sizeof(opt_vtk_file),
                                  &has_opt_vtk_file));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-ems_ann_dir",
                                  ems_options.ann_dir, sizeof(ems_options.ann_dir),
                                  nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-ems_sub_n",
                               &ems_options.sub_n, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-ems_cache_element_matrices",
                                &ems_options.cache_element_matrices, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-ems_cache_gib_limit",
                                &ems_options.cache_gib_limit, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-control_arm_bc",
                                &ems_options.control_arm_bc, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-load_case",
                               &ems_options.load_case, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-include_spring_load",
                                &ems_options.include_spring_load, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-benchmark_case",
                                  ems_options.benchmark_case,
                                  sizeof(ems_options.benchmark_case),
                                  nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr,
                                  "-ems_ann_filter_mode",
                                  ems_options.filter_mode,
                                  sizeof(ems_options.filter_mode),
                                  nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr,
                                  "-ems_filter_mode",
                                  ems_options.filter_mode,
                                  sizeof(ems_options.filter_mode),
                                  nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr,
                                  "-ems_ann_update_method",
                                  ems_options.update_method,
                                  sizeof(ems_options.update_method),
                                  &has_ems_update_method));
  PetscCall(PetscOptionsGetString(nullptr, nullptr,
                                  "-ems_update_method",
                                  ems_options.update_method,
                                  sizeof(ems_options.update_method),
                                  &has_ems_update_alias));
  if (!has_ems_update_method && !has_ems_update_alias &&
      optimizer_options.use_mma) {
    // 兼容旧的通用开关：ANN 未显式指定更新器时，-opt_use_mma true 切到 MMA。
    PetscCall(PetscStrncpy(ems_options.update_method, "mma",
                           sizeof(ems_options.update_method)));
  }
  PetscCall(PetscOptionsGetString(nullptr, nullptr,
                                  "-ems_ann_draft_closure_mode",
                                  ems_options.draft_closure_mode,
                                  sizeof(ems_options.draft_closure_mode),
                                  nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr,
                                  "-ems_draft_closure_mode",
                                  ems_options.draft_closure_mode,
                                  sizeof(ems_options.draft_closure_mode),
                                  nullptr));

  PetscCall(PetscStrcmp(mode, "geometry", &is_geometry_mode));
  PetscCall(PetscStrcmp(mode, "solve", &is_solve_mode));
  PetscCall(PetscStrcmp(mode, "density", &is_density_mode));
  PetscCall(PetscStrcmp(mode, "optimize", &is_optimize_mode));
  PetscCall(PetscStrcmp(mode, "ems_ann_postsolve",
                        &is_ems_ann_postsolve_mode));
  PetscCall(PetscStrcmp(mode, "h8_initial_vtk", &is_h8_initial_vtk_mode));
  PetscCall(PetscStrcmp(mode, "h8_postprocess", &is_h8_postprocess_mode));
  PetscCall(PetscStrcmp(mode, "h8_full_vtk", &is_h8_full_vtk_mode));
  PetscCall(PetscStrcmp(operator_type, "low_order", &is_low_order));
  PetscCall(PetscStrcmp(operator_type, "h8_matrix_free", &is_h8_matrix_free));
  PetscCall(PetscStrcmp(operator_type, "emsfem_ann", &is_emsfem_ann));

  PetscCheck(is_geometry_mode || is_solve_mode || is_density_mode ||
                 is_optimize_mode || is_ems_ann_postsolve_mode ||
                 is_h8_initial_vtk_mode ||
                 is_h8_postprocess_mode ||
                 is_h8_full_vtk_mode,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "-mode must be geometry, density, optimize, solve, ems_ann_postsolve, h8_initial_vtk, h8_postprocess, or h8_full_vtk");
  PetscCheck(is_low_order || is_h8_matrix_free || is_emsfem_ann,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "-operator must be low_order, h8_matrix_free, or emsfem_ann");
  PetscCheck(ems_options.sub_n >= 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "-ems_sub_n must be at least 2");
  PetscCheck(!has_density_file || is_ems_ann_postsolve_mode ||
                 is_h8_postprocess_mode || is_h8_full_vtk_mode,
             PETSC_COMM_WORLD,
             PETSC_ERR_SUP,
             "-density_file is currently accepted only by -mode ems_ann_postsolve, h8_postprocess, or h8_full_vtk");
  PetscCheck(grid.nx >= 2 && grid.ny >= 2 && grid.nz >= 2, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_OUTOFRANGE, "nx, ny, and nz must be at least 2");
  PetscCheck(grid.physical_height > 0.0, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_OUTOFRANGE, "-domain_height must be positive");
  PetscCheck(3.0 * static_cast<double>(grid.nx) * static_cast<double>(grid.ny) *
                     static_cast<double>(grid.nz) <=
                 static_cast<double>(PETSC_MAX_INT),
             PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "Requested grid exceeds this PETSc build's PetscInt range");

  if (!has_structured_vtk_file) {
    PetscCall(PetscSNPrintf(structured_vtk_file, sizeof(structured_vtk_file),
                            "%s.vtk", output_prefix));
  }
  if (!has_solid_vtk_file) {
    PetscCall(PetscSNPrintf(solid_vtk_file, sizeof(solid_vtk_file),
                            "%s_solid.vtk", output_prefix));
  }
  if (!has_density_vec_file) {
    PetscCall(PetscSNPrintf(density_vec_file, sizeof(density_vec_file),
                            "%s_rho.petscbin", output_prefix));
  }
  if (!has_solution_file) {
    PetscCall(PetscSNPrintf(solution_file, sizeof(solution_file),
                            "%s_u.petscbin", output_prefix));
  }
  if (!has_density_vtk_file) {
    PetscCall(PetscSNPrintf(density_vtk_file, sizeof(density_vtk_file),
                            "%s_density.vtk", output_prefix));
  }
  if (!has_post_vtk_file) {
    PetscCall(PetscSNPrintf(post_vtk_file, sizeof(post_vtk_file),
                            "%s_downsample.vtk", output_prefix));
  }
  if (!has_opt_vtk_file) {
    PetscCall(PetscSNPrintf(opt_vtk_file, sizeof(opt_vtk_file),
                            "%s_final.vtk", output_prefix));
  }
  if (optimizer_options.checkpoint_prefix[0] == '\0') {
    PetscCall(PetscSNPrintf(optimizer_options.checkpoint_prefix,
                            sizeof(optimizer_options.checkpoint_prefix),
                            "%s_checkpoint", output_prefix));
  }

  if (is_geometry_mode) {
    PetscCall(run_geometry_mode(grid, density_options, output_prefix,
                                write_structured_vtk, write_solid_vtk,
                                write_density_binary_flag, structured_vtk_file,
                                solid_vtk_file, density_vec_file, max_vtk_cells));
  } else if (is_density_mode) {
    PetscCall(run_density_mode(grid, density_options, pipeline_options,
                               output_prefix, write_density_vtk,
                               write_density_binary_flag, density_vtk_file,
                               density_vec_file));
  } else if (is_optimize_mode) {
    PetscCall(run_optimize_mode(grid, density_options, optimizer_options,
                                ems_options,
                                operator_type, output_prefix, opt_vtk_file));
  } else if (is_ems_ann_postsolve_mode) {
    PetscCheck(is_emsfem_ann, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "-mode ems_ann_postsolve requires -operator emsfem_ann");
    PetscCheck(has_density_file, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
               "-mode ems_ann_postsolve requires -density_file");
    PetscCall(run_emsfem_ann_postsolve(grid, density_options,
                                       optimizer_options, ems_options,
                                       density_file,
                                       has_mask_file ? mask_file : "",
                                       output_prefix));
  } else if (is_h8_initial_vtk_mode) {
    PetscCall(run_h8_initial_vtk(grid, density_options, optimizer_options,
                                 post_vtk_file, post_stride,
                                 post_max_samples));
  } else if (is_h8_postprocess_mode) {
    PetscCall(run_h8_density_postprocess(grid, optimizer_options, density_file,
                                         has_mask_file ? mask_file : "",
                                         post_vtk_file, post_stride,
                                         post_max_samples));
  } else if (is_h8_full_vtk_mode) {
    PetscCall(run_h8_full_vtk_postprocess(grid, density_options,
                                          optimizer_options, density_file,
                                          has_mask_file ? mask_file : "",
                                          post_vtk_file));
  } else {
    PetscCall(run_solve_mode(grid, density_options, ems_options, operator_type,
                             output_prefix, load, write_solution, write_structured_vtk,
                             write_solid_vtk, solution_file, structured_vtk_file,
                             solid_vtk_file, max_vtk_cells));
  }

  PetscCall(PetscFinalize());
  return 0;
}
