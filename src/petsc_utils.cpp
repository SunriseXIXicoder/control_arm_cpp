#include "control_arm/petsc_utils.hpp"

namespace control_arm {

void count_prealloc_column(PetscInt col, PetscInt cstart, PetscInt cend,
                           PetscInt *dnz, PetscInt *onz) {
  if (col >= cstart && col < cend) {
    ++(*dnz);
  } else {
    ++(*onz);
  }
}

PetscErrorCode write_geometry_report(const char *output_prefix,
                                     const Grid &grid,
                                     const DensityOptions &options,
                                     PetscMPIInt ranks,
                                     PetscReal volume_fraction,
                                     PetscInt solid_cells,
                                     PetscInt total_cells) {
  PetscViewer viewer = nullptr;
  char report_file[PETSC_MAX_PATH_LEN];
  PetscLogDouble mem_current = 0.0;
  PetscLogDouble mem_peak = 0.0;
  PetscLogDouble mem_current_max = 0.0;
  PetscLogDouble mem_peak_max = 0.0;

  PetscCall(PetscMemoryGetCurrentUsage(&mem_current));
  PetscCall(PetscMemoryGetMaximumUsage(&mem_peak));
  PetscCallMPI(MPI_Reduce(&mem_current, &mem_current_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                          PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&mem_peak, &mem_peak_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                          PETSC_COMM_WORLD));

  PetscCall(PetscSNPrintf(report_file, sizeof(report_file), "%s_geometry_summary.txt",
                          output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, report_file, &viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "mode=geometry\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "nx=%lld\nny=%lld\nnz=%lld\nranks=%d\n",
                                   static_cast<long long>(grid.nx),
                                   static_cast<long long>(grid.ny),
                                   static_cast<long long>(grid.nz),
                                   static_cast<int>(ranks)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "control_arm_mask=%d\n",
                                   static_cast<int>(options.use_control_arm_mask)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "void_density=%.12e\n",
                                   static_cast<double>(options.void_density)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "mask_threshold=%.12e\n",
                                   static_cast<double>(options.mask_threshold)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "solid_cells=%lld\n",
                                   static_cast<long long>(solid_cells)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "total_cells=%lld\n",
                                   static_cast<long long>(total_cells)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "volume_fraction=%.12e\n",
                                   static_cast<double>(volume_fraction)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_current_memory_bytes=%.0f\n",
                                   static_cast<double>(mem_current_max)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_peak_memory_bytes=%.0f\n",
                                   static_cast<double>(mem_peak_max)));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Geometry report: %s, volume_fraction=%.6g, solid_cells=%lld/%lld\n",
                        report_file, static_cast<double>(volume_fraction),
                        static_cast<long long>(solid_cells),
                        static_cast<long long>(total_cells)));
  return 0;
}

PetscErrorCode write_solve_report(const char *output_prefix,
                                  const Grid &grid,
                                  PetscMPIInt ranks,
                                  const char *operator_type,
                                  PetscLogDouble assembly_time,
                                  PetscLogDouble solve_time,
                                  PetscInt iterations,
                                  PetscReal residual_norm,
                                  PetscReal compliance) {
  PetscViewer viewer = nullptr;
  char report_file[PETSC_MAX_PATH_LEN];
  PetscLogDouble mem_current = 0.0;
  PetscLogDouble mem_peak = 0.0;
  PetscLogDouble mem_current_max = 0.0;
  PetscLogDouble mem_peak_max = 0.0;

  PetscCall(PetscMemoryGetCurrentUsage(&mem_current));
  PetscCall(PetscMemoryGetMaximumUsage(&mem_peak));
  PetscCallMPI(MPI_Reduce(&mem_current, &mem_current_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                          PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&mem_peak, &mem_peak_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                          PETSC_COMM_WORLD));

  PetscCall(PetscSNPrintf(report_file, sizeof(report_file), "%s_solve_summary.txt",
                          output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, report_file, &viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "mode=solve\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "operator=%s\n", operator_type));
  PetscCall(PetscViewerASCIIPrintf(viewer, "nx=%lld\nny=%lld\nnz=%lld\n",
                                   static_cast<long long>(grid.nx),
                                   static_cast<long long>(grid.ny),
                                   static_cast<long long>(grid.nz)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "dof=%lld\nranks=%d\n",
                                   static_cast<long long>(dof_count(grid)),
                                   static_cast<int>(ranks)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "assembly_time_sec=%.12e\n",
                                   static_cast<double>(assembly_time)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "solve_time_sec=%.12e\n",
                                   static_cast<double>(solve_time)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ksp_iterations=%lld\n",
                                   static_cast<long long>(iterations)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ksp_residual=%.12e\n",
                                   static_cast<double>(residual_norm)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "compliance=%.12e\n",
                                   static_cast<double>(compliance)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_current_memory_bytes=%.0f\n",
                                   static_cast<double>(mem_current_max)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_peak_memory_bytes=%.0f\n",
                                   static_cast<double>(mem_peak_max)));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Solve report: %s, max peak memory %.3f GiB/rank\n",
                        report_file,
                        static_cast<double>(mem_peak_max / 1073741824.0)));
  return 0;
}

} // namespace control_arm
