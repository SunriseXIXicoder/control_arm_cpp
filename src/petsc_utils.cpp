#include "control_arm/petsc_utils.hpp"

namespace control_arm {
namespace {

void expand_axis(PetscReal *min_value, PetscReal *max_value) {
  if (*min_value < *max_value) return;
  const PetscReal scale = PetscMax(1.0, PetscAbsReal(*min_value));
  const PetscReal delta = 0.05 * scale;
  *min_value -= delta;
  *max_value += delta;
}

PetscReal map_linear(PetscReal value, PetscReal min_value, PetscReal max_value,
                     PetscReal out_min, PetscReal out_max) {
  return out_min + (value - min_value) * (out_max - out_min) /
                       (max_value - min_value);
}

} // namespace

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
  PetscCall(PetscViewerASCIIPrintf(viewer,
                                   "domain_length_m=%.12e\ndomain_width_m=%.12e\ndomain_height_m=%.12e\n",
                                   static_cast<double>(domain_length(grid)),
                                   static_cast<double>(domain_width(grid)),
                                   static_cast<double>(domain_height(grid))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "control_arm_mask=%d\n",
                                   static_cast<int>(options.use_control_arm_mask)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "void_density=%.12e\n",
                                   static_cast<double>(options.void_density)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "young_modulus=%.12e\n",
                                   static_cast<double>(options.young_modulus)));
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
  PetscCall(PetscViewerASCIIPrintf(viewer,
                                   "domain_length_m=%.12e\ndomain_width_m=%.12e\ndomain_height_m=%.12e\n",
                                   static_cast<double>(domain_length(grid)),
                                   static_cast<double>(domain_width(grid)),
                                   static_cast<double>(domain_height(grid))));
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

PetscErrorCode write_objective_volume_history(
    const char *output_prefix,
    const std::vector<ObjectiveVolumePoint> &points,
    PetscReal volume_target) {
  if (points.empty()) return 0;

  char csv_path[PETSC_MAX_PATH_LEN];
  char svg_path[PETSC_MAX_PATH_LEN];
  PetscViewer viewer = nullptr;
  PetscInt iter_min = points.front().iter;
  PetscInt iter_max = points.front().iter;
  PetscReal obj_min = points.front().objective;
  PetscReal obj_max = points.front().objective;
  PetscReal vol_min = points.front().volume;
  PetscReal vol_max = points.front().volume;

  for (std::size_t p = 0; p < points.size(); ++p) {
    iter_min = PetscMin(iter_min, points[p].iter);
    iter_max = PetscMax(iter_max, points[p].iter);
    obj_min = PetscMin(obj_min, points[p].objective);
    obj_max = PetscMax(obj_max, points[p].objective);
    vol_min = PetscMin(vol_min, points[p].volume);
    vol_max = PetscMax(vol_max, points[p].volume);
  }
  if (volume_target >= 0.0) {
    vol_min = PetscMin(vol_min, volume_target);
    vol_max = PetscMax(vol_max, volume_target);
  }
  if (iter_min == iter_max) {
    --iter_min;
    ++iter_max;
  }
  expand_axis(&obj_min, &obj_max);
  expand_axis(&vol_min, &vol_max);

  PetscCall(PetscSNPrintf(csv_path, sizeof(csv_path), "%s_objective_volume.csv",
                          output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, csv_path, &viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,
                                   "iter,objective,volume,volume_target\n"));
  for (std::size_t p = 0; p < points.size(); ++p) {
    PetscCall(PetscViewerASCIIPrintf(
        viewer, "%lld,%.12e,%.12e,%.12e\n",
        static_cast<long long>(points[p].iter),
        static_cast<double>(points[p].objective),
        static_cast<double>(points[p].volume),
        static_cast<double>(volume_target)));
  }
  PetscCall(PetscViewerDestroy(&viewer));

  const PetscReal width = 960.0;
  const PetscReal height = 560.0;
  const PetscReal left = 88.0;
  const PetscReal right = 92.0;
  const PetscReal top = 54.0;
  const PetscReal bottom = 76.0;
  const PetscReal plot_w = width - left - right;
  const PetscReal plot_h = height - top - bottom;
  const PetscReal x0 = left;
  const PetscReal y0 = top;
  const PetscReal x1 = left + plot_w;
  const PetscReal y1 = top + plot_h;

  PetscCall(PetscSNPrintf(svg_path, sizeof(svg_path), "%s_objective_volume.svg",
                          output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, svg_path, &viewer));
  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">\n",
      static_cast<double>(width), static_cast<double>(height),
      static_cast<double>(width), static_cast<double>(height)));
  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<rect x=\"0\" y=\"0\" width=\"100%%\" height=\"100%%\" fill=\"#ffffff\"/>\n"));
  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<text x=\"%.1f\" y=\"30\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"20\" fill=\"#111827\">Objective and Volume History</text>\n",
      static_cast<double>(width * 0.5)));
  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/>\n",
      static_cast<double>(x0), static_cast<double>(y0),
      static_cast<double>(plot_w), static_cast<double>(plot_h)));

  for (PetscInt tick = 0; tick <= 5; ++tick) {
    const PetscReal a = static_cast<PetscReal>(tick) / 5.0;
    const PetscReal y = y1 - a * plot_h;
    const PetscReal obj = obj_min + a * (obj_max - obj_min);
    const PetscReal vol = vol_min + a * (vol_max - vol_min);
    PetscCall(PetscViewerASCIIPrintf(
        viewer,
        "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#e2e8f0\"/>\n",
        static_cast<double>(x0), static_cast<double>(y),
        static_cast<double>(x1), static_cast<double>(y)));
    PetscCall(PetscViewerASCIIPrintf(
        viewer,
        "<text x=\"%.1f\" y=\"%.1f\" text-anchor=\"end\" font-family=\"Arial\" font-size=\"12\" fill=\"#1f2937\">%.3e</text>\n",
        static_cast<double>(x0 - 10.0), static_cast<double>(y + 4.0),
        static_cast<double>(obj)));
    PetscCall(PetscViewerASCIIPrintf(
        viewer,
        "<text x=\"%.1f\" y=\"%.1f\" text-anchor=\"start\" font-family=\"Arial\" font-size=\"12\" fill=\"#065f46\">%.3f</text>\n",
        static_cast<double>(x1 + 10.0), static_cast<double>(y + 4.0),
        static_cast<double>(vol)));
  }

  for (PetscInt tick = 0; tick <= 5; ++tick) {
    const PetscReal a = static_cast<PetscReal>(tick) / 5.0;
    const PetscReal x = x0 + a * plot_w;
    const PetscReal iter = iter_min + a * (iter_max - iter_min);
    PetscCall(PetscViewerASCIIPrintf(
        viewer,
        "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#cbd5e1\"/>\n",
        static_cast<double>(x), static_cast<double>(y1),
        static_cast<double>(x), static_cast<double>(y1 + 5.0)));
    PetscCall(PetscViewerASCIIPrintf(
        viewer,
        "<text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"12\" fill=\"#1f2937\">%.0f</text>\n",
        static_cast<double>(x), static_cast<double>(y1 + 24.0),
        static_cast<double>(iter)));
  }

  if (volume_target >= 0.0) {
    const PetscReal ty = map_linear(volume_target, vol_min, vol_max, y1, y0);
    PetscCall(PetscViewerASCIIPrintf(
        viewer,
        "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#10b981\" stroke-width=\"1.4\" stroke-dasharray=\"6 5\"/>\n",
        static_cast<double>(x0), static_cast<double>(ty),
        static_cast<double>(x1), static_cast<double>(ty)));
  }

  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<polyline fill=\"none\" stroke=\"#2563eb\" stroke-width=\"2.4\" points=\""));
  for (std::size_t p = 0; p < points.size(); ++p) {
    const PetscReal x =
        map_linear(points[p].iter, iter_min, iter_max, x0, x1);
    const PetscReal y =
        map_linear(points[p].objective, obj_min, obj_max, y1, y0);
    PetscCall(PetscViewerASCIIPrintf(viewer, "%.2f,%.2f ",
                                     static_cast<double>(x),
                                     static_cast<double>(y)));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "\"/>\n"));

  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<polyline fill=\"none\" stroke=\"#059669\" stroke-width=\"2.4\" points=\""));
  for (std::size_t p = 0; p < points.size(); ++p) {
    const PetscReal x =
        map_linear(points[p].iter, iter_min, iter_max, x0, x1);
    const PetscReal y =
        map_linear(points[p].volume, vol_min, vol_max, y1, y0);
    PetscCall(PetscViewerASCIIPrintf(viewer, "%.2f,%.2f ",
                                     static_cast<double>(x),
                                     static_cast<double>(y)));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "\"/>\n"));

  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\" fill=\"#111827\">Iteration</text>\n",
      static_cast<double>((x0 + x1) * 0.5), static_cast<double>(height - 24.0)));
  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<text x=\"24\" y=\"%.1f\" text-anchor=\"middle\" transform=\"rotate(-90 24 %.1f)\" font-family=\"Arial\" font-size=\"14\" fill=\"#2563eb\">Objective</text>\n",
      static_cast<double>((y0 + y1) * 0.5), static_cast<double>((y0 + y1) * 0.5)));
  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<text x=\"936\" y=\"%.1f\" text-anchor=\"middle\" transform=\"rotate(90 936 %.1f)\" font-family=\"Arial\" font-size=\"14\" fill=\"#059669\">Volume</text>\n",
      static_cast<double>((y0 + y1) * 0.5), static_cast<double>((y0 + y1) * 0.5)));
  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<line x1=\"650\" y1=\"32\" x2=\"690\" y2=\"32\" stroke=\"#2563eb\" stroke-width=\"2.4\"/><text x=\"698\" y=\"36\" font-family=\"Arial\" font-size=\"13\" fill=\"#111827\">objective</text>\n"));
  PetscCall(PetscViewerASCIIPrintf(
      viewer,
      "<line x1=\"760\" y1=\"32\" x2=\"800\" y2=\"32\" stroke=\"#059669\" stroke-width=\"2.4\"/><text x=\"808\" y=\"36\" font-family=\"Arial\" font-size=\"13\" fill=\"#111827\">volume</text>\n"));
  if (volume_target >= 0.0) {
    PetscCall(PetscViewerASCIIPrintf(
        viewer,
        "<line x1=\"835\" y1=\"32\" x2=\"875\" y2=\"32\" stroke=\"#10b981\" stroke-width=\"1.4\" stroke-dasharray=\"6 5\"/><text x=\"883\" y=\"36\" font-family=\"Arial\" font-size=\"13\" fill=\"#111827\">target</text>\n"));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "</svg>\n"));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Objective-volume history: %s, plot: %s\n",
                        csv_path, svg_path));
  return 0;
}

} // namespace control_arm
