#include "control_arm/h8_optimizer.hpp"

#include "control_arm/draft_closure.hpp"
#include "control_arm/petsc_utils.hpp"

#include <petscdmda.h>
#include <petscviewer.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace control_arm {
namespace {

struct H8OptContext {
  DM uda = nullptr;
  DM eda = nullptr;
  Vec local_u = nullptr;
  Vec local_rho = nullptr;
  Vec rho = nullptr;
  Grid grid;
  DensityOptions density_options;
  PetscReal ke[24 * 24]{};
};

struct H8BlockJacobiContext {
  DM uda = nullptr;
  DM eda = nullptr;
  DM bda = nullptr;
  Vec rho = nullptr;
  Vec local_rho = nullptr;
  Vec inv_blocks = nullptr;
  Grid grid;
  DensityOptions density_options;
  PetscReal ke[24 * 24]{};
};

struct H8MatlabMMAState {
  PetscInt n = 0;
  PetscReal c_scale = 0.0;
  std::vector<PetscReal> xold1;
  std::vector<PetscReal> xold2;
  std::vector<PetscReal> L;
  std::vector<PetscReal> U;
};

struct H8VectorStats {
  PetscReal min = 0.0;
  PetscReal max = 0.0;
  PetscReal sum = 0.0;
  PetscReal l1 = 0.0;
};

struct H8MatlabMMADiagnostics {
  PetscReal f0val = 0.0;
  PetscReal fval = 0.0;
  PetscReal c_scale = 0.0;
  PetscReal move = 0.0;
  PetscReal beta1 = 0.0;
  PetscReal beta2 = 0.0;
  PetscReal x_change = 0.0;
  H8VectorStats x;
  H8VectorStats xmma;
  H8VectorStats df0dx;
  H8VectorStats dfdx;
  std::vector<PetscReal> x_values;
  std::vector<PetscReal> xmma_values;
  std::vector<PetscReal> df0dx_values;
  std::vector<PetscReal> dfdx_values;
};

struct ElementGrid {
  PetscInt ex = 0;
  PetscInt ey = 0;
  PetscInt ez = 0;
};

PetscErrorCode log_h8_memory_stage(const char *stage) {
  PetscLogDouble current_bytes = 0.0;
  PetscLogDouble peak_bytes = 0.0;
  PetscLogDouble max_current_bytes = 0.0;
  PetscLogDouble max_peak_bytes = 0.0;

  PetscCall(PetscMemoryGetCurrentUsage(&current_bytes));
  PetscCall(PetscMemoryGetMaximumUsage(&peak_bytes));
  PetscCallMPI(MPI_Allreduce(&current_bytes, &max_current_bytes, 1, MPI_DOUBLE,
                             MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&peak_bytes, &max_peak_bytes, 1, MPI_DOUBLE,
                             MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 memory [%s]: current_max_rank=%.3f MiB peak_max_rank=%.3f MiB\n",
                        stage,
                        static_cast<double>(max_current_bytes /
                                            (1024.0 * 1024.0)),
                        static_cast<double>(max_peak_bytes /
                                            (1024.0 * 1024.0))));
  return 0;
}

PetscInt h8_local_node(PetscInt lx, PetscInt ly, PetscInt lz) {
  if (lz == 0) {
    if (ly == 0) return lx == 0 ? 0 : 1;
    return lx == 0 ? 3 : 2;
  }
  if (ly == 0) return lx == 0 ? 4 : 5;
  return lx == 0 ? 7 : 6;
}

void h8_node_coords(PetscInt ex, PetscInt ey, PetscInt ez,
                    PetscInt node, PetscInt *i, PetscInt *j, PetscInt *k) {
  static const PetscInt dx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
  static const PetscInt dy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  static const PetscInt dz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  *i = ex + dx[node];
  *j = ey + dy[node];
  *k = ez + dz[node];
}

void h8_node_physical(PetscInt i, PetscInt j, PetscInt k, const Grid &grid,
                      PetscReal *x, PetscReal *y, PetscReal *z) {
  *x = grid.nx > 1 ? static_cast<PetscReal>(i) /
                         static_cast<PetscReal>(grid.nx - 1) *
                         domain_length(grid)
                   : 0.0;
  *y = grid.ny > 1 ? static_cast<PetscReal>(j) /
                         static_cast<PetscReal>(grid.ny - 1) *
                         domain_width(grid)
                   : 0.0;
  *z = grid.nz > 1 ? static_cast<PetscReal>(k) /
                         static_cast<PetscReal>(grid.nz - 1) *
                         domain_height(grid)
                   : 0.0;
}

PetscBool h8_is_c_support_node(PetscInt i, PetscInt j, PetscInt k,
                               const Grid &grid) {
  PetscReal X = 0.0, Y = 0.0, Z = 0.0;
  h8_node_physical(i, j, k, grid, &X, &Y, &Z);
  const PetscReal DL = domain_length(grid);
  const PetscReal DW = domain_width(grid);
  const PetscReal DH = domain_height(grid);
  const PetscReal C[3] = {0.78 * DL, 0.50 * DW, 0.50 * DH};
  const PetscReal r = 0.10 * PetscMin(DL, DW);
  const PetscReal hmax = PetscMax(PetscMax(DL / PetscMax(1, grid.nx - 1),
                                           DW / PetscMax(1, grid.ny - 1)),
                                  DH / PetscMax(1, grid.nz - 1));
  const PetscReal tol = 0.75 * hmax;
  const PetscReal axial = PetscMax(0.15 * PetscMin(DL, DW), 2.0 * hmax);
  const PetscReal rr =
      PetscSqrtReal((X - C[0]) * (X - C[0]) + (Y - C[1]) * (Y - C[1]));
  return (PetscAbsReal(rr - r) <= tol && PetscAbsReal(Z - C[2]) <= axial)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool h8_is_fixed_node(PetscInt i, PetscInt j, PetscInt k,
                            const Grid &grid,
                            const DensityOptions &options) {
  if (!options.use_control_arm_mask) {
    if (std::strcmp(options.benchmark_case, "bridge") == 0) {
      return (k == 0 && (i == 0 || i == grid.nx - 1)) ? PETSC_TRUE
                                                        : PETSC_FALSE;
    }
    if (std::strcmp(options.benchmark_case, "tri") == 0) {
      const bool left_lower_corner = (i == 0 && j == 0 && k == 0);
      const bool left_upper_corner =
          (i == 0 && j == grid.ny - 1 && k == 0);
      const bool right_upper_corner =
          (i == grid.nx - 1 && j == grid.ny - 1 && k == 0);
      return (left_lower_corner || left_upper_corner || right_upper_corner)
                  ? PETSC_TRUE
                  : PETSC_FALSE;
    }
    // MATLAB cant/mbb clamp the whole left end face.
    return i == 0 ? PETSC_TRUE : PETSC_FALSE;
  }
  return h8_is_c_support_node(i, j, k, grid);
}

PetscBool benchmark_is_torsion(const char *benchmark_case) {
  return std::strcmp(benchmark_case, "torsion") == 0 ? PETSC_TRUE : PETSC_FALSE;
}

PetscBool benchmark_is_torsion_edge(const char *benchmark_case) {
  return (std::strcmp(benchmark_case, "torsion_edge") == 0 ||
          std::strcmp(benchmark_case, "edge_torsion") == 0 ||
          std::strcmp(benchmark_case, "torsion_ring") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool benchmark_is_bridge(const char *benchmark_case) {
  return std::strcmp(benchmark_case, "bridge") == 0 ? PETSC_TRUE : PETSC_FALSE;
}

PetscBool benchmark_is_mbb(const char *benchmark_case) {
  return std::strcmp(benchmark_case, "mbb") == 0 ? PETSC_TRUE : PETSC_FALSE;
}

PetscBool benchmark_is_tri(const char *benchmark_case) {
  return std::strcmp(benchmark_case, "tri") == 0 ? PETSC_TRUE : PETSC_FALSE;
}

PetscBool benchmark_is_bottom_point_cantilever(const char *benchmark_case) {
  return (std::strcmp(benchmark_case, "bottom_point") == 0 ||
          std::strcmp(benchmark_case, "bottom_center") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool benchmark_is_tip_center_cantilever(const char *benchmark_case) {
  return (std::strcmp(benchmark_case, "tip_center") == 0 ||
          std::strcmp(benchmark_case, "center_point") == 0 ||
          std::strcmp(benchmark_case, "face_center") == 0 ||
          std::strcmp(benchmark_case, "right_center") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool benchmark_is_cantilever(const char *benchmark_case) {
  return (std::strcmp(benchmark_case, "cantilever") == 0 ||
          std::strcmp(benchmark_case, "cant") == 0 ||
          benchmark_is_bottom_point_cantilever(benchmark_case) ||
          benchmark_is_tip_center_cantilever(benchmark_case))
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool benchmark_has_matlab_load_elements(const char *benchmark_case) {
  return (benchmark_is_cantilever(benchmark_case) ||
          benchmark_is_bridge(benchmark_case) ||
          benchmark_is_mbb(benchmark_case) ||
          benchmark_is_tri(benchmark_case) ||
          benchmark_is_torsion_edge(benchmark_case))
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool h8_is_right_face_edge_node(PetscInt i, PetscInt j, PetscInt k,
                                     const Grid &grid) {
  return (i == grid.nx - 1 &&
          (j == 0 || j == grid.ny - 1 || k == 0 || k == grid.nz - 1))
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool h8_is_ab_load_node(PetscInt i, PetscInt j, PetscInt k,
                             const Grid &grid, PetscInt which) {
  const PetscReal DL = domain_length(grid);
  const PetscReal DW = domain_width(grid);
  const PetscReal DH = domain_height(grid);
  const PetscReal center[3] = {0.18 * DL, which == 1 ? DW : 0.0,
                               0.50 * DH};
  const PetscReal r = 0.25 * PetscMin(DL, DH);
  const PetscReal hmax = PetscMax(PetscMax(DL / PetscMax(1, grid.nx - 1),
                                           DW / PetscMax(1, grid.ny - 1)),
                                  DH / PetscMax(1, grid.nz - 1));
  const PetscReal tol = 0.75 * hmax;
  const PetscReal axial = PetscMax(0.35 * PetscMin(DL, DH), 2.0 * hmax);
  PetscReal X = 0.0, Y = 0.0, Z = 0.0;
  h8_node_physical(i, j, k, grid, &X, &Y, &Z);

  const PetscReal rr = PetscSqrtReal((X - center[0]) * (X - center[0]) +
                                     (Z - center[2]) * (Z - center[2]));
  return (PetscAbsReal(rr - r) <= tol && PetscAbsReal(Y - center[1]) <= axial)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool h8_is_spring_load_node(PetscInt i, PetscInt j, PetscInt k,
                                 const Grid &grid) {
  const PetscReal DL = domain_length(grid);
  const PetscReal DW = domain_width(grid);
  const PetscReal DH = domain_height(grid);
  PetscReal X = 0.0, Y = 0.0, Z = 0.0;
  h8_node_physical(i, j, k, grid, &X, &Y, &Z);
  return (PetscAbsReal(X - 0.50 * DL) <= 0.105 * DL / 2.0 &&
          PetscAbsReal(Y - 0.50 * DW) <= 0.105 * DW / 2.0 &&
          Z >= 0.25 * DH && Z <= 0.75 * DH)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool h8_cell_has_c_support_node(PetscInt ex, PetscInt ey, PetscInt ez,
                                     const Grid &grid) {
  for (PetscInt node = 0; node < 8; ++node) {
    PetscInt i = 0, j = 0, k = 0;
    h8_node_coords(ex, ey, ez, node, &i, &j, &k);
    if (h8_is_c_support_node(i, j, k, grid)) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

PetscBool h8_cell_has_ab_load_node(PetscInt ex, PetscInt ey, PetscInt ez,
                                   const Grid &grid, PetscInt which) {
  for (PetscInt node = 0; node < 8; ++node) {
    PetscInt i = 0, j = 0, k = 0;
    h8_node_coords(ex, ey, ez, node, &i, &j, &k);
    if (h8_is_ab_load_node(i, j, k, grid, which)) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

PetscBool h8_cell_has_spring_load_node(PetscInt ex, PetscInt ey, PetscInt ez,
                                       const Grid &grid) {
  for (PetscInt node = 0; node < 8; ++node) {
    PetscInt i = 0, j = 0, k = 0;
    h8_node_coords(ex, ey, ez, node, &i, &j, &k);
    if (h8_is_spring_load_node(i, j, k, grid)) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

bool h8_mask_is_void(PetscReal value) { return value <= 0.5; }

bool h8_mask_is_design(PetscReal value) {
  return value > 0.5 && value < 1.5;
}

bool h8_mask_is_fixed_solid(PetscReal value) { return value >= 1.5; }

bool h8_mask_is_active(PetscReal value) { return value > 0.5; }

bool h8_mask_counts_in_volume(PetscReal value, PetscBool include_fixed_solid) {
  return h8_mask_is_design(value) ||
         (include_fixed_solid && h8_mask_is_fixed_solid(value));
}

void h8_cell_center_physical(PetscInt i, PetscInt j, PetscInt k,
                             const Grid &grid, PetscReal *x, PetscReal *y,
                             PetscReal *z) {
  *x = grid.nx > 1 ? (static_cast<PetscReal>(i) + 0.5) /
                         static_cast<PetscReal>(grid.nx - 1) *
                         domain_length(grid)
                   : 0.0;
  *y = grid.ny > 1 ? (static_cast<PetscReal>(j) + 0.5) /
                         static_cast<PetscReal>(grid.ny - 1) *
                         domain_width(grid)
                   : 0.0;
  *z = grid.nz > 1 ? (static_cast<PetscReal>(k) + 0.5) /
                         static_cast<PetscReal>(grid.nz - 1) *
                         domain_height(grid)
                   : 0.0;
}

PetscBool h8_cell_is_forced_solid(PetscInt i, PetscInt j, PetscInt k,
                                   const Grid &grid,
                                   const DensityOptions &options,
                                   const OptimizerOptions &opt) {
  if (!options.use_control_arm_mask) {
    const PetscInt ex = grid.nx - 1;
    const PetscInt ey = grid.ny - 1;
    const PetscInt ez = grid.nz - 1;
    if (ex <= 0 || ey <= 0 || ez <= 0) return PETSC_FALSE;
    if (benchmark_is_bridge(opt.benchmark_case)) {
      return k == ez - 1 ? PETSC_TRUE : PETSC_FALSE;
    }
    if (benchmark_is_mbb(opt.benchmark_case)) {
      return (i == ex - 1 && k == 0) ? PETSC_TRUE : PETSC_FALSE;
    }
    if (benchmark_is_tri(opt.benchmark_case)) {
      return (i == 0 && j == 0 && k == ez - 1) ? PETSC_TRUE : PETSC_FALSE;
    }
    if (benchmark_is_torsion_edge(opt.benchmark_case)) {
      return (i == ex - 1 && (j == 0 || j == ey - 1 || k == 0 || k == ez - 1))
                 ? PETSC_TRUE
                 : PETSC_FALSE;
    }
    if (benchmark_is_tip_center_cantilever(opt.benchmark_case)) {
      const PetscInt mid_j = ey / 2;
      const PetscInt lower_j = PetscMax(static_cast<PetscInt>(0), mid_j - 1);
      const PetscInt upper_j = PetscMin(ey - 1, mid_j);
      const PetscInt mid_k = ez / 2;
      const PetscInt lower_k = PetscMax(static_cast<PetscInt>(0), mid_k - 1);
      const PetscInt upper_k = PetscMin(ez - 1, mid_k);
      return (i == ex - 1 && (j == lower_j || j == upper_j) &&
              (k == lower_k || k == upper_k))
                 ? PETSC_TRUE
                 : PETSC_FALSE;
    }
    if (benchmark_is_cantilever(opt.benchmark_case) &&
        !benchmark_is_bottom_point_cantilever(opt.benchmark_case) &&
        !benchmark_is_tip_center_cantilever(opt.benchmark_case)) {
      const PetscInt mid_node_k = ez / 2;
      const PetscInt lower_k =
          PetscMax(static_cast<PetscInt>(0), mid_node_k - 1);
      const PetscInt upper_k = PetscMin(ez - 1, mid_node_k);
      return (i == ex - 1 && (k == lower_k || k == upper_k)) ? PETSC_TRUE
                                                             : PETSC_FALSE;
    }
    return PETSC_FALSE;
  }
  const PetscReal DL = domain_length(grid);
  const PetscReal DW = domain_width(grid);
  const PetscReal DH = domain_height(grid);
  const PetscReal min_ld = PetscMin(DL, DH);
  const PetscReal min_lw = PetscMin(DL, DW);
  const PetscReal A[3] = {0.18 * DL, DW, 0.50 * DH};
  const PetscReal B[3] = {0.18 * DL, 0.0, 0.50 * DH};
  const PetscReal C[3] = {0.78 * DL, 0.50 * DW, 0.50 * DH};
  const PetscReal spring_center[3] = {0.50 * DL, 0.50 * DW, 0.50 * DH};
  const PetscReal rA_hole = 0.25 * min_ld;
  const PetscReal rB_hole = 0.25 * min_ld;
  const PetscReal rC_hole = 0.10 * min_lw;
  const PetscReal rA_pad = 0.35 * min_ld;
  const PetscReal rB_pad = 0.35 * min_ld;
  const PetscReal rC_pad = 0.15 * min_lw;
  PetscReal X = 0.0, Y = 0.0, Z = 0.0;
  h8_cell_center_physical(i, j, k, grid, &X, &Y, &Z);

  const bool middle = (Z >= 0.25 * DH && Z <= 0.75 * DH);
  const bool axialA = (A[1] - Y <= rA_pad);
  const bool axialB = (Y - B[1] <= rB_pad);
  const PetscReal rrA2 = (X - A[0]) * (X - A[0]) + (Z - A[2]) * (Z - A[2]);
  const PetscReal rrB2 = (X - B[0]) * (X - B[0]) + (Z - B[2]) * (Z - B[2]);
  const PetscReal rrC2 = (X - C[0]) * (X - C[0]) + (Y - C[1]) * (Y - C[1]);
  const bool holeA = rrA2 <= rA_hole * rA_hole && axialA;
  const bool holeB = rrB2 <= rB_hole * rB_hole && axialB;
  const bool holeC = rrC2 <= rC_hole * rC_hole;
  const bool ringA = rrA2 <= rA_pad * rA_pad && axialA && !holeA;
  const bool ringB = rrB2 <= rB_pad * rB_pad && axialB && !holeB;
  const bool ringC = rrC2 <= rC_pad * rC_pad && middle && !holeC;
  const bool spring_mount =
      middle &&
      (PetscAbsReal(X - spring_center[0]) <= 0.105 * DL / 2.0) &&
      (PetscAbsReal(Y - spring_center[1]) <= 0.105 * DW / 2.0);

  if (ringA || ringB || ringC || spring_mount) return PETSC_TRUE;
  if (h8_cell_has_ab_load_node(i, j, k, grid, 1)) return PETSC_TRUE;
  if (h8_cell_has_ab_load_node(i, j, k, grid, 2)) return PETSC_TRUE;
  if (h8_cell_has_c_support_node(i, j, k, grid)) return PETSC_TRUE;
  if (h8_cell_has_spring_load_node(i, j, k, grid)) return PETSC_TRUE;
  return PETSC_FALSE;
}

PetscReal simp_scale(PetscReal rho, const DensityOptions &opts) {
  rho = PetscMax(1.0e-12, PetscMin(1.0, rho));
  return opts.emin + (1.0 - opts.emin) * PetscPowReal(rho, opts.penal);
}

PetscReal simp_derivative(PetscReal rho, const DensityOptions &opts) {
  rho = PetscMax(1.0e-12, PetscMin(1.0, rho));
  return (1.0 - opts.emin) * opts.penal * PetscPowReal(rho, opts.penal - 1.0);
}

void compute_matlab_lk_h8(PetscReal nu, PetscReal *ke) {
  PetscReal matlab_ke[24 * 24]{};
  const PetscReal A[2][14] = {
      {32.0, 6.0, -8.0, 6.0, -6.0, 4.0, 3.0, -6.0, -10.0, 3.0,
       -3.0, -3.0, -4.0, -8.0},
      {-48.0, 0.0, 0.0, -24.0, 24.0, 0.0, 0.0, 0.0, 12.0, -12.0,
       0.0, 12.0, 12.0, 12.0}};
  PetscReal k[14] = {};
  for (PetscInt q = 0; q < 14; ++q) {
    k[q] = (A[0][q] + nu * A[1][q]) / 144.0;
  }

  const PetscReal K1[6][6] = {
      {k[0], k[1], k[1], k[2], k[4], k[4]},
      {k[1], k[0], k[1], k[3], k[5], k[6]},
      {k[1], k[1], k[0], k[3], k[6], k[5]},
      {k[2], k[3], k[3], k[0], k[7], k[7]},
      {k[4], k[5], k[6], k[7], k[0], k[1]},
      {k[4], k[6], k[5], k[7], k[1], k[0]}};
  const PetscReal K2[6][6] = {
      {k[8], k[7], k[11], k[5], k[3], k[6]},
      {k[7], k[8], k[11], k[4], k[2], k[4]},
      {k[9], k[9], k[12], k[6], k[3], k[5]},
      {k[5], k[4], k[10], k[8], k[1], k[9]},
      {k[3], k[2], k[4], k[1], k[8], k[11]},
      {k[10], k[3], k[5], k[11], k[9], k[12]}};
  const PetscReal K3[6][6] = {
      {k[5], k[6], k[3], k[8], k[11], k[7]},
      {k[6], k[5], k[3], k[9], k[12], k[9]},
      {k[4], k[4], k[2], k[7], k[11], k[8]},
      {k[8], k[9], k[1], k[5], k[10], k[4]},
      {k[11], k[12], k[9], k[10], k[5], k[3]},
      {k[1], k[11], k[8], k[3], k[4], k[2]}};
  const PetscReal K4[6][6] = {
      {k[13], k[10], k[10], k[12], k[9], k[9]},
      {k[10], k[13], k[10], k[11], k[8], k[7]},
      {k[10], k[10], k[13], k[11], k[7], k[8]},
      {k[12], k[11], k[11], k[13], k[6], k[6]},
      {k[9], k[8], k[7], k[6], k[13], k[10]},
      {k[9], k[7], k[8], k[6], k[10], k[13]}};
  const PetscReal K5[6][6] = {
      {k[0], k[1], k[7], k[2], k[4], k[3]},
      {k[1], k[0], k[7], k[3], k[5], k[10]},
      {k[7], k[7], k[0], k[4], k[10], k[5]},
      {k[2], k[3], k[4], k[0], k[7], k[1]},
      {k[4], k[5], k[10], k[7], k[0], k[7]},
      {k[3], k[10], k[5], k[1], k[7], k[0]}};
  const PetscReal K6[6][6] = {
      {k[13], k[10], k[6], k[12], k[9], k[11]},
      {k[10], k[13], k[6], k[11], k[8], k[1]},
      {k[6], k[6], k[13], k[9], k[1], k[8]},
      {k[12], k[11], k[9], k[13], k[6], k[10]},
      {k[9], k[8], k[1], k[6], k[13], k[6]},
      {k[11], k[1], k[8], k[10], k[6], k[13]}};
  const PetscReal factor = 1.0 / ((nu + 1.0) * (1.0 - 2.0 * nu));
  for (PetscInt q = 0; q < 24 * 24; ++q) matlab_ke[q] = 0.0;

  auto set_block = [&](PetscInt br, PetscInt bc, const PetscReal M[6][6],
                       PetscBool transpose) {
    for (PetscInt r = 0; r < 6; ++r) {
      for (PetscInt c = 0; c < 6; ++c) {
        const PetscInt row = br * 6 + r;
        const PetscInt col = bc * 6 + c;
        matlab_ke[24 * row + col] =
            factor * (transpose ? M[c][r] : M[r][c]);
      }
    }
  };

  set_block(0, 0, K1, PETSC_FALSE);
  set_block(0, 1, K2, PETSC_FALSE);
  set_block(0, 2, K3, PETSC_FALSE);
  set_block(0, 3, K4, PETSC_FALSE);
  set_block(1, 0, K2, PETSC_TRUE);
  set_block(1, 1, K5, PETSC_FALSE);
  set_block(1, 2, K6, PETSC_FALSE);
  set_block(1, 3, K3, PETSC_TRUE);
  set_block(2, 0, K3, PETSC_TRUE);
  set_block(2, 1, K6, PETSC_FALSE);
  set_block(2, 2, K5, PETSC_TRUE);
  set_block(2, 3, K2, PETSC_TRUE);
  set_block(3, 0, K4, PETSC_FALSE);
  set_block(3, 1, K3, PETSC_FALSE);
  set_block(3, 2, K2, PETSC_FALSE);
  set_block(3, 3, K1, PETSC_TRUE);

  // MATLAB initiMesh uses a y-reversed local node order for lk_H8.
  // Reorder once into the C++ kernels' local node convention.
  const PetscInt cpp_to_matlab_node[8] = {3, 2, 1, 0, 7, 6, 5, 4};
  PetscInt cpp_to_matlab_dof[24]{};
  for (PetscInt node = 0; node < 8; ++node) {
    for (PetscInt d = 0; d < 3; ++d) {
      cpp_to_matlab_dof[3 * node + d] = 3 * cpp_to_matlab_node[node] + d;
    }
  }
  for (PetscInt r = 0; r < 24; ++r) {
    for (PetscInt c = 0; c < 24; ++c) {
      ke[24 * r + c] =
          matlab_ke[24 * cpp_to_matlab_dof[r] + cpp_to_matlab_dof[c]];
    }
  }
}

void compute_h8_element_stiffness(const Grid &grid,
                                  const DensityOptions &options,
                                  PetscReal *ke) {
  const PetscReal nu = 0.30;
  if (!options.use_control_arm_mask &&
      benchmark_has_matlab_load_elements(options.benchmark_case)) {
    (void)grid;
    compute_matlab_lk_h8(nu, ke);
    return;
  }
  const PetscReal E = options.young_modulus;
  const PetscReal lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const PetscReal mu = E / (2.0 * (1.0 + nu));
  PetscReal D[6][6] = {};
  const PetscReal hx = domain_length(grid) / static_cast<PetscReal>(grid.nx - 1);
  const PetscReal hy = domain_width(grid) / static_cast<PetscReal>(grid.ny - 1);
  const PetscReal hz = domain_height(grid) / static_cast<PetscReal>(grid.nz - 1);
  const PetscReal detJ = hx * hy * hz / 8.0;
  const PetscReal g = 1.0 / PetscSqrtReal(3.0);
  const PetscReal gps[2] = {-g, g};
  static const PetscReal xi_node[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
  static const PetscReal eta_node[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
  static const PetscReal zeta_node[8] = {-1, -1, -1, -1, 1, 1, 1, 1};

  for (PetscInt i = 0; i < 24 * 24; ++i) ke[i] = 0.0;
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) D[i][j] = lambda;
    D[i][i] = lambda + 2.0 * mu;
  }
  D[3][3] = mu;
  D[4][4] = mu;
  D[5][5] = mu;

  for (PetscInt gz = 0; gz < 2; ++gz) {
    for (PetscInt gy = 0; gy < 2; ++gy) {
      for (PetscInt gx = 0; gx < 2; ++gx) {
        const PetscReal xi = gps[gx];
        const PetscReal eta = gps[gy];
        const PetscReal zeta = gps[gz];
        PetscReal B[6][24] = {};

        for (PetscInt a = 0; a < 8; ++a) {
          const PetscReal dN_dxi =
              0.125 * xi_node[a] * (1.0 + eta * eta_node[a]) *
              (1.0 + zeta * zeta_node[a]);
          const PetscReal dN_deta =
              0.125 * eta_node[a] * (1.0 + xi * xi_node[a]) *
              (1.0 + zeta * zeta_node[a]);
          const PetscReal dN_dzeta =
              0.125 * zeta_node[a] * (1.0 + xi * xi_node[a]) *
              (1.0 + eta * eta_node[a]);
          const PetscReal dN_dx = dN_dxi * 2.0 / hx;
          const PetscReal dN_dy = dN_deta * 2.0 / hy;
          const PetscReal dN_dz = dN_dzeta * 2.0 / hz;
          const PetscInt col = 3 * a;

          B[0][col + 0] = dN_dx;
          B[1][col + 1] = dN_dy;
          B[2][col + 2] = dN_dz;
          B[3][col + 0] = dN_dy;
          B[3][col + 1] = dN_dx;
          B[4][col + 1] = dN_dz;
          B[4][col + 2] = dN_dy;
          B[5][col + 0] = dN_dz;
          B[5][col + 2] = dN_dx;
        }

        for (PetscInt a = 0; a < 24; ++a) {
          for (PetscInt b = 0; b < 24; ++b) {
            PetscReal value = 0.0;
            for (PetscInt r = 0; r < 6; ++r) {
              for (PetscInt s = 0; s < 6; ++s) {
                value += B[r][a] * D[r][s] * B[s][b];
              }
            }
            ke[24 * a + b] += value * detJ;
          }
        }
      }
    }
  }
}

PetscReal cone_weight(PetscInt dx, PetscInt dy, PetscInt dz, PetscReal radius) {
  const PetscReal dist =
      PetscSqrtReal(static_cast<PetscReal>(dx * dx + dy * dy + dz * dz));
  return PetscMax(0.0, radius - dist);
}

PetscInt ceil_div(PetscInt a, PetscInt b) {
  return (a + b - 1) / b;
}

PetscErrorCode elapsed_max(PetscLogDouble start, PetscReal *elapsed) {
  PetscLogDouble now = 0.0;
  PetscReal local_elapsed = 0.0;
  PetscCall(PetscTime(&now));
  local_elapsed = static_cast<PetscReal>(now - start);
  PetscCallMPI(MPI_Allreduce(&local_elapsed, elapsed, 1, MPIU_REAL, MPI_MAX,
                             PETSC_COMM_WORLD));
  return 0;
}

const char *ksp_reason_name(KSPConvergedReason reason) {
  switch (reason) {
    case KSP_CONVERGED_RTOL_NORMAL: return "KSP_CONVERGED_RTOL_NORMAL";
    case KSP_CONVERGED_ATOL_NORMAL: return "KSP_CONVERGED_ATOL_NORMAL";
    case KSP_CONVERGED_RTOL: return "KSP_CONVERGED_RTOL";
    case KSP_CONVERGED_ATOL: return "KSP_CONVERGED_ATOL";
    case KSP_CONVERGED_ITS: return "KSP_CONVERGED_ITS";
    case KSP_CONVERGED_NEG_CURVE: return "KSP_CONVERGED_NEG_CURVE";
    case KSP_CONVERGED_STEP_LENGTH: return "KSP_CONVERGED_STEP_LENGTH";
    case KSP_CONVERGED_HAPPY_BREAKDOWN: return "KSP_CONVERGED_HAPPY_BREAKDOWN";
    case KSP_DIVERGED_NULL: return "KSP_DIVERGED_NULL";
    case KSP_DIVERGED_ITS: return "KSP_DIVERGED_ITS";
    case KSP_DIVERGED_DTOL: return "KSP_DIVERGED_DTOL";
    case KSP_DIVERGED_BREAKDOWN: return "KSP_DIVERGED_BREAKDOWN";
    case KSP_DIVERGED_BREAKDOWN_BICG: return "KSP_DIVERGED_BREAKDOWN_BICG";
    case KSP_DIVERGED_NONSYMMETRIC: return "KSP_DIVERGED_NONSYMMETRIC";
    case KSP_DIVERGED_INDEFINITE_PC: return "KSP_DIVERGED_INDEFINITE_PC";
    case KSP_DIVERGED_NANORINF: return "KSP_DIVERGED_NANORINF";
    case KSP_DIVERGED_INDEFINITE_MAT: return "KSP_DIVERGED_INDEFINITE_MAT";
    case KSP_DIVERGED_PC_FAILED: return "KSP_DIVERGED_PC_FAILED";
    case KSP_CONVERGED_ITERATING: return "KSP_CONVERGED_ITERATING";
    default: return "KSP_REASON_UNKNOWN";
  }
}

PetscErrorCode choose_h8_process_grid(const Grid &grid,
                                      PetscInt element_stencil_width,
                                      PetscMPIInt ranks,
                                      PetscInt *px, PetscInt *py, PetscInt *pz) {
  const PetscInt ex = grid.nx - 1;
  const PetscInt ey = grid.ny - 1;
  const PetscInt ez = grid.nz - 1;
  const PetscInt stencil = PetscMax(1, element_stencil_width);
  PetscInt opt_px = 0, opt_py = 0, opt_pz = 0;
  PetscBool set_px = PETSC_FALSE, set_py = PETSC_FALSE, set_pz = PETSC_FALSE;

  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-h8_dm_px", &opt_px, &set_px));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-h8_dm_py", &opt_py, &set_py));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-h8_dm_pz", &opt_pz, &set_pz));
  if (set_px || set_py || set_pz) {
    PetscCheck(set_px && set_py && set_pz, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_INCOMP,
               "Set -h8_dm_px, -h8_dm_py, and -h8_dm_pz together");
    PetscCheck(opt_px > 0 && opt_py > 0 && opt_pz > 0, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_OUTOFRANGE,
               "H8 process-grid dimensions must be positive");
    PetscCheck(opt_px * opt_py * opt_pz == static_cast<PetscInt>(ranks),
               PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
               "H8 process grid product must equal MPI ranks");
    PetscCheck(ex >= opt_px * stencil && ey >= opt_py * stencil &&
                   ez >= opt_pz * stencil,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "H8 process grid needs each local element block at least as thick as the density filter stencil in x, y, and z");
    *px = opt_px;
    *py = opt_py;
    *pz = opt_pz;
    return 0;
  }

  PetscBool found = PETSC_FALSE;
  PetscReal best_score = PETSC_MAX_REAL;
  PetscInt best_px = 1, best_py = 1, best_pz = static_cast<PetscInt>(ranks);

  for (PetscMPIInt tx = 1; tx <= ranks; ++tx) {
    if (ranks % tx != 0) continue;
    const PetscMPIInt rem = ranks / tx;
    for (PetscMPIInt ty = 1; ty <= rem; ++ty) {
      if (rem % ty != 0) continue;
      const PetscMPIInt tz = rem / ty;
      if (ex < static_cast<PetscInt>(tx) * stencil ||
          ey < static_cast<PetscInt>(ty) * stencil ||
          ez < static_cast<PetscInt>(tz) * stencil) {
        continue;
      }

      const PetscReal lx =
          static_cast<PetscReal>(ceil_div(ex, static_cast<PetscInt>(tx)));
      const PetscReal ly =
          static_cast<PetscReal>(ceil_div(ey, static_cast<PetscInt>(ty)));
      const PetscReal lz =
          static_cast<PetscReal>(ceil_div(ez, static_cast<PetscInt>(tz)));
      const PetscReal local_elems = lx * ly * lz;
      const PetscReal halo_ratio =
          (lx + 2.0 * stencil) * (ly + 2.0 * stencil) *
          (lz + 2.0 * stencil) / PetscMax(1.0, local_elems);
      const PetscReal local_min = PetscMin(lx, PetscMin(ly, lz));
      const PetscReal local_max = PetscMax(lx, PetscMax(ly, lz));
      const PetscReal aspect = local_max / PetscMax(1.0, local_min);
      const PetscReal score =
          local_elems * (1.0 + 0.10 * (halo_ratio - 1.0)) *
          (1.0 + 0.02 * (aspect - 1.0));

      if (!found || score < best_score) {
        found = PETSC_TRUE;
        best_score = score;
        best_px = static_cast<PetscInt>(tx);
        best_py = static_cast<PetscInt>(ty);
        best_pz = static_cast<PetscInt>(tz);
      }
    }
  }

  PetscCheck(found, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "H8 optimizer could not find a 3D process grid where each local element block covers the density filter stencil; reduce ranks/filter radius or increase nx/ny/nz");
  *px = best_px;
  *py = best_py;
  *pz = best_pz;
  return 0;
}

PetscErrorCode compute_masked_volume_fraction_with_fixed(
    DM eda, Vec rho, Vec mask, PetscBool include_fixed_solid,
    PetscReal *volume_fraction) {
  PetscScalar ***r = nullptr;
  PetscScalar ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_rho_sum = 0.0;
  PetscReal local_mask_sum = 0.0;
  PetscReal global_rho_sum = 0.0;
  PetscReal global_mask_sum = 0.0;

  PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (h8_mask_counts_in_volume(mask_value, include_fixed_solid)) {
          local_mask_sum += 1.0;
          local_rho_sum += PetscRealPart(r[k][j][i]);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCallMPI(MPI_Allreduce(&local_rho_sum, &global_rho_sum, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_mask_sum, &global_mask_sum, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  *volume_fraction = global_mask_sum > 0.0 ? global_rho_sum / global_mask_sum : 0.0;
  return 0;
}

PetscErrorCode compute_masked_volume_fraction(DM eda, Vec rho, Vec mask,
                                              PetscReal *volume_fraction) {
  PetscCall(compute_masked_volume_fraction_with_fixed(
      eda, rho, mask, PETSC_FALSE, volume_fraction));
  return 0;
}

PetscErrorCode create_h8_dms(const Grid &grid, PetscInt element_stencil_width,
                             DM *uda, DM *eda) {
  PetscMPIInt ranks = 1;
  const PetscInt estencil = PetscMax(1, element_stencil_width);
  const PetscInt density_stencil = PetscMax(estencil, 2);
  PetscInt px = 1, py = 1, pz = 1;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCall(choose_h8_process_grid(grid, estencil, ranks, &px, &py, &pz));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 optimizer DMDA process grid: px=%lld py=%lld pz=%lld, filter_stencil=%lld density_stencil=%lld\n",
                        static_cast<long long>(px),
                        static_cast<long long>(py),
                        static_cast<long long>(pz),
                        static_cast<long long>(estencil),
                        static_cast<long long>(density_stencil)));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         grid.nx, grid.ny, grid.nz,
                         px, py, pz, 3, 1,
                         nullptr, nullptr, nullptr, uda));
  PetscCall(DMSetFromOptions(*uda));
  PetscCall(DMSetUp(*uda));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         grid.nx - 1, grid.ny - 1, grid.nz - 1,
                         px, py, pz, 1, density_stencil,
                         nullptr, nullptr, nullptr, eda));
  PetscCall(DMSetFromOptions(*eda));
  PetscCall(DMSetUp(*eda));
  return 0;
}

PetscErrorCode compute_filter_denominator(DM eda, Vec mask, PetscReal radius,
                                          Vec denom) {
  if (radius <= 0.0) {
    PetscCall(VecSet(denom, 1.0));
    return 0;
  }

  Vec local_mask = nullptr;
  PetscScalar ***m = nullptr;
  PetscScalar ***d = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  const PetscInt r = PetscMax(0, static_cast<PetscInt>(PetscCeilReal(radius)));

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecGetArray(eda, denom, &d));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        PetscReal sum = 0.0;
        for (PetscInt dz = -r; dz <= r; ++dz) {
          const PetscInt kk = k + dz;
          if (kk < 0 || kk >= ez) continue;
          for (PetscInt dy = -r; dy <= r; ++dy) {
            const PetscInt jj = j + dy;
            if (jj < 0 || jj >= ey) continue;
            for (PetscInt dx = -r; dx <= r; ++dx) {
              const PetscInt ii = i + dx;
              if (ii < 0 || ii >= ex) continue;
              const PetscReal w = cone_weight(dx, dy, dz, radius);
              if (w > 0.0 && PetscRealPart(m[kk][jj][ii]) > 0.5) {
                sum += w;
              }
            }
          }
        }
        d[k][j][i] = PetscMax(sum, 1.0e-12);
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecRestoreArray(eda, denom, &d));
  PetscCall(VecDestroy(&local_mask));
  return 0;
}

PetscErrorCode apply_density_filter(DM eda, Vec rho_design, Vec mask,
                                    Vec denom, PetscReal radius,
                                    PetscReal void_density, Vec rho_phys) {
  if (radius <= 0.0) {
    PetscScalar ***r = nullptr, ***m = nullptr, ***out = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscCall(DMDAVecGetArrayRead(eda, rho_design, &r));
    PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
    PetscCall(DMDAVecGetArray(eda, rho_phys, &out));
    PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          const PetscReal mask_value = PetscRealPart(m[k][j][i]);
          if (h8_mask_is_fixed_solid(mask_value)) {
            out[k][j][i] = 1.0;
          } else if (h8_mask_is_design(mask_value)) {
            out[k][j][i] = r[k][j][i];
          } else {
            out[k][j][i] = void_density;
          }
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(eda, rho_design, &r));
    PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
    PetscCall(DMDAVecRestoreArray(eda, rho_phys, &out));
    return 0;
  }

  Vec local_rho = nullptr, local_mask = nullptr;
  PetscScalar ***r = nullptr, ***m = nullptr, ***den = nullptr, ***out = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  const PetscInt rr = PetscMax(0, static_cast<PetscInt>(PetscCeilReal(radius)));

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCall(DMCreateLocalVector(eda, &local_rho));
  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, rho_design, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(eda, rho_design, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalBegin(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayRead(eda, local_rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecGetArrayRead(eda, denom, &den));
  PetscCall(DMDAVecGetArray(eda, rho_phys, &out));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (h8_mask_is_fixed_solid(mask_value)) {
          out[k][j][i] = 1.0;
          continue;
        }
        if (h8_mask_is_void(mask_value)) {
          out[k][j][i] = void_density;
          continue;
        }
        PetscReal sum = 0.0;
        for (PetscInt dz = -rr; dz <= rr; ++dz) {
          const PetscInt kk = k + dz;
          if (kk < 0 || kk >= ez) continue;
          for (PetscInt dy = -rr; dy <= rr; ++dy) {
            const PetscInt jj = j + dy;
            if (jj < 0 || jj >= ey) continue;
            for (PetscInt dx = -rr; dx <= rr; ++dx) {
              const PetscInt ii = i + dx;
              if (ii < 0 || ii >= ex) continue;
              const PetscReal w = cone_weight(dx, dy, dz, radius);
              if (w > 0.0 && h8_mask_is_active(PetscRealPart(m[kk][jj][ii]))) {
                sum += w * PetscRealPart(r[kk][jj][ii]);
              }
            }
          }
        }
        out[k][j][i] = sum / PetscRealPart(den[k][j][i]);
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(eda, local_rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecRestoreArrayRead(eda, denom, &den));
  PetscCall(DMDAVecRestoreArray(eda, rho_phys, &out));
  PetscCall(VecDestroy(&local_rho));
  PetscCall(VecDestroy(&local_mask));
  return 0;
}

// 根据当前 H8 优化步数计算 Heaviside 投影 beta。
// interval<=0 时沿整个优化过程自动分段翻倍；interval>0 时每隔 interval 步翻倍，直到 beta_max。
PetscReal h8_heaviside_beta_for_iter(const OptimizerOptions &opt,
                                     PetscInt iter) {
  if (!opt.heaviside_projection) return 0.0;
  PetscReal beta = PetscMax(1.0e-12, opt.heaviside_beta_initial);
  if (opt.heaviside_beta_interval <= 0) {
    if (opt.max_iter <= 1 || opt.heaviside_beta_max <= beta) {
      return PetscMin(beta, opt.heaviside_beta_max);
    }
    const PetscReal ratio = opt.heaviside_beta_max / beta;
    const PetscInt stages = PetscMax(
        static_cast<PetscInt>(std::ceil(std::log(ratio) / std::log(2.0))),
        static_cast<PetscInt>(1));
    const PetscReal progress =
        static_cast<PetscReal>(PetscMax(iter - 1, 0)) *
        static_cast<PetscReal>(stages + 1) /
        static_cast<PetscReal>(PetscMax(opt.max_iter, 1));
    const PetscInt stage = PetscMin(
        stages, static_cast<PetscInt>(std::floor(progress)));
    for (PetscInt s = 0; s < stage; ++s) beta *= 2.0;
    return PetscMin(beta, opt.heaviside_beta_max);
  }
  if (iter > 1) {
    const PetscInt stage = (iter - 1) / opt.heaviside_beta_interval;
    for (PetscInt s = 0; s < stage; ++s) beta *= 2.0;
  }
  return PetscMin(beta, opt.heaviside_beta_max);
}

PetscReal h8_matlab_outer_beta_for_iter(const OptimizerOptions &opt,
                                        PetscInt iter) {
  const PetscReal beta0 =
      opt.heaviside_projection ? opt.heaviside_beta_initial : 2.0;
  const PetscReal beta_max =
      opt.heaviside_projection ? opt.heaviside_beta_max : 10.0;
  return PetscMin(PetscMax(1.0e-12, beta0) +
                      0.1 * static_cast<PetscReal>(PetscMax(iter, 0)),
                  PetscMax(PetscMax(1.0e-12, beta0), beta_max));
}

PetscReal h8_matlab_inner_beta_for_iter(const OptimizerOptions &opt,
                                        PetscInt iter) {
  return PetscMin(PetscMax(1.0e-12, opt.draft_beta) +
                      0.1 * static_cast<PetscReal>(PetscMax(iter, 0)),
                  PetscMax(PetscMax(1.0e-12, opt.draft_beta), 20.0));
}

OptimizerOptions h8_projection_options_for_iter(const OptimizerOptions &opt,
                                                PetscBool matlab_z_projection,
                                                PetscInt iter) {
  OptimizerOptions step = opt;
  if (matlab_z_projection) {
    step.draft_beta = h8_matlab_inner_beta_for_iter(opt, iter);
  }
  return step;
}

PetscReal h8_smooth_heaviside_value(PetscReal x, PetscReal beta,
                                    PetscReal eta) {
  const PetscReal den =
      PetscTanhReal(beta * eta) + PetscTanhReal(beta * (1.0 - eta));
  if (den <= 0.0) return x;
  return (PetscTanhReal(beta * eta) + PetscTanhReal(beta * (x - eta))) /
         den;
}

PetscReal h8_smooth_heaviside_derivative(PetscReal x, PetscReal beta,
                                         PetscReal eta) {
  const PetscReal den =
      PetscTanhReal(beta * eta) + PetscTanhReal(beta * (1.0 - eta));
  if (den <= 0.0) return 1.0;
  const PetscReal t = PetscTanhReal(beta * (x - eta));
  return beta * (1.0 - t * t) / den;
}

// H8 密度链路：设计密度 -> 密度滤波 -> Heaviside 投影 -> 拔模闭包。
// 这里仅处理 Heaviside 步，拔模闭包仍由后续 apply_h8_draft_closure 负责。
PetscErrorCode apply_h8_heaviside_projection(
    DM eda, Vec filtered, Vec mask, const OptimizerOptions &opt,
    PetscReal beta, PetscReal void_density, Vec projected) {
  if (!opt.heaviside_projection) {
    PetscCall(VecCopy(filtered, projected));
    return 0;
  }

  PetscScalar ***src = nullptr, ***m = nullptr, ***dst = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(DMDAVecGetArrayRead(eda, filtered, &src));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAVecGetArray(eda, projected, &dst));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (h8_mask_is_fixed_solid(mask_value)) {
          dst[k][j][i] = 1.0;
        } else if (h8_mask_is_design(mask_value)) {
          dst[k][j][i] = PetscMax(
              opt.rho_min,
              PetscMin(1.0, h8_smooth_heaviside_value(
                                PetscRealPart(src[k][j][i]), beta,
                                opt.heaviside_eta)));
        } else {
          dst[k][j][i] = void_density;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, filtered, &src));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCall(DMDAVecRestoreArray(eda, projected, &dst));
  return 0;
}

// 将投影后物理密度上的柔度灵敏度反传到滤波密度上，再交给密度滤波伴随反传。
PetscErrorCode apply_h8_heaviside_sensitivity(
    DM eda, Vec dc_projected, Vec filtered, Vec mask,
    const OptimizerOptions &opt, PetscReal beta, Vec dc_filtered) {
  if (!opt.heaviside_projection) {
    PetscCall(VecCopy(dc_projected, dc_filtered));
    return 0;
  }

  PetscScalar ***dc = nullptr, ***rho = nullptr, ***m = nullptr;
  PetscScalar ***out = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(DMDAVecGetArrayRead(eda, dc_projected, &dc));
  PetscCall(DMDAVecGetArrayRead(eda, filtered, &rho));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAVecGetArray(eda, dc_filtered, &out));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (!h8_mask_is_design(mask_value)) {
          out[k][j][i] = 0.0;
          continue;
        }
        const PetscReal deriv = h8_smooth_heaviside_derivative(
            PetscRealPart(rho[k][j][i]), beta, opt.heaviside_eta);
        out[k][j][i] = PetscRealPart(dc[k][j][i]) * deriv;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, dc_projected, &dc));
  PetscCall(DMDAVecRestoreArrayRead(eda, filtered, &rho));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCall(DMDAVecRestoreArray(eda, dc_filtered, &out));
  return 0;
}

#if 0
PetscErrorCode apply_axis_draft_closure(DM eda, Vec mask, PetscReal eta,
                                        PetscInt axis, PetscInt sign, Vec rho) {
  PetscScalar ***r = nullptr;
  PetscScalar ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  PetscMPIInt reduce_count = 0;

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  const PetscInt axis_length = h8_draft_axis_length(axis, ex, ey, ez);
  const PetscInt column_count = h8_draft_column_count(axis, ex, ey, ez);
  PetscCall(PetscMPIIntCast(column_count, &reduce_count));
  const PetscReal threshold = PetscMax(0.0, PetscMin(1.0, eta));
  std::vector<PetscInt> local_min(static_cast<std::size_t>(column_count),
                                  axis_length);
  std::vector<PetscInt> local_max(static_cast<std::size_t>(column_count), -1);
  std::vector<PetscInt> global_min(static_cast<std::size_t>(column_count),
                                   axis_length);
  std::vector<PetscInt> global_max(static_cast<std::size_t>(column_count), -1);
  std::vector<PetscReal> local_peak(static_cast<std::size_t>(column_count),
                                    0.0);
  std::vector<PetscReal> global_peak(static_cast<std::size_t>(column_count),
                                     0.0);

  PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        const PetscReal rho_value = PetscRealPart(r[k][j][i]);
        if (!h8_mask_is_active(mask_value) || rho_value < threshold) continue;
        const PetscInt id = h8_draft_column_id(axis, i, j, k, ex, ey, ez);
        const PetscInt coord = h8_draft_axis_coordinate(axis, i, j, k);
        local_min[static_cast<std::size_t>(id)] =
            PetscMin(local_min[static_cast<std::size_t>(id)], coord);
        local_max[static_cast<std::size_t>(id)] =
            PetscMax(local_max[static_cast<std::size_t>(id)], coord);
        local_peak[static_cast<std::size_t>(id)] =
            PetscMax(local_peak[static_cast<std::size_t>(id)], rho_value);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));

  PetscCallMPI(MPI_Allreduce(local_min.data(), global_min.data(), reduce_count,
                             MPIU_INT, MPI_MIN, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_max.data(), global_max.data(), reduce_count,
                             MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_peak.data(), global_peak.data(), reduce_count,
                             MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));

  PetscCall(DMDAVecGetArray(eda, rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (!h8_mask_is_active(mask_value)) continue;
        const PetscInt id = h8_draft_column_id(axis, i, j, k, ex, ey, ez);
        const PetscInt coord = h8_draft_axis_coordinate(axis, i, j, k);
        const PetscInt axis_min = global_min[static_cast<std::size_t>(id)];
        const PetscInt axis_max = global_max[static_cast<std::size_t>(id)];
        const bool fill_split = (axis_min <= coord && coord <= axis_max);
        const bool fill_plus = (axis_min < axis_length && coord >= axis_min);
        const bool fill_minus = (axis_max >= 0 && coord <= axis_max);
        const bool should_fill =
            sign > 0 ? fill_plus : (sign < 0 ? fill_minus : fill_split);
        if (should_fill) {
          // sign=0 是 split 闭包；sign>0/<0 是向正/负开模侧的保守单向闭包。
          r[k][j][i] = PetscMax(PetscRealPart(r[k][j][i]),
                                global_peak[static_cast<std::size_t>(id)]);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(eda, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  return 0;
}

PetscErrorCode collect_h8_effective_draft_axes(
    const OptimizerOptions &options, std::vector<H8DraftDirection> *effective) {
  std::vector<H8DraftDirection> dirs;
  bool processed_axis[3] = {false, false, false};

  effective->clear();
  if (!options.z_draft_closure) return 0;
  PetscCall(parse_h8_draft_axes(options.draft_axes, &dirs));
  if (dirs.empty()) return 0;

  for (const H8DraftDirection &dir : dirs) {
    const PetscInt axis = dir.axis;
    if (processed_axis[axis]) continue;

    bool has_plus = false;
    bool has_minus = false;
    bool has_split = false;
    for (const H8DraftDirection &same_axis : dirs) {
      if (same_axis.axis != axis) continue;
      has_plus = has_plus || same_axis.sign > 0;
      has_minus = has_minus || same_axis.sign < 0;
      has_split = has_split || same_axis.sign == 0;
    }

    PetscInt effective_sign = 0;
    if (!has_split && !(has_plus && has_minus)) {
      effective_sign = has_plus ? 1 : -1;
    }

    H8DraftDirection effective_dir;
    effective_dir.axis = axis;
    effective_dir.sign = effective_sign;
    effective->push_back(effective_dir);
    processed_axis[axis] = true;
  }
  return 0;
}

PetscErrorCode apply_h8_draft_closure(DM eda, Vec mask,
                                      const OptimizerOptions &options,
                                      Vec rho) {
  std::vector<H8DraftDirection> dirs;
  PetscCall(collect_h8_effective_draft_axes(options, &dirs));
  for (const H8DraftDirection &dir : dirs) {
    PetscCall(apply_axis_draft_closure(eda, mask, options.z_draft_eta,
                                       dir.axis, dir.sign, rho));
  }
  return 0;
}
#endif

PetscErrorCode apply_h8_draft_closure(DM eda, Vec mask,
                                      const OptimizerOptions &options,
                                      Vec rho) {
  PetscCall(apply_draft_closure(eda, mask, options, rho));
  return 0;
}

PetscErrorCode h8_uses_matlab_z_projection(const OptimizerOptions &options,
                                           PetscBool *use_projection) {
  std::vector<DraftDirection> dirs;
  *use_projection = PETSC_FALSE;
  if (!options.z_draft_closure) return 0;
  if (!options.matlab_z_projection) return 0;
  PetscCall(parse_draft_axes(options.draft_axes, &dirs));
  for (const DraftDirection &dir : dirs) {
    if (dir.axis == 2) {
      *use_projection = PETSC_TRUE;
      return 0;
    }
  }
  return 0;
}

PetscReal h8_matlab_cell_coordinate(PetscInt coord, PetscInt count) {
  if (count <= 0) return 0.5;
  const PetscReal raw =
      (static_cast<PetscReal>(coord) + 0.5) / static_cast<PetscReal>(count);
  return 0.05 + 0.90 * raw;
}

PetscReal h8_matlab_cut_value(PetscReal x, PetscReal eta, PetscReal beta) {
  if (beta <= 1.0e-12) return x >= eta ? 1.0 : 0.0;
  eta = PetscMax(0.0, PetscMin(1.0, eta));
  const PetscReal den =
      PetscTanhReal(beta * eta) + PetscTanhReal(beta * (1.0 - eta));
  if (PetscAbsReal(den) <= PETSC_SMALL) return x >= eta ? 1.0 : 0.0;
  return (PetscTanhReal(beta * eta) +
          PetscTanhReal(beta * (x - eta))) /
         den;
}

PetscReal h8_matlab_cut_deta(PetscReal x, PetscReal eta, PetscReal beta) {
  if (beta <= 1.0e-12) return 0.0;
  eta = PetscMax(0.0, PetscMin(1.0, eta));
  const PetscReal te = PetscTanhReal(beta * eta);
  const PetscReal tx = PetscTanhReal(beta * (x - eta));
  const PetscReal t1 = PetscTanhReal(beta * (1.0 - eta));
  const PetscReal u = te + tx;
  const PetscReal v = te + t1;
  if (PetscAbsReal(v) <= PETSC_SMALL) return 0.0;
  const PetscReal se = 1.0 - te * te;
  const PetscReal sx = 1.0 - tx * tx;
  const PetscReal s1 = 1.0 - t1 * t1;
  const PetscReal u_prime = beta * se - beta * sx;
  const PetscReal v_prime = beta * se - beta * s1;
  return (v * u_prime - u * v_prime) / (v * v);
}

PetscReal h8_matlab_fixed_xy_factor(PetscInt i, PetscInt j, PetscInt ex,
                                    PetscInt ey, PetscReal beta) {
  const PetscReal xf = h8_matlab_cell_coordinate(i, ex);
  const PetscReal yf = h8_matlab_cell_coordinate(j, ey);
  const PetscReal xt1 = h8_matlab_cut_value(xf, 0.0, beta);
  const PetscReal xt2 = 1.0 - h8_matlab_cut_value(xf, 1.0, beta);
  const PetscReal yt1 = h8_matlab_cut_value(yf, 0.0, beta);
  const PetscReal yt2 = 1.0 - h8_matlab_cut_value(yf, 1.0, beta);
  return xt1 * xt2 * yt1 * yt2;
}

PetscInt h8_matlab_z_var_id(PetscInt i, PetscInt j, PetscInt ey) {
  return i * ey + j;
}

PetscReal h8_outer_projection_value(PetscReal rho1,
                                    const OptimizerOptions &opt,
                                    PetscReal beta) {
  return h8_smooth_heaviside_value(rho1, PetscMax(1.0e-12, beta),
                                   opt.z_draft_eta);
}

PetscReal h8_outer_projection_derivative(PetscReal rho1,
                                         const OptimizerOptions &opt,
                                         PetscReal beta) {
  return h8_smooth_heaviside_derivative(rho1, PetscMax(1.0e-12, beta),
                                        opt.z_draft_eta);
}

PetscErrorCode gather_h8_z_surface_variables(
    DM eda, Vec rho_design, Vec mask, const OptimizerOptions &opt,
    std::vector<PetscReal> *bottom, std::vector<PetscReal> *top) {
  (void)opt;
  PetscScalar ***r = nullptr, ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  PetscMPIInt reduce_count = 0;

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  const PetscInt column_count = ex * ey;
  PetscCall(PetscMPIIntCast(column_count, &reduce_count));
  std::vector<PetscReal> local_bottom(static_cast<std::size_t>(column_count),
                                      0.0);
  std::vector<PetscReal> local_top(static_cast<std::size_t>(column_count), 0.0);
  bottom->assign(static_cast<std::size_t>(column_count), 0.0);
  top->assign(static_cast<std::size_t>(column_count), 0.0);

  PetscCall(DMDAVecGetArrayRead(eda, rho_design, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    if (k != 0 && k != ez - 1) continue;
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (!h8_mask_is_design(mask_value) &&
            !h8_mask_is_fixed_solid(mask_value)) {
          continue;
        }
        const PetscInt id = h8_matlab_z_var_id(i, j, ey);
        const PetscReal value = PetscMax(
            0.0, PetscMin(1.0, PetscRealPart(r[k][j][i])));
        if (k == 0) {
          local_bottom[static_cast<std::size_t>(id)] = value;
        } else {
          local_top[static_cast<std::size_t>(id)] = value;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, rho_design, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCallMPI(MPI_Allreduce(local_bottom.data(), bottom->data(), reduce_count,
                             MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_top.data(), top->data(), reduce_count,
                             MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));
  return 0;
}

PetscErrorCode apply_h8_matlab_z_projection(
    DM eda, Vec rho_design, Vec mask, const OptimizerOptions &opt,
    const DensityOptions &density_options, PetscReal outer_beta,
    Vec rho_phys) {
  PetscScalar ***m = nullptr, ***out = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  const PetscReal beta = PetscMax(1.0e-12, opt.draft_beta);
  std::vector<PetscReal> bottom;
  std::vector<PetscReal> top;

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCall(gather_h8_z_surface_variables(eda, rho_design, mask, opt,
                                          &bottom, &top));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAVecGetArray(eda, rho_phys, &out));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    const PetscReal zf = h8_matlab_cell_coordinate(k, ez);
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (h8_mask_is_fixed_solid(mask_value)) {
          out[k][j][i] = 1.0;
          continue;
        }
        if (!h8_mask_is_design(mask_value)) {
          out[k][j][i] = density_options.void_density;
          continue;
        }
        const PetscInt id = h8_matlab_z_var_id(i, j, ey);
        const PetscReal eta_lower =
            1.0 - bottom[static_cast<std::size_t>(id)];
        const PetscReal eta_upper = top[static_cast<std::size_t>(id)];
        const PetscReal zt1 = h8_matlab_cut_value(zf, eta_lower, beta);
        const PetscReal zt2 =
            1.0 - h8_matlab_cut_value(zf, eta_upper, beta);
        const PetscReal xy_factor =
            h8_matlab_fixed_xy_factor(i, j, ex, ey, beta);
        const PetscReal rho1 = xy_factor * zt1 * zt2;
        out[k][j][i] = h8_outer_projection_value(rho1, opt, outer_beta);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCall(DMDAVecRestoreArray(eda, rho_phys, &out));
  return 0;
}

PetscErrorCode build_h8_physical_density(
    DM eda, Vec rho_design, Vec mask, Vec filter_denom,
    const DensityOptions &density_options, const OptimizerOptions &opt,
    PetscReal beta, Vec rho_filtered, Vec rho_phys) {
  PetscBool matlab_z_projection = PETSC_FALSE;
  PetscCall(h8_uses_matlab_z_projection(opt, &matlab_z_projection));
  if (matlab_z_projection) {
    PetscCall(apply_h8_matlab_z_projection(
        eda, rho_design, mask, opt, density_options, beta, rho_phys));
    return 0;
  }
  PetscCall(apply_density_filter(eda, rho_design, mask, filter_denom,
                                 opt.filter_radius,
                                 density_options.void_density, rho_filtered));
  PetscCall(apply_h8_heaviside_projection(
      eda, rho_filtered, mask, opt, beta, density_options.void_density,
      rho_phys));
  PetscCall(apply_h8_draft_closure(eda, mask, opt, rho_phys));
  return 0;
}

PetscErrorCode create_h8_z_surface_update_mask(DM eda, Vec mask,
                                               Vec update_mask) {
  PetscScalar ***m = nullptr, ***u = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ez = 0;
  PetscCall(DMDAGetInfo(eda, nullptr, nullptr, nullptr, &ez, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAVecGetArray(eda, update_mask, &u));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        u[k][j][i] =
            ((h8_mask_is_design(mask_value) ||
              h8_mask_is_fixed_solid(mask_value)) &&
             (k == 0 || k == ez - 1))
                ? 1.0
                : 0.0;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCall(DMDAVecRestoreArray(eda, update_mask, &u));
  return 0;
}

void filter_h8_z_surface_array(const std::vector<PetscReal> &input,
                               PetscInt ex, PetscInt ey, PetscReal radius,
                               std::vector<PetscReal> *output) {
  output->assign(input.size(), 0.0);
  if (radius <= 0.0) {
    *output = input;
    return;
  }

  const PetscInt rr =
      PetscMax(0, static_cast<PetscInt>(PetscCeilReal(radius)));
  for (PetscInt i = 0; i < ex; ++i) {
    for (PetscInt j = 0; j < ey; ++j) {
      PetscReal sum = 0.0;
      PetscReal denom = 0.0;
      for (PetscInt dx = -rr; dx <= rr; ++dx) {
        const PetscInt ii = i + dx;
        if (ii < 0 || ii >= ex) continue;
        for (PetscInt dy = -rr; dy <= rr; ++dy) {
          const PetscInt jj = j + dy;
          if (jj < 0 || jj >= ey) continue;
          const PetscReal w = cone_weight(dx, dy, 0, radius);
          if (w <= 0.0) continue;
          const std::size_t sid =
              static_cast<std::size_t>(h8_matlab_z_var_id(ii, jj, ey));
          sum += w * input[sid];
          denom += w;
        }
      }
      const std::size_t out_id =
          static_cast<std::size_t>(h8_matlab_z_var_id(i, j, ey));
      (*output)[out_id] = denom > 0.0 ? sum / denom : input[out_id];
    }
  }
}

PetscErrorCode apply_h8_matlab_z_surface_sensitivity(
    DM eda, Vec dc_phys, Vec mask, Vec rho_design,
    const DensityOptions &density_options, const OptimizerOptions &opt,
    PetscReal outer_beta, Vec dc_design, Vec dv_design) {
  (void)density_options;
  PetscScalar ***dc = nullptr, ***m = nullptr;
  PetscScalar ***dc_out = nullptr, ***dv_out = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  PetscMPIInt reduce_count = 0;
  const PetscReal beta = PetscMax(1.0e-12, opt.draft_beta);
  std::vector<PetscReal> bottom;
  std::vector<PetscReal> top;

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  const PetscInt column_count = ex * ey;
  PetscCall(PetscMPIIntCast(column_count, &reduce_count));
  std::vector<PetscReal> local_dc_bottom(static_cast<std::size_t>(column_count),
                                         0.0);
  std::vector<PetscReal> local_dc_top(static_cast<std::size_t>(column_count),
                                      0.0);
  std::vector<PetscReal> local_dv_bottom(static_cast<std::size_t>(column_count),
                                         0.0);
  std::vector<PetscReal> local_dv_top(static_cast<std::size_t>(column_count),
                                      0.0);
  std::vector<PetscReal> global_dc_bottom(static_cast<std::size_t>(column_count),
                                          0.0);
  std::vector<PetscReal> global_dc_top(static_cast<std::size_t>(column_count),
                                       0.0);
  std::vector<PetscReal> global_dv_bottom(static_cast<std::size_t>(column_count),
                                          0.0);
  std::vector<PetscReal> global_dv_top(static_cast<std::size_t>(column_count),
                                       0.0);
  std::vector<PetscReal> filtered_dc_bottom;
  std::vector<PetscReal> filtered_dc_top;
  std::vector<PetscReal> filtered_dv_bottom;
  std::vector<PetscReal> filtered_dv_top;

  PetscCall(gather_h8_z_surface_variables(eda, rho_design, mask, opt,
                                          &bottom, &top));
  PetscCall(DMDAVecGetArrayRead(eda, dc_phys, &dc));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    const PetscReal zf = h8_matlab_cell_coordinate(k, ez);
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (!h8_mask_is_design(PetscRealPart(m[k][j][i]))) continue;
        const PetscInt id = h8_matlab_z_var_id(i, j, ey);
        const std::size_t sid = static_cast<std::size_t>(id);
        const PetscReal eta_lower = 1.0 - bottom[sid];
        const PetscReal eta_upper = top[sid];
        const PetscReal zt1 = h8_matlab_cut_value(zf, eta_lower, beta);
        const PetscReal zt2 =
            1.0 - h8_matlab_cut_value(zf, eta_upper, beta);
        const PetscReal xy_factor =
            h8_matlab_fixed_xy_factor(i, j, ex, ey, beta);
        const PetscReal rho1 = xy_factor * zt1 * zt2;
        const PetscReal outer =
            h8_outer_projection_derivative(rho1, opt, outer_beta);
        const PetscReal dzt1_db =
            -h8_matlab_cut_deta(zf, eta_lower, beta);
        const PetscReal dzt2_dt =
            -h8_matlab_cut_deta(zf, eta_upper, beta);
        const PetscReal deriv_bottom = outer * xy_factor * dzt1_db * zt2;
        const PetscReal deriv_top = outer * xy_factor * zt1 * dzt2_dt;
        local_dc_bottom[sid] += PetscRealPart(dc[k][j][i]) * deriv_bottom;
        local_dc_top[sid] += PetscRealPart(dc[k][j][i]) * deriv_top;
        local_dv_bottom[sid] += deriv_bottom;
        local_dv_top[sid] += deriv_top;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, dc_phys, &dc));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));

  PetscCallMPI(MPI_Allreduce(local_dc_bottom.data(), global_dc_bottom.data(),
                             reduce_count, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_dc_top.data(), global_dc_top.data(),
                             reduce_count, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_dv_bottom.data(), global_dv_bottom.data(),
                             reduce_count, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_dv_top.data(), global_dv_top.data(),
                             reduce_count, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));

  filter_h8_z_surface_array(global_dc_bottom, ex, ey, opt.filter_radius,
                            &filtered_dc_bottom);
  filter_h8_z_surface_array(global_dc_top, ex, ey, opt.filter_radius,
                            &filtered_dc_top);
  filter_h8_z_surface_array(global_dv_bottom, ex, ey, opt.filter_radius,
                            &filtered_dv_bottom);
  filter_h8_z_surface_array(global_dv_top, ex, ey, opt.filter_radius,
                            &filtered_dv_top);

  PetscCall(VecSet(dc_design, 0.0));
  PetscCall(VecSet(dv_design, 0.0));
  PetscCall(DMDAVecGetArray(eda, dc_design, &dc_out));
  PetscCall(DMDAVecGetArray(eda, dv_design, &dv_out));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    if (k != 0 && k != ez - 1) continue;
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscInt id = h8_matlab_z_var_id(i, j, ey);
        const std::size_t sid = static_cast<std::size_t>(id);
        if (k == 0) {
          dc_out[k][j][i] = filtered_dc_bottom[sid];
          dv_out[k][j][i] = PetscMax(0.0, filtered_dv_bottom[sid]);
        } else {
          dc_out[k][j][i] = filtered_dc_top[sid];
          dv_out[k][j][i] = PetscMax(0.0, filtered_dv_top[sid]);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(eda, dc_design, &dc_out));
  PetscCall(DMDAVecRestoreArray(eda, dv_design, &dv_out));
  return 0;
}

PetscReal h8_clamp(PetscReal value, PetscReal lo, PetscReal hi) {
  return PetscMax(lo, PetscMin(hi, value));
}

H8VectorStats h8_vector_stats(const std::vector<PetscReal> &values) {
  H8VectorStats stats;
  if (values.empty()) return stats;
  stats.min = values.front();
  stats.max = values.front();
  for (const PetscReal value : values) {
    stats.min = PetscMin(stats.min, value);
    stats.max = PetscMax(stats.max, value);
    stats.sum += value;
    stats.l1 += PetscAbsReal(value);
  }
  return stats;
}

void round_h8_mma_sensitivity_like_matlab(std::vector<PetscReal> *values) {
  PetscReal max_abs = 0.0;
  for (const PetscReal value : *values) {
    max_abs = PetscMax(max_abs, PetscAbsReal(value));
  }
  if (max_abs <= 0.0) return;
  const PetscReal digits =
      5.0 - std::floor(std::log10(static_cast<double>(max_abs)));
  const PetscReal scale = std::pow(10.0, static_cast<double>(digits));
  if (!(scale > 0.0) || !std::isfinite(static_cast<double>(scale))) return;
  for (PetscReal &value : *values) {
    value = std::round(static_cast<double>(value * scale)) / scale;
  }
}

PetscErrorCode write_h8_matlab_z_mma_variables(
    DM eda, Vec mask, const OptimizerOptions &opt,
    const std::vector<PetscReal> &x, Vec rho_design,
    PetscReal *max_change) {
  PetscScalar ***r = nullptr, ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  PetscReal local_change = 0.0;

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  const PetscInt column_count = ex * ey;
  const PetscInt x_face_count = ey * ez;
  const PetscInt y_face_count = ex * ez;
  const PetscInt z_offset = 2 * x_face_count + 2 * y_face_count;
  const PetscInt full_count = z_offset + 2 * column_count;
  const PetscBool full_mma =
      x.size() == static_cast<std::size_t>(full_count) ? PETSC_TRUE
                                                       : PETSC_FALSE;
  PetscCheck(full_mma || x.size() == static_cast<std::size_t>(2 * column_count),
             PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ,
             "MATLAB z MMA variable vector has the wrong length");
  PetscCall(DMDAVecGetArray(eda, rho_design, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    if (k != 0 && k != ez - 1) continue;
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        const PetscReal old_value = PetscRealPart(r[k][j][i]);
        PetscReal new_value = old_value;
        if (h8_mask_is_design(mask_value) ||
            h8_mask_is_fixed_solid(mask_value)) {
          const PetscInt id = h8_matlab_z_var_id(i, j, ey);
          if (k == 0) {
            const PetscInt var_id = full_mma ? z_offset + id : id;
            new_value = 1.0 - x[static_cast<std::size_t>(var_id)];
          } else {
            const PetscInt var_id =
                full_mma ? z_offset + column_count + id : column_count + id;
            new_value =
                x[static_cast<std::size_t>(var_id)];
          }
          new_value = h8_clamp(new_value, 0.0, 1.0);
        }
        r[k][j][i] = new_value;
        local_change =
            PetscMax(local_change, PetscAbsReal(new_value - old_value));
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(eda, rho_design, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCallMPI(MPI_Allreduce(&local_change, max_change, 1, MPIU_REAL, MPI_MAX,
                             PETSC_COMM_WORLD));
  (void)opt;
  return 0;
}

void filter_h8_mma_face_block(const std::vector<PetscReal> &input,
                              PetscInt offset, PetscInt n1, PetscInt n2,
                              PetscReal radius,
                              std::vector<PetscReal> *output) {
  if (radius <= 0.0) {
    for (PetscInt a = 0; a < n1; ++a) {
      for (PetscInt b = 0; b < n2; ++b) {
        const std::size_t id = static_cast<std::size_t>(offset + a * n2 + b);
        (*output)[id] = input[id];
      }
    }
    return;
  }
  const PetscInt rr =
      PetscMax(0, static_cast<PetscInt>(PetscCeilReal(radius) - 1));
  for (PetscInt a = 0; a < n1; ++a) {
    for (PetscInt b = 0; b < n2; ++b) {
      PetscReal sum = 0.0;
      PetscReal denom = 0.0;
      for (PetscInt da = -rr; da <= rr; ++da) {
        const PetscInt aa = a + da;
        if (aa < 0 || aa >= n1) continue;
        for (PetscInt db = -rr; db <= rr; ++db) {
          const PetscInt bb = b + db;
          if (bb < 0 || bb >= n2) continue;
          const PetscReal w =
              PetscMax(0.0, radius - PetscSqrtReal(da * da + db * db));
          if (w <= 0.0) continue;
          const std::size_t src =
              static_cast<std::size_t>(offset + aa * n2 + bb);
          sum += w * input[src];
          denom += w;
        }
      }
      const std::size_t dst = static_cast<std::size_t>(offset + a * n2 + b);
      (*output)[dst] = denom > 0.0 ? sum / denom : input[dst];
    }
  }
}

void filter_h8_matlab_full_mma_sensitivity(
    const std::vector<PetscReal> &input, PetscInt ex, PetscInt ey, PetscInt ez,
    PetscReal radius, std::vector<PetscReal> *output) {
  const PetscInt x_face_count = ey * ez;
  const PetscInt y_face_count = ex * ez;
  const PetscInt z_face_count = ex * ey;
  output->assign(input.size(), 0.0);
  filter_h8_mma_face_block(input, 0, ey, ez, radius, output);
  filter_h8_mma_face_block(input, x_face_count, ey, ez, radius, output);
  filter_h8_mma_face_block(input, 2 * x_face_count, ex, ez, radius, output);
  filter_h8_mma_face_block(input, 2 * x_face_count + y_face_count, ex, ez,
                           radius, output);
  filter_h8_mma_face_block(input, 2 * x_face_count + 2 * y_face_count, ex, ey,
                           radius, output);
  filter_h8_mma_face_block(input,
                           2 * x_face_count + 2 * y_face_count + z_face_count,
                           ex, ey, radius, output);
}

PetscErrorCode collect_h8_matlab_full_mma_data(
    DM eda, Vec rho_design, Vec mask, Vec dc_phys,
    const DensityOptions &density_options, const OptimizerOptions &opt,
    PetscReal outer_beta, std::vector<PetscReal> *x,
    std::vector<PetscReal> *df0dx, std::vector<PetscReal> *dfdx) {
  (void)density_options;
  PetscScalar ***dc = nullptr, ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  PetscMPIInt reduce_count = 0;
  std::vector<PetscReal> bottom;
  std::vector<PetscReal> top;

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  const PetscInt x_face_count = ey * ez;
  const PetscInt y_face_count = ex * ez;
  const PetscInt z_face_count = ex * ey;
  const PetscInt y_offset = 2 * x_face_count;
  const PetscInt z_offset = y_offset + 2 * y_face_count;
  const PetscInt n = z_offset + 2 * z_face_count;
  const PetscReal inv_total =
      1.0 / static_cast<PetscReal>(PetscMax(1, ex * ey * ez));
  const PetscReal beta = PetscMax(1.0e-12, opt.draft_beta);
  PetscCall(PetscMPIIntCast(n, &reduce_count));
  PetscCall(gather_h8_z_surface_variables(eda, rho_design, mask, opt,
                                          &bottom, &top));

  x->assign(static_cast<std::size_t>(n), 0.0);
  std::vector<PetscReal> local_df0dx(static_cast<std::size_t>(n), 0.0);
  std::vector<PetscReal> local_dfdx(static_cast<std::size_t>(n), 0.0);
  std::vector<PetscReal> global_df0dx(static_cast<std::size_t>(n), 0.0);
  std::vector<PetscReal> global_dfdx(static_cast<std::size_t>(n), 0.0);

  for (PetscInt j = 0; j < ey; ++j) {
    for (PetscInt k = 0; k < ez; ++k) {
      const PetscInt id = j * ez + k;
      (*x)[static_cast<std::size_t>(id)] = 0.0;
      (*x)[static_cast<std::size_t>(x_face_count + id)] = 1.0;
    }
  }
  for (PetscInt i = 0; i < ex; ++i) {
    for (PetscInt k = 0; k < ez; ++k) {
      const PetscInt id = i * ez + k;
      (*x)[static_cast<std::size_t>(y_offset + id)] = 0.0;
      (*x)[static_cast<std::size_t>(y_offset + y_face_count + id)] = 1.0;
    }
  }
  for (PetscInt i = 0; i < ex; ++i) {
    for (PetscInt j = 0; j < ey; ++j) {
      const PetscInt id = h8_matlab_z_var_id(i, j, ey);
      const std::size_t sid = static_cast<std::size_t>(id);
      (*x)[static_cast<std::size_t>(z_offset + id)] = 1.0 - bottom[sid];
      (*x)[static_cast<std::size_t>(z_offset + z_face_count + id)] = top[sid];
    }
  }

  PetscCall(DMDAVecGetArrayRead(eda, dc_phys, &dc));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    const PetscReal zf = h8_matlab_cell_coordinate(k, ez);
    for (PetscInt j = ys; j < ys + ym; ++j) {
      const PetscReal yf = h8_matlab_cell_coordinate(j, ey);
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (!h8_mask_is_design(PetscRealPart(m[k][j][i]))) continue;
        const PetscReal xf = h8_matlab_cell_coordinate(i, ex);
        const PetscInt zid = h8_matlab_z_var_id(i, j, ey);
        const std::size_t zsid = static_cast<std::size_t>(zid);
        const PetscReal eta1_z = 1.0 - bottom[zsid];
        const PetscReal eta2_z = top[zsid];
        const PetscReal xt1 = h8_matlab_cut_value(xf, 0.0, beta);
        const PetscReal xt2 = 1.0 - h8_matlab_cut_value(xf, 1.0, beta);
        const PetscReal yt1 = h8_matlab_cut_value(yf, 0.0, beta);
        const PetscReal yt2 = 1.0 - h8_matlab_cut_value(yf, 1.0, beta);
        const PetscReal zt1 = h8_matlab_cut_value(zf, eta1_z, beta);
        const PetscReal zt2 = 1.0 - h8_matlab_cut_value(zf, eta2_z, beta);
        const PetscReal rho1 = xt1 * xt2 * yt1 * yt2 * zt1 * zt2;
        const PetscReal outer =
            h8_outer_projection_derivative(rho1, opt, outer_beta);
        const PetscReal dc_value = PetscRealPart(dc[k][j][i]);
        const PetscInt x_var = j * ez + k;
        const PetscInt y_var = i * ez + k;
        const PetscReal d_xt1 = h8_matlab_cut_deta(xf, 0.0, beta);
        const PetscReal d_xt2 = -h8_matlab_cut_deta(xf, 1.0, beta);
        const PetscReal d_yt1 = h8_matlab_cut_deta(yf, 0.0, beta);
        const PetscReal d_yt2 = -h8_matlab_cut_deta(yf, 1.0, beta);
        const PetscReal d_zt1 = h8_matlab_cut_deta(zf, eta1_z, beta);
        const PetscReal d_zt2 = -h8_matlab_cut_deta(zf, eta2_z, beta);
        const PetscReal derivs[6] = {
            outer * d_xt1 * xt2 * yt1 * yt2 * zt1 * zt2,
            outer * d_xt2 * xt1 * yt1 * yt2 * zt1 * zt2,
            outer * d_yt1 * xt1 * xt2 * yt2 * zt1 * zt2,
            outer * d_yt2 * xt1 * xt2 * yt1 * zt1 * zt2,
            outer * d_zt1 * xt1 * xt2 * yt1 * yt2 * zt2,
            outer * d_zt2 * xt1 * xt2 * yt1 * yt2 * zt1};
        const PetscInt ids[6] = {
            x_var,
            x_face_count + x_var,
            y_offset + y_var,
            y_offset + y_face_count + y_var,
            z_offset + zid,
            z_offset + z_face_count + zid};
        for (PetscInt q = 0; q < 6; ++q) {
          const std::size_t sid = static_cast<std::size_t>(ids[q]);
          local_df0dx[sid] += dc_value * derivs[q];
          local_dfdx[sid] += inv_total * derivs[q];
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, dc_phys, &dc));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCallMPI(MPI_Allreduce(local_df0dx.data(), global_df0dx.data(),
                             reduce_count, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_dfdx.data(), global_dfdx.data(),
                             reduce_count, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  filter_h8_matlab_full_mma_sensitivity(global_df0dx, ex, ey, ez,
                                        opt.filter_radius, df0dx);
  filter_h8_matlab_full_mma_sensitivity(global_dfdx, ex, ey, ez,
                                        opt.filter_radius, dfdx);
  return 0;
}

void h8_mma_xyz_scalar(const std::vector<PetscReal> &L,
                       const std::vector<PetscReal> &U,
                       const std::vector<PetscReal> &alpha,
                       const std::vector<PetscReal> &beta,
                       const std::vector<PetscReal> &p0,
                       const std::vector<PetscReal> &q0,
                       const std::vector<PetscReal> &pij,
                       const std::vector<PetscReal> &qij,
                       PetscReal lam_in, std::vector<PetscReal> *x,
                       PetscReal *y, PetscReal *z) {
  const PetscReal lam = PetscMax(0.0, lam_in);
  const PetscInt n = static_cast<PetscInt>(L.size());
  x->assign(static_cast<std::size_t>(n), 0.0);
  *y = PetscMax(0.0, lam - 100.0);
  *z = 0.0;
  for (PetscInt j = 0; j < n; ++j) {
    const std::size_t sid = static_cast<std::size_t>(j);
    const PetscReal pjlam = PetscMax(0.0, p0[sid] + pij[sid] * lam);
    const PetscReal qjlam = PetscMax(0.0, q0[sid] + qij[sid] * lam);
    const PetscReal sp = std::sqrt(static_cast<double>(pjlam));
    const PetscReal sq = std::sqrt(static_cast<double>(qjlam));
    const PetscReal denom = sp + sq;
    PetscReal value =
        denom > 0.0 ? (sp * L[sid] + sq * U[sid]) / denom
                    : 0.5 * (alpha[sid] + beta[sid]);
    (*x)[sid] = h8_clamp(value, alpha[sid], beta[sid]);
  }
}

PetscReal h8_mma_constraint_value(const std::vector<PetscReal> &L,
                                  const std::vector<PetscReal> &U,
                                  const std::vector<PetscReal> &pij,
                                  const std::vector<PetscReal> &qij,
                                  const std::vector<PetscReal> &x) {
  PetscReal gx = 0.0;
  const PetscInt n = static_cast<PetscInt>(x.size());
  for (PetscInt j = 0; j < n; ++j) {
    const std::size_t sid = static_cast<std::size_t>(j);
    const PetscReal ux = PetscMax(1.0e-30, U[sid] - x[sid]);
    const PetscReal xl = PetscMax(1.0e-30, x[sid] - L[sid]);
    gx += pij[sid] / ux + qij[sid] / xl;
  }
  return gx;
}

PetscReal h8_mma_dual_hessian_scalar(
    const std::vector<PetscReal> &L, const std::vector<PetscReal> &U,
    const std::vector<PetscReal> &alpha,
    const std::vector<PetscReal> &beta,
    const std::vector<PetscReal> &p0, const std::vector<PetscReal> &q0,
    const std::vector<PetscReal> &pij, const std::vector<PetscReal> &qij,
    const std::vector<PetscReal> &x, PetscReal lam, PetscReal mu) {
  PetscReal hess = 0.0;
  const PetscInt n = static_cast<PetscInt>(x.size());
  for (PetscInt j = 0; j < n; ++j) {
    const std::size_t sid = static_cast<std::size_t>(j);
    const PetscReal ux = PetscMax(1.0e-30, U[sid] - x[sid]);
    const PetscReal xl = PetscMax(1.0e-30, x[sid] - L[sid]);
    const PetscReal ux2 = ux * ux;
    const PetscReal xl2 = xl * xl;
    const PetscReal pjlam = PetscMax(0.0, p0[sid] + pij[sid] * lam);
    const PetscReal qjlam = PetscMax(0.0, q0[sid] + qij[sid] * lam);
    const PetscReal pq = pij[sid] / ux2 - qij[sid] / xl2;
    const PetscReal df2 =
        2.0 * pjlam / (ux2 * ux) + 2.0 * qjlam / (xl2 * xl);
    const PetscReal sp = std::sqrt(static_cast<double>(pjlam));
    const PetscReal sq = std::sqrt(static_cast<double>(qjlam));
    const PetscReal denom = sp + sq;
    const PetscReal xp =
        denom > 0.0 ? (sp * L[sid] + sq * U[sid]) / denom
                    : 0.5 * (alpha[sid] + beta[sid]);
    PetscReal df2_inv = df2 > 0.0 ? -1.0 / df2 : 0.0;
    if (xp < alpha[sid] || xp > beta[sid]) df2_inv = 0.0;
    hess += pq * df2_inv * pq;
  }
  if (lam > 100.0) hess -= 1.0;
  hess -= mu / PetscMax(1.0e-30, lam);
  PetscReal hess_corr = 1.0e-4 * hess;
  if (-hess_corr < 1.0e-7) hess_corr = -1.0e-7;
  hess += hess_corr;
  if (PetscAbsReal(hess) < 1.0e-30) hess = -1.0e-30;
  return hess;
}

void h8_mma_solve_scalar(
    const std::vector<PetscReal> &L, const std::vector<PetscReal> &U,
    const std::vector<PetscReal> &alpha,
    const std::vector<PetscReal> &beta,
    const std::vector<PetscReal> &p0, const std::vector<PetscReal> &q0,
    const std::vector<PetscReal> &pij, const std::vector<PetscReal> &qij,
    PetscReal b, std::vector<PetscReal> *x_out) {
  const PetscInt n = static_cast<PetscInt>(L.size());
  const PetscReal tol =
      1.0e-9 * std::sqrt(static_cast<double>(PetscMax(1, n + 1)));
  PetscReal lam = 50.0;
  PetscReal mu = 1.0;
  PetscReal y = 0.0;
  PetscReal z = 0.0;
  PetscReal epsi = 1.0;
  PetscReal nrI = 1.0;
  std::vector<PetscReal> x;

  while (epsi > tol) {
    PetscInt loop = 0;
    while (nrI > 0.9 * epsi && loop < 100) {
      ++loop;
      h8_mma_xyz_scalar(L, U, alpha, beta, p0, q0, pij, qij, lam, &x, &y, &z);
      const PetscReal gx = h8_mma_constraint_value(L, U, pij, qij, x);
      const PetscReal grad =
          -1.0 * (gx - b - y) - epsi / PetscMax(1.0e-30, lam);
      const PetscReal hess =
          h8_mma_dual_hessian_scalar(L, U, alpha, beta, p0, q0, pij, qij, x,
                                     lam, mu);
      const PetscReal s1 = grad / hess;
      const PetscReal s2 =
          -mu + epsi / PetscMax(1.0e-30, lam) -
          s1 * mu / PetscMax(1.0e-30, lam);
      PetscReal theta = 1.005;
      if (theta < -1.01 * s1 / PetscMax(1.0e-30, lam)) {
        theta = -1.01 * s1 / PetscMax(1.0e-30, lam);
      }
      if (theta < -1.01 * s2 / PetscMax(1.0e-30, mu)) {
        theta = -1.01 * s2 / PetscMax(1.0e-30, mu);
      }
      const PetscReal step = 1.0 / theta;
      lam = PetscMax(1.0e-30, lam + step * s1);
      mu = PetscMax(1.0e-30, mu + step * s2);
      h8_mma_xyz_scalar(L, U, alpha, beta, p0, q0, pij, qij, lam, &x, &y, &z);
      const PetscReal gx_new = h8_mma_constraint_value(L, U, pij, qij, x);
      const PetscReal res1 = -b - y + gx_new + mu;
      const PetscReal res2 = mu * lam - epsi;
      nrI = PetscMax(PetscAbsReal(res1), PetscAbsReal(res2));
    }
    epsi *= 0.1;
  }
  if (x.empty()) {
    h8_mma_xyz_scalar(L, U, alpha, beta, p0, q0, pij, qij, lam, &x, &y, &z);
  }
  *x_out = std::move(x);
}

void h8_matlab_mma_gensub(PetscInt iter, const std::vector<PetscReal> &x,
                          const std::vector<PetscReal> &dfdx,
                          PetscReal gx,
                          const std::vector<PetscReal> &dgdx,
                          PetscReal move, H8MatlabMMAState *state,
                          std::vector<PetscReal> *xmma) {
  const PetscInt n = static_cast<PetscInt>(x.size());
  if (state->n != n) {
    state->n = n;
    state->xold1.assign(static_cast<std::size_t>(n), 0.0);
    state->xold2.assign(static_cast<std::size_t>(n), 0.0);
    state->L.assign(static_cast<std::size_t>(n), 0.0);
    state->U.assign(static_cast<std::size_t>(n), 0.0);
  }
  std::vector<PetscReal> xmin(static_cast<std::size_t>(n), 0.0);
  std::vector<PetscReal> xmax(static_cast<std::size_t>(n), 1.0);
  std::vector<PetscReal> alpha(static_cast<std::size_t>(n), 0.0);
  std::vector<PetscReal> beta(static_cast<std::size_t>(n), 1.0);
  std::vector<PetscReal> p0(static_cast<std::size_t>(n), 0.0);
  std::vector<PetscReal> q0(static_cast<std::size_t>(n), 0.0);
  std::vector<PetscReal> pij(static_cast<std::size_t>(n), 0.0);
  std::vector<PetscReal> qij(static_cast<std::size_t>(n), 0.0);

  for (PetscInt j = 0; j < n; ++j) {
    const std::size_t sid = static_cast<std::size_t>(j);
    xmin[sid] = PetscMax(0.0, x[sid] - move);
    xmax[sid] = PetscMin(1.0, x[sid] + move);
  }

  if (iter < 3) {
    for (PetscInt j = 0; j < n; ++j) {
      const std::size_t sid = static_cast<std::size_t>(j);
      const PetscReal width = xmax[sid] - xmin[sid];
      state->L[sid] = x[sid] - 0.5 * width;
      state->U[sid] = x[sid] + 0.5 * width;
    }
  } else {
    for (PetscInt j = 0; j < n; ++j) {
      const std::size_t sid = static_cast<std::size_t>(j);
      PetscReal gamma = 1.0;
      const PetscReal help =
          (x[sid] - state->xold1[sid]) *
          (state->xold1[sid] - state->xold2[sid]);
      if (help < 0.0) gamma = 0.7;
      else if (help > 0.0) gamma = 1.2;
      state->L[sid] = x[sid] - gamma * (state->xold1[sid] - state->L[sid]);
      state->U[sid] = x[sid] + gamma * (state->U[sid] - state->xold1[sid]);
      const PetscReal xmi = PetscMax(1.0e-5, xmax[sid] - xmin[sid]);
      state->L[sid] = PetscMax(state->L[sid], x[sid] - 10.0 * xmi);
      state->L[sid] = PetscMin(state->L[sid], x[sid] - 0.01 * xmi);
      state->U[sid] = PetscMax(state->U[sid], x[sid] + 0.01 * xmi);
      state->U[sid] = PetscMin(state->U[sid], x[sid] + 10.0 * xmi);
    }
  }

  PetscReal b = -gx;
  for (PetscInt j = 0; j < n; ++j) {
    const std::size_t sid = static_cast<std::size_t>(j);
    alpha[sid] = PetscMax(xmin[sid], 0.9 * state->L[sid] + 0.1 * x[sid]);
    beta[sid] = PetscMin(xmax[sid], 0.9 * state->U[sid] + 0.1 * x[sid]);
    const PetscReal ux = PetscMax(1.0e-30, state->U[sid] - x[sid]);
    const PetscReal xl = PetscMax(1.0e-30, x[sid] - state->L[sid]);
    const PetscReal inv_width =
        1.0 / PetscMax(1.0e-30, state->U[sid] - state->L[sid]);
    const PetscReal dfdxp = PetscMax(0.0, dfdx[sid]);
    const PetscReal dfdxm = PetscMax(0.0, -dfdx[sid]);
    const PetscReal dgdxp = PetscMax(0.0, dgdx[sid]);
    const PetscReal dgdxm = PetscMax(0.0, -dgdx[sid]);
    p0[sid] = ux * ux *
              (dfdxp + 0.001 * PetscAbsReal(dfdx[sid]) +
               1.0e-6 * inv_width);
    q0[sid] = xl * xl *
              (dfdxm + 0.001 * PetscAbsReal(dfdx[sid]) +
               1.0e-6 * inv_width);
    pij[sid] = ux * ux *
               (dgdxp + 0.001 * PetscAbsReal(dgdx[sid]) +
                1.0e-6 * inv_width);
    qij[sid] = xl * xl *
               (dgdxm + 0.001 * PetscAbsReal(dgdx[sid]) +
                1.0e-6 * inv_width);
    b += pij[sid] / ux + qij[sid] / xl;
  }

  h8_mma_solve_scalar(state->L, state->U, alpha, beta, p0, q0, pij, qij, b,
                      xmma);
  state->xold2 = state->xold1;
  state->xold1 = x;
}

PetscReal h8_matlab_mma_move_for_iter(const OptimizerOptions &opt,
                                      PetscInt iter) {
  PetscReal move = PetscMax(0.01 - 0.0001 * static_cast<PetscReal>(iter),
                            0.001);
  if (opt.move > 0.0 && opt.move < 0.05) move = PetscMin(move, opt.move);
  return move;
}

PetscErrorCode h8_mma_update_matlab_z(
    DM eda, Vec rho_design, Vec mask, Vec dc_phys,
    const DensityOptions &density_options, const OptimizerOptions &opt,
    PetscReal outer_beta, PetscInt iter, PetscReal compliance,
    PetscReal volume, H8MatlabMMAState *mma_state,
    H8MatlabMMADiagnostics *diag, PetscReal *max_change) {
  std::vector<PetscReal> x;
  std::vector<PetscReal> df0dx;
  std::vector<PetscReal> dfdx;
  std::vector<PetscReal> xmma;
  PetscCall(collect_h8_matlab_full_mma_data(
      eda, rho_design, mask, dc_phys, density_options, opt, outer_beta, &x,
      &df0dx, &dfdx));

  if (mma_state->c_scale <= 0.0) {
    mma_state->c_scale = compliance != 0.0 ? 1.0 / compliance : 1.0;
  }
  for (PetscReal &value : df0dx) value *= mma_state->c_scale;
  round_h8_mma_sensitivity_like_matlab(&df0dx);
  round_h8_mma_sensitivity_like_matlab(&dfdx);

  const PetscReal fval = volume - PetscMax(0.9 * volume, opt.volfrac);
  const PetscReal move = h8_matlab_mma_move_for_iter(opt, iter);
  h8_matlab_mma_gensub(iter, x, df0dx, fval, dfdx, move, mma_state, &xmma);
  if (diag != nullptr) {
    diag->f0val = compliance * mma_state->c_scale;
    diag->fval = fval;
    diag->c_scale = mma_state->c_scale;
    diag->move = move;
    diag->beta1 = opt.draft_beta;
    diag->beta2 = outer_beta;
    diag->x = h8_vector_stats(x);
    diag->xmma = h8_vector_stats(xmma);
    diag->df0dx = h8_vector_stats(df0dx);
    diag->dfdx = h8_vector_stats(dfdx);
    diag->x_values = x;
    diag->xmma_values = xmma;
    diag->df0dx_values = df0dx;
    diag->dfdx_values = dfdx;
    diag->x_change = 0.0;
    const PetscInt n =
        static_cast<PetscInt>(PetscMin(x.size(), xmma.size()));
    for (PetscInt q = 0; q < n; ++q) {
      const std::size_t sid = static_cast<std::size_t>(q);
      diag->x_change =
          PetscMax(diag->x_change, PetscAbsReal(xmma[sid] - x[sid]));
    }
  }
  PetscCall(write_h8_matlab_z_mma_variables(eda, mask, opt, xmma, rho_design,
                                            max_change));
  return 0;
}

PetscErrorCode h8_scatter_natural_to_zero(DM eda, Vec global, Vec *seq) {
  Vec natural = nullptr;
  VecScatter scatter = nullptr;
  PetscCall(DMDACreateNaturalVector(eda, &natural));
  PetscCall(DMDAGlobalToNaturalBegin(eda, global, INSERT_VALUES, natural));
  PetscCall(DMDAGlobalToNaturalEnd(eda, global, INSERT_VALUES, natural));
  PetscCall(VecScatterCreateToZero(natural, &scatter, seq));
  PetscCall(VecScatterBegin(scatter, natural, *seq, INSERT_VALUES,
                            SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, natural, *seq, INSERT_VALUES,
                          SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecDestroy(&natural));
  return 0;
}

PetscErrorCode h8_mma_update_general(
    DM eda, Vec rho_design, Vec update_mask, Vec dc_design, Vec dv_design,
    const OptimizerOptions &opt, PetscReal projection_beta, PetscInt iter,
    PetscReal compliance, PetscReal volume, H8MatlabMMAState *mma_state,
    H8MatlabMMADiagnostics *diag, PetscReal *max_change) {
  MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(eda));
  PetscMPIInt rank = 0;
  PetscMPIInt bcast_count = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  Vec rho_seq = nullptr, mask_seq = nullptr, dc_seq = nullptr, dv_seq = nullptr;
  std::vector<PetscReal> xmma_full;

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  const PetscInt full_count = ex * ey * ez;
  PetscCall(PetscMPIIntCast(full_count, &bcast_count));

  PetscCall(h8_scatter_natural_to_zero(eda, rho_design, &rho_seq));
  PetscCall(h8_scatter_natural_to_zero(eda, update_mask, &mask_seq));
  PetscCall(h8_scatter_natural_to_zero(eda, dc_design, &dc_seq));
  PetscCall(h8_scatter_natural_to_zero(eda, dv_design, &dv_seq));

  xmma_full.assign(static_cast<std::size_t>(full_count), 0.0);
  if (rank == 0) {
    const PetscScalar *rho = nullptr;
    const PetscScalar *mask = nullptr;
    const PetscScalar *dc = nullptr;
    const PetscScalar *dv = nullptr;
    std::vector<PetscReal> x;
    std::vector<PetscReal> df0dx;
    std::vector<PetscReal> dfdx;
    std::vector<PetscReal> xmma;

    PetscCall(VecGetArrayRead(rho_seq, &rho));
    PetscCall(VecGetArrayRead(mask_seq, &mask));
    PetscCall(VecGetArrayRead(dc_seq, &dc));
    PetscCall(VecGetArrayRead(dv_seq, &dv));
    for (PetscInt q = 0; q < full_count; ++q) {
      const std::size_t sid = static_cast<std::size_t>(q);
      const PetscReal rho_value =
          PetscMax(0.0, PetscMin(1.0, PetscRealPart(rho[q])));
      xmma_full[sid] = rho_value;
      if (!h8_mask_is_design(PetscRealPart(mask[q]))) continue;
      x.push_back(rho_value);
      df0dx.push_back(PetscRealPart(dc[q]));
      dfdx.push_back(PetscRealPart(dv[q]));
    }

    if (mma_state->c_scale <= 0.0) {
      mma_state->c_scale =
          compliance != 0.0 ? 1.0 / PetscAbsReal(compliance) : 1.0;
    }
    for (PetscReal &value : df0dx) value *= mma_state->c_scale;
    round_h8_mma_sensitivity_like_matlab(&df0dx);
    round_h8_mma_sensitivity_like_matlab(&dfdx);

    const PetscReal fval = volume - PetscMax(0.9 * volume, opt.volfrac);
    const PetscReal move = h8_matlab_mma_move_for_iter(opt, iter);
    h8_matlab_mma_gensub(iter, x, df0dx, fval, dfdx, move, mma_state,
                         &xmma);

    PetscInt design_id = 0;
    PetscReal local_change = 0.0;
    for (PetscInt q = 0; q < full_count; ++q) {
      if (!h8_mask_is_design(PetscRealPart(mask[q]))) continue;
      const std::size_t did = static_cast<std::size_t>(design_id++);
      const PetscReal new_value =
          did < xmma.size() ? PetscMax(0.0, PetscMin(1.0, xmma[did]))
                            : PetscRealPart(rho[q]);
      xmma_full[static_cast<std::size_t>(q)] = new_value;
      local_change =
          PetscMax(local_change, PetscAbsReal(new_value - PetscRealPart(rho[q])));
    }

    if (diag != nullptr) {
      diag->f0val = compliance * mma_state->c_scale;
      diag->fval = fval;
      diag->c_scale = mma_state->c_scale;
      diag->move = move;
      diag->beta1 = opt.draft_beta;
      diag->beta2 = projection_beta;
      diag->x = h8_vector_stats(x);
      diag->xmma = h8_vector_stats(xmma);
      diag->df0dx = h8_vector_stats(df0dx);
      diag->dfdx = h8_vector_stats(dfdx);
      diag->x_values = x;
      diag->xmma_values = xmma;
      diag->df0dx_values = df0dx;
      diag->dfdx_values = dfdx;
      diag->x_change = local_change;
    }

    PetscCall(VecRestoreArrayRead(rho_seq, &rho));
    PetscCall(VecRestoreArrayRead(mask_seq, &mask));
    PetscCall(VecRestoreArrayRead(dc_seq, &dc));
    PetscCall(VecRestoreArrayRead(dv_seq, &dv));
  }

  PetscCallMPI(MPI_Bcast(xmma_full.data(), bcast_count, MPIU_REAL, 0, comm));

  PetscScalar ***r = nullptr;
  PetscScalar ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_change = 0.0;
  PetscReal global_change = 0.0;
  PetscCall(DMDAVecGetArray(eda, rho_design, &r));
  PetscCall(DMDAVecGetArrayRead(eda, update_mask, &m));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (!h8_mask_is_design(PetscRealPart(m[k][j][i]))) continue;
        const PetscInt id = i + ex * (j + ey * k);
        const PetscReal old_value = PetscRealPart(r[k][j][i]);
        const PetscReal new_value =
            xmma_full[static_cast<std::size_t>(id)];
        r[k][j][i] = new_value;
        local_change =
            PetscMax(local_change, PetscAbsReal(new_value - old_value));
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(eda, rho_design, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, update_mask, &m));
  PetscCallMPI(MPI_Allreduce(&local_change, &global_change, 1, MPIU_REAL,
                             MPI_MAX, comm));
  *max_change = global_change;

  PetscCall(VecDestroy(&dv_seq));
  PetscCall(VecDestroy(&dc_seq));
  PetscCall(VecDestroy(&mask_seq));
  PetscCall(VecDestroy(&rho_seq));
  return 0;
}

PetscErrorCode apply_sensitivity_filter_adjoint(DM eda, Vec dc_phys, Vec mask,
                                                Vec denom, PetscReal radius,
                                                Vec dc_design) {
  if (radius <= 0.0) {
    PetscScalar ***dc = nullptr, ***m = nullptr, ***out = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscCall(DMDAVecGetArrayRead(eda, dc_phys, &dc));
    PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
    PetscCall(DMDAVecGetArray(eda, dc_design, &out));
    PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          out[k][j][i] = h8_mask_is_design(PetscRealPart(m[k][j][i]))
                             ? dc[k][j][i]
                             : 0.0;
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(eda, dc_phys, &dc));
    PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
    PetscCall(DMDAVecRestoreArray(eda, dc_design, &out));
    return 0;
  }

  Vec local_dc = nullptr, local_mask = nullptr, local_denom = nullptr;
  PetscScalar ***dc = nullptr, ***m = nullptr, ***den = nullptr, ***out = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  const PetscInt rr = PetscMax(0, static_cast<PetscInt>(PetscCeilReal(radius)));

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCall(DMCreateLocalVector(eda, &local_dc));
  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMCreateLocalVector(eda, &local_denom));
  PetscCall(DMGlobalToLocalBegin(eda, dc_phys, INSERT_VALUES, local_dc));
  PetscCall(DMGlobalToLocalEnd(eda, dc_phys, INSERT_VALUES, local_dc));
  PetscCall(DMGlobalToLocalBegin(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, denom, INSERT_VALUES, local_denom));
  PetscCall(DMGlobalToLocalEnd(eda, denom, INSERT_VALUES, local_denom));
  PetscCall(DMDAVecGetArrayRead(eda, local_dc, &dc));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecGetArrayRead(eda, local_denom, &den));
  PetscCall(DMDAVecGetArray(eda, dc_design, &out));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (!h8_mask_is_design(PetscRealPart(m[k][j][i]))) {
          out[k][j][i] = 0.0;
          continue;
        }
        PetscReal sum = 0.0;
        for (PetscInt dz = -rr; dz <= rr; ++dz) {
          const PetscInt kk = k + dz;
          if (kk < 0 || kk >= ez) continue;
          for (PetscInt dy = -rr; dy <= rr; ++dy) {
            const PetscInt jj = j + dy;
            if (jj < 0 || jj >= ey) continue;
            for (PetscInt dx = -rr; dx <= rr; ++dx) {
              const PetscInt ii = i + dx;
              if (ii < 0 || ii >= ex) continue;
              if (!h8_mask_is_design(PetscRealPart(m[kk][jj][ii]))) continue;
              const PetscReal w = cone_weight(dx, dy, dz, radius);
              if (w > 0.0) {
                sum += w * PetscRealPart(dc[kk][jj][ii]) /
                       PetscRealPart(den[kk][jj][ii]);
              }
            }
          }
        }
        out[k][j][i] = sum;
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(eda, local_dc, &dc));
  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecRestoreArrayRead(eda, local_denom, &den));
  PetscCall(DMDAVecRestoreArray(eda, dc_design, &out));
  PetscCall(VecDestroy(&local_dc));
  PetscCall(VecDestroy(&local_mask));
  PetscCall(VecDestroy(&local_denom));
  return 0;
}

PetscErrorCode create_element_mask_and_density(DM eda, const Grid &grid,
                                               const DensityOptions &density_options,
                                               const OptimizerOptions &opt,
                                               Vec *mask, Vec *rho) {
  PetscScalar ***m = nullptr;
  PetscScalar ***r = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ez = 0;
  PetscBool matlab_z_projection = PETSC_FALSE;
  PetscCall(h8_uses_matlab_z_projection(opt, &matlab_z_projection));
  PetscCall(DMDAGetInfo(eda, nullptr, nullptr, nullptr, &ez, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr));
  PetscCall(DMCreateGlobalVector(eda, mask));
  PetscCall(VecDuplicate(*mask, rho));
  PetscCall(DMDAVecGetArray(eda, *mask, &m));
  PetscCall(DMDAVecGetArray(eda, *rho, &r));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscBool forced_solid =
            h8_cell_is_forced_solid(i, j, k, grid, density_options, opt);
        const PetscBool design =
            cell_density(i, j, k, grid, density_options) >=
                    density_options.mask_threshold
                ? PETSC_TRUE
                : PETSC_FALSE;
        if (forced_solid) {
          m[k][j][i] = 2.0;
          r[k][j][i] = 1.0;
        } else if (design) {
          m[k][j][i] = 1.0;
          r[k][j][i] =
              matlab_z_projection
                  ? ((k == 0 || k == ez - 1) ? 1.0 : opt.rho_min)
                  : opt.volfrac;
        } else {
          m[k][j][i] = 0.0;
          r[k][j][i] = density_options.void_density;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(eda, *mask, &m));
  PetscCall(DMDAVecRestoreArray(eda, *rho, &r));
  return 0;
}

PetscErrorCode h8_opt_mult(Mat A, Vec x, Vec y) {
  H8OptContext *ctx = nullptr;
  PetscScalar ****xg = nullptr;
  PetscScalar ****yg = nullptr;
  PetscScalar ***rho = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(MatShellGetContext(A, &ctx));
  const Grid &g = ctx->grid;

  PetscCall(DMGlobalToLocalBegin(ctx->uda, x, INSERT_VALUES, ctx->local_u));
  PetscCall(DMGlobalToLocalEnd(ctx->uda, x, INSERT_VALUES, ctx->local_u));
  PetscCall(DMGlobalToLocalBegin(ctx->eda, ctx->rho, INSERT_VALUES, ctx->local_rho));
  PetscCall(DMGlobalToLocalEnd(ctx->eda, ctx->rho, INSERT_VALUES, ctx->local_rho));
  PetscCall(DMDAVecGetArrayDOFRead(ctx->uda, ctx->local_u, &xg));
  PetscCall(DMDAVecGetArrayDOF(ctx->uda, y, &yg));
  PetscCall(DMDAVecGetArrayRead(ctx->eda, ctx->local_rho, &rho));
  PetscCall(DMDAGetCorners(ctx->uda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        for (PetscInt c = 0; c < 3; ++c) {
          if (h8_is_fixed_node(i, j, k, g, ctx->density_options)) {
            yg[k][j][i][c] = xg[k][j][i][c];
            continue;
          }
          PetscScalar value = 0.0;
          const PetscInt ex0 = PetscMax(0, i - 1);
          const PetscInt ex1 = PetscMin(i, g.nx - 2);
          const PetscInt ey0 = PetscMax(0, j - 1);
          const PetscInt ey1 = PetscMin(j, g.ny - 2);
          const PetscInt ez0 = PetscMax(0, k - 1);
          const PetscInt ez1 = PetscMin(k, g.nz - 2);
          for (PetscInt ez = ez0; ez <= ez1; ++ez) {
            for (PetscInt ey = ey0; ey <= ey1; ++ey) {
              for (PetscInt ex = ex0; ex <= ex1; ++ex) {
                const PetscInt row_node = h8_local_node(i - ex, j - ey, k - ez);
                const PetscInt row = 3 * row_node + c;
                const PetscReal scale = simp_scale(PetscRealPart(rho[ez][ey][ex]),
                                                   ctx->density_options);
                for (PetscInt col_node = 0; col_node < 8; ++col_node) {
                  PetscInt ni = 0, nj = 0, nk = 0;
                  h8_node_coords(ex, ey, ez, col_node, &ni, &nj, &nk);
                  if (h8_is_fixed_node(ni, nj, nk, g, ctx->density_options)) continue;
                  for (PetscInt d = 0; d < 3; ++d) {
                    const PetscInt col = 3 * col_node + d;
                    value += scale * ctx->ke[24 * row + col] * xg[nk][nj][ni][d];
                  }
                }
              }
            }
          }
          yg[k][j][i][c] = value;
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(ctx->uda, ctx->local_u, &xg));
  PetscCall(DMDAVecRestoreArrayDOF(ctx->uda, y, &yg));
  PetscCall(DMDAVecRestoreArrayRead(ctx->eda, ctx->local_rho, &rho));
  return 0;
}

PetscErrorCode h8_opt_get_diagonal(Mat A, Vec diag_vec) {
  H8OptContext *ctx = nullptr;
  PetscScalar ****diag = nullptr;
  PetscScalar ***rho = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(MatShellGetContext(A, &ctx));
  const Grid &g = ctx->grid;

  PetscCall(DMGlobalToLocalBegin(ctx->eda, ctx->rho, INSERT_VALUES, ctx->local_rho));
  PetscCall(DMGlobalToLocalEnd(ctx->eda, ctx->rho, INSERT_VALUES, ctx->local_rho));
  PetscCall(DMDAVecGetArrayDOF(ctx->uda, diag_vec, &diag));
  PetscCall(DMDAVecGetArrayRead(ctx->eda, ctx->local_rho, &rho));
  PetscCall(DMDAGetCorners(ctx->uda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        for (PetscInt c = 0; c < 3; ++c) {
          if (h8_is_fixed_node(i, j, k, g, ctx->density_options)) {
            diag[k][j][i][c] = 1.0;
            continue;
          }
          PetscReal value = 0.0;
          const PetscInt ex0 = PetscMax(0, i - 1);
          const PetscInt ex1 = PetscMin(i, g.nx - 2);
          const PetscInt ey0 = PetscMax(0, j - 1);
          const PetscInt ey1 = PetscMin(j, g.ny - 2);
          const PetscInt ez0 = PetscMax(0, k - 1);
          const PetscInt ez1 = PetscMin(k, g.nz - 2);
          for (PetscInt ez = ez0; ez <= ez1; ++ez) {
            for (PetscInt ey = ey0; ey <= ey1; ++ey) {
              for (PetscInt ex = ex0; ex <= ex1; ++ex) {
                const PetscInt node = h8_local_node(i - ex, j - ey, k - ez);
                const PetscInt row = 3 * node + c;
                value += simp_scale(PetscRealPart(rho[ez][ey][ex]),
                                    ctx->density_options) *
                         ctx->ke[24 * row + row];
              }
            }
          }
          diag[k][j][i][c] = value;
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(ctx->uda, diag_vec, &diag));
  PetscCall(DMDAVecRestoreArrayRead(ctx->eda, ctx->local_rho, &rho));
  return 0;
}

PetscErrorCode h8_opt_destroy(Mat A) {
  H8OptContext *ctx = nullptr;
  PetscCall(MatShellGetContext(A, &ctx));
  if (ctx) {
    PetscCall(VecDestroy(&ctx->local_u));
    PetscCall(VecDestroy(&ctx->local_rho));
    delete ctx;
  }
  return 0;
}

PetscBool invert_3x3_block(const PetscReal in[9], PetscReal out[9]) {
  PetscReal a[9];
  PetscReal scale = 0.0;
  for (PetscInt i = 0; i < 9; ++i) {
    a[i] = in[i];
    scale = PetscMax(scale, PetscAbsReal(in[i]));
  }
  scale = PetscMax(scale, 1.0);
  for (PetscInt attempt = 0; attempt < 6; ++attempt) {
    if (attempt > 0) {
      const PetscReal shift = scale * PetscPowReal(10.0, -14.0 + attempt);
      a[0] = in[0] + shift;
      a[4] = in[4] + shift;
      a[8] = in[8] + shift;
      a[1] = in[1];
      a[2] = in[2];
      a[3] = in[3];
      a[5] = in[5];
      a[6] = in[6];
      a[7] = in[7];
    }
    const PetscReal c00 = a[4] * a[8] - a[5] * a[7];
    const PetscReal c01 = a[2] * a[7] - a[1] * a[8];
    const PetscReal c02 = a[1] * a[5] - a[2] * a[4];
    const PetscReal c10 = a[5] * a[6] - a[3] * a[8];
    const PetscReal c11 = a[0] * a[8] - a[2] * a[6];
    const PetscReal c12 = a[2] * a[3] - a[0] * a[5];
    const PetscReal c20 = a[3] * a[7] - a[4] * a[6];
    const PetscReal c21 = a[1] * a[6] - a[0] * a[7];
    const PetscReal c22 = a[0] * a[4] - a[1] * a[3];
    const PetscReal det = a[0] * c00 + a[1] * c10 + a[2] * c20;
    if (PetscAbsReal(det) > 1.0e-24 * scale * scale * scale) {
      const PetscReal inv_det = 1.0 / det;
      out[0] = c00 * inv_det;
      out[1] = c01 * inv_det;
      out[2] = c02 * inv_det;
      out[3] = c10 * inv_det;
      out[4] = c11 * inv_det;
      out[5] = c12 * inv_det;
      out[6] = c20 * inv_det;
      out[7] = c21 * inv_det;
      out[8] = c22 * inv_det;
      return PETSC_TRUE;
    }
  }
  for (PetscInt i = 0; i < 9; ++i) out[i] = 0.0;
  out[0] = 1.0 / PetscMax(PetscAbsReal(in[0]), 1.0e-30 * scale);
  out[4] = 1.0 / PetscMax(PetscAbsReal(in[4]), 1.0e-30 * scale);
  out[8] = 1.0 / PetscMax(PetscAbsReal(in[8]), 1.0e-30 * scale);
  return PETSC_FALSE;
}

PetscErrorCode h8_block_jacobi_setup(PC pc) {
  H8BlockJacobiContext *ctx = nullptr;
  PetscScalar ***rho = nullptr;
  PetscScalar ****inv = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(PCShellGetContext(pc, &ctx));
  const Grid &g = ctx->grid;

  PetscCall(DMGlobalToLocalBegin(ctx->eda, ctx->rho, INSERT_VALUES, ctx->local_rho));
  PetscCall(DMGlobalToLocalEnd(ctx->eda, ctx->rho, INSERT_VALUES, ctx->local_rho));
  PetscCall(VecSet(ctx->inv_blocks, 0.0));
  PetscCall(DMDAVecGetArrayRead(ctx->eda, ctx->local_rho, &rho));
  PetscCall(DMDAVecGetArrayDOF(ctx->bda, ctx->inv_blocks, &inv));
  PetscCall(DMDAGetCorners(ctx->uda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        PetscReal block[9] = {};
        PetscReal inverse[9] = {};
        if (h8_is_fixed_node(i, j, k, g, ctx->density_options)) {
          block[0] = 1.0;
          block[4] = 1.0;
          block[8] = 1.0;
        } else {
          const PetscInt ex0 = PetscMax(0, i - 1);
          const PetscInt ex1 = PetscMin(i, g.nx - 2);
          const PetscInt ey0 = PetscMax(0, j - 1);
          const PetscInt ey1 = PetscMin(j, g.ny - 2);
          const PetscInt ez0 = PetscMax(0, k - 1);
          const PetscInt ez1 = PetscMin(k, g.nz - 2);
          for (PetscInt ez = ez0; ez <= ez1; ++ez) {
            for (PetscInt ey = ey0; ey <= ey1; ++ey) {
              for (PetscInt ex = ex0; ex <= ex1; ++ex) {
                const PetscInt node = h8_local_node(i - ex, j - ey, k - ez);
                const PetscReal scale =
                    simp_scale(PetscRealPart(rho[ez][ey][ex]),
                               ctx->density_options);
                for (PetscInt c = 0; c < 3; ++c) {
                  const PetscInt row = 3 * node + c;
                  for (PetscInt d = 0; d < 3; ++d) {
                    const PetscInt col = 3 * node + d;
                    block[3 * c + d] += scale * ctx->ke[24 * row + col];
                  }
                }
              }
            }
          }
        }
        (void)invert_3x3_block(block, inverse);
        for (PetscInt q = 0; q < 9; ++q) {
          inv[k][j][i][q] = inverse[q];
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(ctx->eda, ctx->local_rho, &rho));
  PetscCall(DMDAVecRestoreArrayDOF(ctx->bda, ctx->inv_blocks, &inv));
  return 0;
}

PetscErrorCode h8_block_jacobi_apply(PC pc, Vec x, Vec y) {
  H8BlockJacobiContext *ctx = nullptr;
  PetscScalar ****xg = nullptr;
  PetscScalar ****yg = nullptr;
  PetscScalar ****inv = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(DMDAVecGetArrayDOFRead(ctx->uda, x, &xg));
  PetscCall(DMDAVecGetArrayDOF(ctx->uda, y, &yg));
  PetscCall(DMDAVecGetArrayDOFRead(ctx->bda, ctx->inv_blocks, &inv));
  PetscCall(DMDAGetCorners(ctx->uda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        for (PetscInt c = 0; c < 3; ++c) {
          PetscScalar value = 0.0;
          for (PetscInt d = 0; d < 3; ++d) {
            value += inv[k][j][i][3 * c + d] * xg[k][j][i][d];
          }
          yg[k][j][i][c] = value;
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(ctx->uda, x, &xg));
  PetscCall(DMDAVecRestoreArrayDOF(ctx->uda, y, &yg));
  PetscCall(DMDAVecRestoreArrayDOFRead(ctx->bda, ctx->inv_blocks, &inv));
  return 0;
}

PetscErrorCode h8_block_jacobi_destroy(PC pc) {
  H8BlockJacobiContext *ctx = nullptr;
  PetscCall(PCShellGetContext(pc, &ctx));
  if (ctx) {
    PetscCall(VecDestroy(&ctx->inv_blocks));
    PetscCall(VecDestroy(&ctx->local_rho));
    PetscCall(DMDestroy(&ctx->bda));
    delete ctx;
  }
  return 0;
}

PetscErrorCode set_h8_block_jacobi_pc(KSP ksp, DM uda, DM eda, Vec rho,
                                      const Grid &grid,
                                      const DensityOptions &density_options) {
  PC pc = nullptr;
  PetscInt px = 1, py = 1, pz = 1;
  H8BlockJacobiContext *ctx = new H8BlockJacobiContext();
  ctx->uda = uda;
  ctx->eda = eda;
  ctx->rho = rho;
  ctx->grid = grid;
  ctx->density_options = density_options;
  compute_h8_element_stiffness(grid, density_options, ctx->ke);
  PetscCall(DMDAGetInfo(uda, nullptr, nullptr, nullptr, nullptr,
                        &px, &py, &pz, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         grid.nx, grid.ny, grid.nz, px, py, pz, 9, 1,
                         nullptr, nullptr, nullptr, &ctx->bda));
  PetscCall(DMSetUp(ctx->bda));
  PetscCall(DMCreateLocalVector(eda, &ctx->local_rho));
  PetscCall(DMCreateGlobalVector(ctx->bda, &ctx->inv_blocks));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, ctx));
  PetscCall(PCShellSetName(pc, "h8_node_block_jacobi"));
  PetscCall(PCShellSetSetUp(pc, h8_block_jacobi_setup));
  PetscCall(PCShellSetApply(pc, h8_block_jacobi_apply));
  PetscCall(PCShellSetDestroy(pc, h8_block_jacobi_destroy));
  return 0;
}

PetscReal h8_average_adjacent_element_density(PetscScalar ***rho,
                                              const Grid &grid,
                                              PetscInt i, PetscInt j,
                                              PetscInt k) {
  PetscReal sum = 0.0;
  PetscInt count = 0;
  const PetscInt ex0 = PetscMax(0, i - 1);
  const PetscInt ex1 = PetscMin(i, grid.nx - 2);
  const PetscInt ey0 = PetscMax(0, j - 1);
  const PetscInt ey1 = PetscMin(j, grid.ny - 2);
  const PetscInt ez0 = PetscMax(0, k - 1);
  const PetscInt ez1 = PetscMin(k, grid.nz - 2);
  for (PetscInt ez = ez0; ez <= ez1; ++ez) {
    for (PetscInt ey = ey0; ey <= ey1; ++ey) {
      for (PetscInt ex = ex0; ex <= ex1; ++ex) {
        sum += PetscRealPart(rho[ez][ey][ex]);
        ++count;
      }
    }
  }
  return count > 0 ? sum / static_cast<PetscReal>(count) : 0.0;
}

PetscErrorCode assemble_h8_aux_laplacian_matrix(DM uda, DM eda, Vec rho,
                                                const Grid &grid,
                                                const DensityOptions &density_options,
                                                Mat P) {
  Vec local_rho = nullptr;
  PetscScalar ***rg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const PetscInt di[6] = {-1, 1, 0, 0, 0, 0};
  const PetscInt dj[6] = {0, 0, -1, 1, 0, 0};
  const PetscInt dk[6] = {0, 0, 0, 0, -1, 1};
  const PetscReal hx = domain_length(grid) /
                       static_cast<PetscReal>(PetscMax(1, grid.nx - 1));
  const PetscReal hy = domain_width(grid) /
                       static_cast<PetscReal>(PetscMax(1, grid.ny - 1));
  const PetscReal hz = domain_height(grid) /
                       static_cast<PetscReal>(PetscMax(1, grid.nz - 1));
  const PetscReal wx = hy * hz / PetscMax(hx, PETSC_SMALL);
  const PetscReal wy = hx * hz / PetscMax(hy, PETSC_SMALL);
  const PetscReal wz = hx * hy / PetscMax(hz, PETSC_SMALL);
  const PetscReal weight[6] = {wx, wx, wy, wy, wz, wz};

  PetscCall(MatZeroEntries(P));
  PetscCall(DMCreateLocalVector(eda, &local_rho));
  PetscCall(DMGlobalToLocalBegin(eda, rho, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(eda, rho, INSERT_VALUES, local_rho));
  PetscCall(DMDAVecGetArrayRead(eda, local_rho, &rg));
  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal rho0 =
            h8_average_adjacent_element_density(rg, grid, i, j, k);
        for (PetscInt c = 0; c < 3; ++c) {
          MatStencil row;
          MatStencil cols[7];
          PetscScalar vals[7];
          PetscInt ncols = 0;
          PetscReal diag = 0.0;
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = c;

          if (h8_is_fixed_node(i, j, k, grid, density_options)) {
            cols[0] = row;
            vals[0] = 1.0;
            PetscCall(MatSetValuesStencil(P, 1, &row, 1, cols, vals,
                                           ADD_VALUES));
            continue;
          }

          for (PetscInt q = 0; q < 6; ++q) {
            const PetscInt ii = i + di[q];
            const PetscInt jj = j + dj[q];
            const PetscInt kk = k + dk[q];
            if (ii < 0 || ii >= grid.nx || jj < 0 || jj >= grid.ny ||
                kk < 0 || kk >= grid.nz) {
              continue;
            }
            const PetscReal rho1 =
                h8_average_adjacent_element_density(rg, grid, ii, jj, kk);
            const PetscReal kij =
                weight[q] * edge_stiffness(rho0, rho1, density_options);
            diag += kij;
            if (!h8_is_fixed_node(ii, jj, kk, grid, density_options)) {
              cols[ncols].i = ii;
              cols[ncols].j = jj;
              cols[ncols].k = kk;
              cols[ncols].c = c;
              vals[ncols] = -kij;
              ++ncols;
            }
          }

          cols[ncols] = row;
          vals[ncols] = PetscMax(diag, PETSC_SMALL);
          ++ncols;
          PetscCall(MatSetValuesStencil(P, 1, &row, ncols, cols, vals,
                                         ADD_VALUES));
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(eda, local_rho, &rg));
  PetscCall(VecDestroy(&local_rho));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  return 0;
}

PetscErrorCode assemble_h8_aux_elasticity_matrix(DM uda, DM eda, Vec rho,
                                                 const Grid &grid,
                                                 const DensityOptions &density_options,
                                                 Mat P) {
  Vec local_rho = nullptr;
  PetscScalar ***rg = nullptr;
  PetscReal ke[24 * 24]{};
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  compute_h8_element_stiffness(grid, density_options, ke);

  PetscCall(MatZeroEntries(P));
  PetscCall(DMCreateLocalVector(eda, &local_rho));
  PetscCall(DMGlobalToLocalBegin(eda, rho, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(eda, rho, INSERT_VALUES, local_rho));
  PetscCall(DMDAVecGetArrayRead(eda, local_rho, &rg));
  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        for (PetscInt c = 0; c < 3; ++c) {
          MatStencil row;
          MatStencil cols[8 * 8 * 3];
          PetscScalar vals[8 * 8 * 3];
          PetscInt ncols = 0;
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = c;

          if (h8_is_fixed_node(i, j, k, grid, density_options)) {
            cols[0] = row;
            vals[0] = 1.0;
            PetscCall(MatSetValuesStencil(P, 1, &row, 1, cols, vals,
                                           ADD_VALUES));
            continue;
          }

          const PetscInt ex0 = PetscMax(0, i - 1);
          const PetscInt ex1 = PetscMin(i, grid.nx - 2);
          const PetscInt ey0 = PetscMax(0, j - 1);
          const PetscInt ey1 = PetscMin(j, grid.ny - 2);
          const PetscInt ez0 = PetscMax(0, k - 1);
          const PetscInt ez1 = PetscMin(k, grid.nz - 2);
          for (PetscInt ez = ez0; ez <= ez1; ++ez) {
            for (PetscInt ey = ey0; ey <= ey1; ++ey) {
              for (PetscInt ex = ex0; ex <= ex1; ++ex) {
                const PetscInt row_node = h8_local_node(i - ex, j - ey, k - ez);
                const PetscInt erow = 3 * row_node + c;
                const PetscReal scale =
                    simp_scale(PetscRealPart(rg[ez][ey][ex]),
                               density_options);
                for (PetscInt col_node = 0; col_node < 8; ++col_node) {
                  PetscInt ni = 0, nj = 0, nk = 0;
                  h8_node_coords(ex, ey, ez, col_node, &ni, &nj, &nk);
                  if (h8_is_fixed_node(ni, nj, nk, grid, density_options)) continue;
                  for (PetscInt d = 0; d < 3; ++d) {
                    const PetscInt ecol = 3 * col_node + d;
                    cols[ncols].i = ni;
                    cols[ncols].j = nj;
                    cols[ncols].k = nk;
                    cols[ncols].c = d;
                    vals[ncols] = scale * ke[24 * erow + ecol];
                    ++ncols;
                  }
                }
              }
            }
          }

          PetscCall(MatSetValuesStencil(P, 1, &row, ncols, cols, vals,
                                         ADD_VALUES));
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(eda, local_rho, &rg));
  PetscCall(VecDestroy(&local_rho));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  return 0;
}

PetscErrorCode create_h8_aux_laplacian_matrix(DM uda, DM eda, Vec rho,
                                              const Grid &grid,
                                              const DensityOptions &density_options,
                                              Mat *P) {
  PetscCall(DMCreateMatrix(uda, P));
  PetscCall(MatSetOption(*P, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(assemble_h8_aux_laplacian_matrix(uda, eda, rho, grid,
                                             density_options, *P));
  return 0;
}

PetscErrorCode create_h8_aux_elasticity_matrix(DM uda, DM eda, Vec rho,
                                               const Grid &grid,
                                               const DensityOptions &density_options,
                                               Mat *P) {
  PetscCall(DMCreateMatrix(uda, P));
  PetscCall(MatSetOption(*P, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(assemble_h8_aux_elasticity_matrix(uda, eda, rho, grid,
                                              density_options, *P));
  return 0;
}

PetscBool h8_pc_type_uses_aux_matrix(const char *h8_pc_type) {
  return (std::strcmp(h8_pc_type, "aux_gamg") == 0 ||
          std::strcmp(h8_pc_type, "aux_hypre") == 0 ||
          std::strcmp(h8_pc_type, "aux_elastic_gamg") == 0 ||
          std::strcmp(h8_pc_type, "aux_elastic_hypre") == 0 ||
          std::strcmp(h8_pc_type, "aux_elastic_lu") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool h8_pc_type_uses_elastic_aux_matrix(const char *h8_pc_type) {
  return (std::strcmp(h8_pc_type, "aux_elastic_gamg") == 0 ||
          std::strcmp(h8_pc_type, "aux_elastic_hypre") == 0 ||
          std::strcmp(h8_pc_type, "aux_elastic_lu") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscErrorCode get_h8_pc_type_option(char *h8_pc_type, size_t h8_pc_type_size,
                                     PetscBool *uses_aux_matrix) {
  PetscBool has_h8_pc_type = PETSC_FALSE;
  PetscCall(PetscStrncpy(h8_pc_type, "aux_gamg", h8_pc_type_size));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-h8_pc_type",
                                  h8_pc_type, h8_pc_type_size,
                                  &has_h8_pc_type));
  (void)has_h8_pc_type;
  *uses_aux_matrix = h8_pc_type_uses_aux_matrix(h8_pc_type);
  return 0;
}

PetscErrorCode set_h8_ksp_operators(KSP ksp, Mat A, Mat P,
                                    DM uda, DM eda, Vec rho,
                                    const Grid &grid,
                                    const DensityOptions &density_options,
                                    PetscBool rebuild_aux_matrix,
                                    PetscBool use_elastic_aux_matrix) {
  if (P != nullptr) {
    if (rebuild_aux_matrix) {
      if (use_elastic_aux_matrix) {
        PetscCall(assemble_h8_aux_elasticity_matrix(uda, eda, rho, grid,
                                                    density_options, P));
      } else {
        PetscCall(assemble_h8_aux_laplacian_matrix(uda, eda, rho, grid,
                                                   density_options, P));
      }
      PetscCall(KSPSetReusePreconditioner(ksp, PETSC_FALSE));
      PetscCall(KSPSetOperators(ksp, A, P));
    } else {
      PetscCall(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
    }
  } else {
    PetscCall(KSPSetReusePreconditioner(ksp, PETSC_FALSE));
    PetscCall(KSPSetOperators(ksp, A, A));
  }
  return 0;
}

PetscErrorCode configure_h8_ksp(KSP ksp, Mat A, Mat P, DM uda, DM eda, Vec rho,
                                const Grid &grid,
                                const DensityOptions &density_options,
                                const OptimizerOptions &optimizer_options) {
  char h8_pc_type[64] = "aux_gamg";
  PetscBool has_h8_pc_type = PETSC_FALSE;
  PetscBool allow_large_block_jacobi = PETSC_FALSE;
  PetscInt large_block_jacobi_dof_limit = 2000000;
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-h8_pc_type",
                                  h8_pc_type, sizeof(h8_pc_type),
                                  &has_h8_pc_type));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr,
                                "-h8_allow_large_block_jacobi",
                                &allow_large_block_jacobi, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr,
                               "-h8_large_block_jacobi_dof_limit",
                               &large_block_jacobi_dof_limit, nullptr));
  if (std::strcmp(h8_pc_type, "block_jacobi") == 0) {
    const PetscInt ndof = dof_count(grid);
    PetscCheck(allow_large_block_jacobi ||
                   ndof < large_block_jacobi_dof_limit,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "Refusing H8 block_jacobi on %lld DOF. It is intended for small diagnostics and stalled in production; use -h8_pc_type aux_hypre or set -h8_allow_large_block_jacobi true.",
               static_cast<long long>(ndof));
  }
  PetscCall(KSPSetOperators(ksp, A, P ? P : A));
  PetscCall(KSPSetType(ksp, h8_pc_type_uses_aux_matrix(h8_pc_type)
                                ? KSPFGMRES
                                : KSPCG));
  PetscCall(KSPSetTolerances(ksp, optimizer_options.ksp_rtol, PETSC_DEFAULT,
                             PETSC_DEFAULT, optimizer_options.ksp_max_it));
  PetscCall(KSPSetFromOptions(ksp));
  if (std::strcmp(h8_pc_type, "block_jacobi") == 0) {
    PetscCall(set_h8_block_jacobi_pc(ksp, uda, eda, rho, grid, density_options));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 preconditioner: node 3x3 block Jacobi%s\n",
                          has_h8_pc_type ? "" : " (default)"));
  } else if (std::strcmp(h8_pc_type, "jacobi") == 0) {
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 preconditioner: scalar PETSc Jacobi\n"));
  } else if (std::strcmp(h8_pc_type, "petsc") == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 preconditioner: using PETSc -pc_type options\n"));
  } else if (std::strcmp(h8_pc_type, "aux_gamg") == 0) {
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_gamg needs an assembled auxiliary matrix");
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMG));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 preconditioner: auxiliary low-order matrix with PETSc GAMG%s\n",
                          has_h8_pc_type ? "" : " (default)"));
  } else if (std::strcmp(h8_pc_type, "aux_hypre") == 0) {
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_hypre needs an assembled auxiliary matrix");
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCHYPRE));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 preconditioner: auxiliary low-order matrix with PETSc/HYPRE\n"));
  } else if (std::strcmp(h8_pc_type, "aux_elastic_gamg") == 0) {
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_elastic_gamg needs an assembled auxiliary matrix");
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMG));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 preconditioner: assembled H8 elasticity auxiliary matrix with PETSc GAMG\n"));
  } else if (std::strcmp(h8_pc_type, "aux_elastic_hypre") == 0) {
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_elastic_hypre needs an assembled auxiliary matrix");
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCHYPRE));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 preconditioner: assembled H8 elasticity auxiliary matrix with PETSc/HYPRE\n"));
  } else if (std::strcmp(h8_pc_type, "aux_elastic_lu") == 0) {
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_elastic_lu needs an assembled auxiliary matrix");
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCLU));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 preconditioner: assembled H8 elasticity auxiliary matrix with PETSc LU (small diagnostics)\n"));
  } else {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "-h8_pc_type must be block_jacobi, jacobi, petsc, aux_gamg, aux_hypre, aux_elastic_gamg, aux_elastic_hypre, or aux_elastic_lu");
  }
  return 0;
}

PetscErrorCode create_h8_shell_matrix(DM uda, DM eda, Vec rho,
                                      const Grid &grid,
                                      const DensityOptions &density_options,
                                      Mat *A) {
  H8OptContext *ctx = new H8OptContext();
  Vec template_vec = nullptr;
  PetscInt local_size = 0;
  ctx->uda = uda;
  ctx->eda = eda;
  ctx->rho = rho;
  ctx->grid = grid;
  ctx->density_options = density_options;
  compute_h8_element_stiffness(grid, density_options, ctx->ke);
  PetscCall(DMCreateLocalVector(uda, &ctx->local_u));
  PetscCall(DMCreateLocalVector(eda, &ctx->local_rho));
  PetscCall(DMCreateGlobalVector(uda, &template_vec));
  PetscCall(VecGetLocalSize(template_vec, &local_size));
  PetscCall(VecDestroy(&template_vec));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD, local_size, local_size,
                           dof_count(grid), dof_count(grid), ctx, A));
  PetscCall(MatShellSetOperation(*A, MATOP_MULT,
                                 reinterpret_cast<void (*)(void)>(h8_opt_mult)));
  PetscCall(MatShellSetOperation(*A, MATOP_GET_DIAGONAL,
                                 reinterpret_cast<void (*)(void)>(h8_opt_get_diagonal)));
  PetscCall(MatShellSetOperation(*A, MATOP_DESTROY,
                                 reinterpret_cast<void (*)(void)>(h8_opt_destroy)));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  return 0;
}

void h8_add_wrench_matrix(PetscReal x, PetscReal y, PetscReal z,
                          const PetscReal center[3], PetscReal S[6][6]) {
  const PetscReal rx = x - center[0];
  const PetscReal ry = y - center[1];
  const PetscReal rz = z - center[2];
  const PetscReal C[6][3] = {
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {0.0, -rz, ry},
      {rz, 0.0, -rx},
      {-ry, rx, 0.0}};
  for (PetscInt a = 0; a < 6; ++a) {
    for (PetscInt b = 0; b < 6; ++b) {
      for (PetscInt c = 0; c < 3; ++c) S[a][b] += C[a][c] * C[b][c];
    }
  }
}

void h8_solve_6x6(PetscReal A[6][6], const PetscReal b[6], PetscReal x[6]) {
  PetscReal aug[6][7] = {};
  for (PetscInt i = 0; i < 6; ++i) {
    for (PetscInt j = 0; j < 6; ++j) aug[i][j] = A[i][j];
    aug[i][i] += 1.0e-12;
    aug[i][6] = b[i];
  }
  for (PetscInt col = 0; col < 6; ++col) {
    PetscInt pivot = col;
    for (PetscInt row = col + 1; row < 6; ++row) {
      if (PetscAbsReal(aug[row][col]) > PetscAbsReal(aug[pivot][col])) pivot = row;
    }
    if (pivot != col) {
      for (PetscInt j = col; j < 7; ++j) std::swap(aug[col][j], aug[pivot][j]);
    }
    const PetscReal diag = aug[col][col];
    if (PetscAbsReal(diag) < 1.0e-30) {
      x[col] = 0.0;
      continue;
    }
    for (PetscInt j = col; j < 7; ++j) aug[col][j] /= diag;
    for (PetscInt row = 0; row < 6; ++row) {
      if (row == col) continue;
      const PetscReal f = aug[row][col];
      for (PetscInt j = col; j < 7; ++j) aug[row][j] -= f * aug[col][j];
    }
  }
  for (PetscInt i = 0; i < 6; ++i) x[i] = aug[i][6];
}

void h8_wrench_force_at_node(PetscReal x, PetscReal y, PetscReal z,
                             const PetscReal center[3],
                             const PetscReal q[6], PetscReal f[3]) {
  const PetscReal rx = x - center[0];
  const PetscReal ry = y - center[1];
  const PetscReal rz = z - center[2];
  f[0] = q[0] + rz * q[4] - ry * q[5];
  f[1] = q[1] - rz * q[3] + rx * q[5];
  f[2] = q[2] + ry * q[3] - rx * q[4];
}

bool h8_node_has_active_adjacent_element(PetscScalar ***mask,
                                         const Grid &grid,
                                         PetscInt i, PetscInt j,
                                         PetscInt k) {
  const PetscInt ex0 = PetscMax(0, i - 1);
  const PetscInt ex1 = PetscMin(i, grid.nx - 2);
  const PetscInt ey0 = PetscMax(0, j - 1);
  const PetscInt ey1 = PetscMin(j, grid.ny - 2);
  const PetscInt ez0 = PetscMax(0, k - 1);
  const PetscInt ez1 = PetscMin(k, grid.nz - 2);
  for (PetscInt ez = ez0; ez <= ez1; ++ez) {
    for (PetscInt ey = ey0; ey <= ey1; ++ey) {
      for (PetscInt ex = ex0; ex <= ex1; ++ex) {
        if (h8_mask_is_active(PetscRealPart(mask[ez][ey][ex]))) {
          return true;
        }
      }
    }
  }
  return false;
}

PetscErrorCode h8_control_arm_wrench_q(DM da, DM eda, Vec mask_vec,
                                       const Grid &grid,
                                       PetscInt which,
                                       const PetscReal wrench[6],
                                       PetscReal q[6],
                                       PetscInt *global_count,
                                       PetscInt *global_candidates) {
  Vec local_mask = nullptr;
  PetscScalar ***mask = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_s[36] = {};
  PetscReal global_s[36] = {};
  PetscInt local_count = 0;
  PetscInt local_candidates = 0;
  const PetscReal center[3] = {0.18 * domain_length(grid),
                               which == 1 ? domain_width(grid) : 0.0,
                               0.50 * domain_height(grid)};

  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, mask_vec, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask_vec, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &mask));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (!h8_is_ab_load_node(i, j, k, grid, which)) continue;
        ++local_candidates;
        if (!h8_node_has_active_adjacent_element(mask, grid, i, j, k)) {
          continue;
        }
        PetscReal x = 0.0, y = 0.0, z = 0.0;
        PetscReal S[6][6] = {};
        h8_node_physical(i, j, k, grid, &x, &y, &z);
        h8_add_wrench_matrix(x, y, z, center, S);
        for (PetscInt a = 0; a < 6; ++a) {
          for (PetscInt b = 0; b < 6; ++b) local_s[6 * a + b] += S[a][b];
        }
        ++local_count;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &mask));
  PetscCall(VecDestroy(&local_mask));
  PetscCallMPI(MPI_Allreduce(local_s, global_s, 36, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_count, global_count, 1, MPIU_INT, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_candidates, global_candidates, 1, MPIU_INT,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscReal Sg[6][6] = {};
  for (PetscInt a = 0; a < 6; ++a) {
    for (PetscInt b = 0; b < 6; ++b) Sg[a][b] = global_s[6 * a + b];
  }
  h8_solve_6x6(Sg, wrench, q);
  return 0;
}

PetscErrorCode h8_count_supported_ab_load_nodes(DM da, DM eda, Vec mask_vec,
                                                const Grid &grid,
                                                PetscInt which,
                                                PetscInt *global_count,
                                                PetscInt *global_candidates) {
  Vec local_mask = nullptr;
  PetscScalar ***mask = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt local_count = 0;
  PetscInt local_candidates = 0;

  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, mask_vec, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask_vec, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &mask));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (!h8_is_ab_load_node(i, j, k, grid, which)) continue;
        ++local_candidates;
        if (h8_is_c_support_node(i, j, k, grid)) continue;
        if (h8_node_has_active_adjacent_element(mask, grid, i, j, k)) {
          ++local_count;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &mask));
  PetscCall(VecDestroy(&local_mask));
  PetscCallMPI(MPI_Allreduce(&local_count, global_count, 1, MPIU_INT,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_candidates, global_candidates, 1,
                             MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  return 0;
}

PetscReal h8_control_arm_case_weight(PetscInt load_case) {
  static const PetscReal weights[3] = {0.25, 0.50, 0.25};
  if (load_case >= 1 && load_case <= 3) return weights[load_case - 1];
  return 1.0;
}

PetscErrorCode h8_get_load_options(PetscInt *load_case,
                                   PetscBool *include_spring_load) {
  PetscBool found = PETSC_FALSE;
  *load_case = 0;
  *include_spring_load = PETSC_TRUE;
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-h8_load_case",
                               load_case, &found));
  if (!found) {
    PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-load_case",
                                 load_case, nullptr));
  }
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-h8_include_spring_load",
                                include_spring_load, &found));
  if (!found) {
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-include_spring_load",
                                  include_spring_load, nullptr));
  }
  return 0;
}

PetscErrorCode add_h8_control_arm_load_case(DM uda, DM eda, Vec mask,
                                            const Grid &grid,
                                            PetscReal load_scale,
                                            PetscInt load_case,
                                            PetscReal case_weight,
                                            PetscBool include_spring_load,
                                            Vec b) {
  static const PetscReal p1_f[3][3] = {
      {91.0, 3597.0, 10230.0},
      {-814.0, 2716.0, 22726.0},
      {276.0, 5569.0, 16943.0}};
  static const PetscReal p1_t[3][3] = {
      {-6309.0, -37360.0, 13369.0},
      {-29734.0, 16033.0, -3062.0},
      {-14823.0, 58723.0, -19636.0}};
  static const PetscReal p2_f[3][3] = {
      {2534.0, 10162.0, -8288.0},
      {-1790.0, 3503.0, -16398.0},
      {-3488.0, -4632.0, -7851.0}};
  static const PetscReal p2_t[3][3] = {
      {4200.0, -6200.0, 3600.0},
      {-7600.0, 9200.0, -5200.0},
      {5400.0, 6800.0, 6400.0}};
  static const PetscReal spring_mag[3] = {8000.0, 16000.0, 10000.0};
  Vec local_mask = nullptr;
  PetscScalar ****bg = nullptr;
  PetscScalar ***mg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt local_spring = 0, global_spring = 0;
  PetscInt local_spring_candidates = 0, global_spring_candidates = 0;
  const PetscInt cidx = load_case - 1;
  PetscReal wrench1[6] = {p1_f[cidx][0], p1_f[cidx][1], p1_f[cidx][2],
                          p1_t[cidx][0], p1_t[cidx][1], p1_t[cidx][2]};
  PetscReal wrench2[6] = {p2_f[cidx][0], p2_f[cidx][1], p2_f[cidx][2],
                          p2_t[cidx][0], p2_t[cidx][1], p2_t[cidx][2]};
  PetscReal q1[6] = {};
  PetscReal q2[6] = {};
  PetscInt n1 = 0, n2 = 0;
  PetscInt cand1 = 0, cand2 = 0;

  PetscCheck(load_case >= 1 && load_case <= 3, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_OUTOFRANGE,
             "H8 load_case must be 1, 2, or 3 inside add_h8_control_arm_load_case");
  PetscCall(h8_control_arm_wrench_q(uda, eda, mask, grid, 1, wrench1, q1,
                                    &n1, &cand1));
  PetscCall(h8_control_arm_wrench_q(uda, eda, mask, grid, 2, wrench2, q2,
                                    &n2, &cand2));
  PetscCheck(n1 > 0 && n2 > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "Control-arm A/B load rings have no supported nodes. Increase grid resolution or check forced-solid/load geometry.");

  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &mg));
  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (!h8_is_spring_load_node(i, j, k, grid)) continue;
        ++local_spring_candidates;
        if (h8_node_has_active_adjacent_element(mg, grid, i, j, k)) {
          ++local_spring;
        }
      }
    }
  }
  PetscCallMPI(MPI_Allreduce(&local_spring, &global_spring, 1, MPIU_INT,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_spring_candidates,
                             &global_spring_candidates, 1, MPIU_INT,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 load case %lld supported nodes: A=%lld/%lld B=%lld/%lld spring=%lld/%lld\n",
                        static_cast<long long>(load_case),
                        static_cast<long long>(n1),
                        static_cast<long long>(cand1),
                        static_cast<long long>(n2),
                        static_cast<long long>(cand2),
                        static_cast<long long>(global_spring),
                        static_cast<long long>(global_spring_candidates)));

  PetscCall(DMDAVecGetArrayDOF(uda, b, &bg));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (h8_is_c_support_node(i, j, k, grid)) continue;
        if (!h8_node_has_active_adjacent_element(mg, grid, i, j, k)) continue;
        PetscReal x = 0.0, y = 0.0, z = 0.0;
        h8_node_physical(i, j, k, grid, &x, &y, &z);
        if (h8_is_ab_load_node(i, j, k, grid, 1)) {
          const PetscReal center[3] = {0.18 * domain_length(grid),
                                       domain_width(grid),
                                       0.50 * domain_height(grid)};
          PetscReal f[3] = {};
          h8_wrench_force_at_node(x, y, z, center, q1, f);
          for (PetscInt d = 0; d < 3; ++d) {
            bg[k][j][i][d] += load_scale * case_weight * f[d];
          }
        }
        if (h8_is_ab_load_node(i, j, k, grid, 2)) {
          const PetscReal center[3] = {0.18 * domain_length(grid), 0.0,
                                       0.50 * domain_height(grid)};
          PetscReal f[3] = {};
          h8_wrench_force_at_node(x, y, z, center, q2, f);
          for (PetscInt d = 0; d < 3; ++d) {
            bg[k][j][i][d] += load_scale * case_weight * f[d];
          }
        }
        if (include_spring_load && global_spring > 0 &&
            h8_is_spring_load_node(i, j, k, grid)) {
          bg[k][j][i][2] += load_scale * case_weight *
                            (-spring_mag[cidx]) /
                            static_cast<PetscReal>(global_spring);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(uda, b, &bg));
  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &mg));
  PetscCall(VecDestroy(&local_mask));
  return 0;
}

PetscErrorCode add_h8_control_arm_symmetric_vertical_load_case(
    DM uda, DM eda, Vec mask, const Grid &grid, PetscReal load_scale,
    PetscReal case_weight, PetscBool include_spring_load, Vec b) {
  static const PetscReal ab_ring_mag = 10000.0;
  static const PetscReal spring_mag = 8000.0;
  Vec local_mask = nullptr;
  PetscScalar ****bg = nullptr;
  PetscScalar ***mg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt n1 = 0, n2 = 0;
  PetscInt cand1 = 0, cand2 = 0;
  PetscInt local_spring = 0, global_spring = 0;
  PetscInt local_spring_candidates = 0, global_spring_candidates = 0;

  PetscCall(h8_count_supported_ab_load_nodes(uda, eda, mask, grid, 1, &n1,
                                             &cand1));
  PetscCall(h8_count_supported_ab_load_nodes(uda, eda, mask, grid, 2, &n2,
                                             &cand2));
  PetscCheck(n1 > 0 && n2 > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "H8 load_case=4 needs nonempty A/B load rings.");
  PetscCheck(cand1 == cand2, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "H8 load_case=4 requires symmetric A/B candidate load nodes, but A=%lld and B=%lld. Check grid and control-arm mask symmetry.",
             static_cast<long long>(cand1), static_cast<long long>(cand2));
  PetscCheck(n1 == n2, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "H8 load_case=4 requires symmetric A/B active load nodes, but A=%lld and B=%lld. Check grid and control-arm mask symmetry.",
             static_cast<long long>(n1), static_cast<long long>(n2));

  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &mg));
  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (!h8_is_spring_load_node(i, j, k, grid)) continue;
        ++local_spring_candidates;
        if (h8_is_c_support_node(i, j, k, grid)) continue;
        if (h8_node_has_active_adjacent_element(mg, grid, i, j, k)) {
          ++local_spring;
        }
      }
    }
  }
  PetscCallMPI(MPI_Allreduce(&local_spring, &global_spring, 1, MPIU_INT,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_spring_candidates,
                             &global_spring_candidates, 1, MPIU_INT,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCheck(!include_spring_load || global_spring > 0, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONG,
             "H8 load_case=4 includes spring load but the spring load region is empty.");

  const PetscReal total_fz =
      -load_scale * case_weight *
      (2.0 * ab_ring_mag + (include_spring_load ? spring_mag : 0.0));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "H8 load case 4 symmetric vertical load: A=%lld/%lld B=%lld/%lld spring=%lld/%lld Fz_total=%.6e\n",
      static_cast<long long>(n1), static_cast<long long>(cand1),
      static_cast<long long>(n2), static_cast<long long>(cand2),
      static_cast<long long>(global_spring),
      static_cast<long long>(global_spring_candidates),
      static_cast<double>(total_fz)));

  PetscCall(DMDAVecGetArrayDOF(uda, b, &bg));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (h8_is_c_support_node(i, j, k, grid)) continue;
        if (!h8_node_has_active_adjacent_element(mg, grid, i, j, k)) continue;
        if (h8_is_ab_load_node(i, j, k, grid, 1)) {
          bg[k][j][i][2] +=
              load_scale * case_weight * (-ab_ring_mag) /
              static_cast<PetscReal>(n1);
        }
        if (h8_is_ab_load_node(i, j, k, grid, 2)) {
          bg[k][j][i][2] +=
              load_scale * case_weight * (-ab_ring_mag) /
              static_cast<PetscReal>(n2);
        }
        if (include_spring_load && global_spring > 0 &&
            h8_is_spring_load_node(i, j, k, grid)) {
          bg[k][j][i][2] += load_scale * case_weight * (-spring_mag) /
                            static_cast<PetscReal>(global_spring);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(uda, b, &bg));
  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &mg));
  PetscCall(VecDestroy(&local_mask));
  return 0;
}

PetscErrorCode fill_h8_benchmark_load(DM uda, const Grid &grid,
                                      const OptimizerOptions &optimizer_options,
                                      Vec b) {
  PetscScalar ****bg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_torsion_denom = 0.0;
  PetscReal global_torsion_denom = 0.0;
  const PetscBool torsion = benchmark_is_torsion(optimizer_options.benchmark_case);
  const PetscBool torsion_edge =
      benchmark_is_torsion_edge(optimizer_options.benchmark_case);
  const PetscBool bridge = benchmark_is_bridge(optimizer_options.benchmark_case);
  const PetscBool mbb = benchmark_is_mbb(optimizer_options.benchmark_case);
  const PetscBool tri = benchmark_is_tri(optimizer_options.benchmark_case);
  const PetscBool bottom_point =
      benchmark_is_bottom_point_cantilever(optimizer_options.benchmark_case);
  const PetscBool tip_center =
      benchmark_is_tip_center_cantilever(optimizer_options.benchmark_case);
  const PetscInt load_j0 = PetscMax(static_cast<PetscInt>(0), (grid.ny - 1) / 2);
  const PetscInt load_j1 = PetscMax(static_cast<PetscInt>(0), grid.ny / 2);
  const PetscInt cantilever_load_nodes = (load_j0 == load_j1) ? 1 : 2;
  const PetscInt load_k0 = PetscMax(static_cast<PetscInt>(0), (grid.nz - 1) / 2);
  const PetscInt load_k1 = PetscMax(static_cast<PetscInt>(0), grid.nz / 2);
  const PetscInt tip_center_load_nodes =
      ((load_j0 == load_j1) ? 1 : 2) * ((load_k0 == load_k1) ? 1 : 2);
  const PetscInt matlab_load_k =
      PetscMax(static_cast<PetscInt>(0), (grid.nz - 1) / 2);

  PetscCheck(torsion || torsion_edge || bridge || mbb || tri ||
                  benchmark_is_cantilever(optimizer_options.benchmark_case),
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "-benchmark_case must be cantilever, cant, bottom_point, tip_center, bridge, mbb, tri, torsion, or torsion_edge");

  PetscCall(VecSet(b, 0.0));
  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));
  if (torsion || torsion_edge) {
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (i != grid.nx - 1) continue;
          if (torsion_edge && !h8_is_right_face_edge_node(i, j, k, grid)) continue;
          PetscReal x = 0.0, y = 0.0, z = 0.0;
          h8_node_physical(i, j, k, grid, &x, &y, &z);
          const PetscReal yc = y - 0.5 * domain_width(grid);
          const PetscReal zc = z - 0.5 * domain_height(grid);
          local_torsion_denom += yc * yc + zc * zc;
        }
      }
    }
    PetscCallMPI(MPI_Allreduce(&local_torsion_denom, &global_torsion_denom,
                               1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    PetscCheck(global_torsion_denom > PETSC_SMALL, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_WRONG,
               "Torsion benchmark needs a non-degenerate free-end face");
  }

  PetscCall(DMDAVecGetArrayDOF(uda, b, &bg));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (bridge) {
          // MATLAB initiMesh('bridge'): -Z load on every top-surface node.
          if (k != grid.nz - 1) continue;
          bg[k][j][i][2] += -optimizer_options.load;
        } else if (mbb) {
          // MATLAB initiMesh('mbb'): -Z load on the right-end bottom line.
          if (i != grid.nx - 1 || k != 0) continue;
          bg[k][j][i][2] += -optimizer_options.load;
        } else if (tri) {
          // MATLAB initiMesh('tri'): -X and -Y load at the z-top,
          // y-high, x-left corner.
          if (i != 0 || j != grid.ny - 1 || k != grid.nz - 1) continue;
          bg[k][j][i][0] += -optimizer_options.load;
          bg[k][j][i][1] += -optimizer_options.load;
        } else if (torsion || torsion_edge) {
          if (i != grid.nx - 1) continue;
          if (torsion_edge && !h8_is_right_face_edge_node(i, j, k, grid)) continue;
          PetscReal x = 0.0, y = 0.0, z = 0.0;
          h8_node_physical(i, j, k, grid, &x, &y, &z);
          const PetscReal yc = y - 0.5 * domain_width(grid);
          const PetscReal zc = z - 0.5 * domain_height(grid);
          // 右端面切向力形成绕 x 轴的扭矩，合力近似为零，适合扭转梁 benchmark。
          bg[k][j][i][1] += -optimizer_options.load * zc / global_torsion_denom;
          bg[k][j][i][2] += optimizer_options.load * yc / global_torsion_denom;
        } else if (bottom_point) {
          if (i != grid.nx - 1) continue;
          // Concentrated -Z load at the right-end bottom-center node(s).
          if (k != 0 || (j != load_j0 && j != load_j1)) continue;
          // Split the total force across the one or two centerline nodes.
          bg[k][j][i][2] +=
              -optimizer_options.load /
              static_cast<PetscReal>(cantilever_load_nodes);
        } else if (tip_center) {
          if (i != grid.nx - 1) continue;
          // Concentrated -Z load at the center node(s) of the right-end face.
          if ((j != load_j0 && j != load_j1) ||
              (k != load_k0 && k != load_k1)) continue;
          bg[k][j][i][2] +=
              -optimizer_options.load /
              static_cast<PetscReal>(tip_center_load_nodes);
        } else {
          if (i != grid.nx - 1) continue;
          // MATLAB initiMesh('cant'): -Z load on the right-end mid-height line.
          if (k != matlab_load_k) continue;
          bg[k][j][i][2] += -optimizer_options.load;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(uda, b, &bg));
  return 0;
}

PetscErrorCode fill_h8_load(DM uda, DM eda, Vec mask, const Grid &grid,
                            const DensityOptions &density_options,
                            const OptimizerOptions &optimizer_options, Vec b) {
  PetscInt load_case = 0;
  PetscBool include_spring_load = PETSC_TRUE;
  if (!density_options.use_control_arm_mask) {
    PetscCall(fill_h8_benchmark_load(uda, grid, optimizer_options, b));
    return 0;
  }

  PetscCall(h8_get_load_options(&load_case, &include_spring_load));
  PetscCall(VecSet(b, 0.0));
  if (load_case >= 1 && load_case <= 3) {
    PetscCall(add_h8_control_arm_load_case(uda, eda, mask, grid,
                                           optimizer_options.load, load_case,
                                           1.0, include_spring_load, b));
  } else if (load_case == 4) {
    PetscCall(add_h8_control_arm_symmetric_vertical_load_case(
        uda, eda, mask, grid, optimizer_options.load, 1.0,
        include_spring_load, b));
  } else if (load_case == 0) {
    // load_case=0 保持旧三工况加权组合，不自动混入工况 4。
    for (PetscInt c = 1; c <= 3; ++c) {
      PetscCall(add_h8_control_arm_load_case(uda, eda, mask, grid,
                                             optimizer_options.load, c,
                                             h8_control_arm_case_weight(c),
                                             include_spring_load, b));
    }
  } else {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "H8 load_case must be 0, 1, 2, 3, or 4");
  }
  return 0;
}
PetscErrorCode compute_h8_sensitivity(DM uda, DM eda, Vec u, Vec rho, Vec mask,
                                      const Grid &grid,
                                      const DensityOptions &density_options,
                                      const PetscReal *ke,
                                      PetscBool include_fixed_solid_volume,
                                      Vec dc,
                                      PetscReal *volume_fraction) {
  (void)grid;
  Vec local_u = nullptr;
  PetscScalar ****ug = nullptr;
  PetscScalar ***r = nullptr;
  PetscScalar ***m = nullptr;
  PetscScalar ***d = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_rho_sum = 0.0, local_mask_sum = 0.0;
  PetscReal global_rho_sum = 0.0, global_mask_sum = 0.0;

  PetscCall(DMCreateLocalVector(uda, &local_u));
  PetscCall(DMGlobalToLocalBegin(uda, u, INSERT_VALUES, local_u));
  PetscCall(DMGlobalToLocalEnd(uda, u, INSERT_VALUES, local_u));
  PetscCall(DMDAVecGetArrayDOFRead(uda, local_u, &ug));
  PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAVecGetArray(eda, dc, &d));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt ez = zs; ez < zs + zm; ++ez) {
    for (PetscInt ey = ys; ey < ys + ym; ++ey) {
      for (PetscInt ex = xs; ex < xs + xm; ++ex) {
        const PetscReal mask_value = PetscRealPart(m[ez][ey][ex]);
        const PetscReal rho_value = PetscRealPart(r[ez][ey][ex]);
        PetscReal ue[24] = {};
        PetscReal energy = 0.0;

        if (h8_mask_counts_in_volume(mask_value,
                                     include_fixed_solid_volume)) {
          local_mask_sum += 1.0;
          local_rho_sum += rho_value;
        }
        if (h8_mask_is_void(mask_value)) {
          d[ez][ey][ex] = 0.0;
          continue;
        }

        for (PetscInt node = 0; node < 8; ++node) {
          PetscInt ni = 0, nj = 0, nk = 0;
          h8_node_coords(ex, ey, ez, node, &ni, &nj, &nk);
          for (PetscInt c = 0; c < 3; ++c) {
            ue[3 * node + c] =
                h8_is_fixed_node(ni, nj, nk, grid, density_options)
                    ? 0.0
                    : PetscRealPart(ug[nk][nj][ni][c]);
          }
        }

        for (PetscInt a = 0; a < 24; ++a) {
          for (PetscInt b = 0; b < 24; ++b) {
            energy += ue[a] * ke[24 * a + b] * ue[b];
          }
        }
        d[ez][ey][ex] = h8_mask_is_design(mask_value)
                            ? -simp_derivative(rho_value, density_options) * energy
                            : 0.0;
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(uda, local_u, &ug));
  PetscCall(DMDAVecRestoreArrayRead(eda, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCall(DMDAVecRestoreArray(eda, dc, &d));
  PetscCall(VecDestroy(&local_u));
  PetscCallMPI(MPI_Allreduce(&local_rho_sum, &global_rho_sum, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_mask_sum, &global_mask_sum, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  *volume_fraction = global_mask_sum > 0.0 ? global_rho_sum / global_mask_sum : 0.0;
  return 0;
}

PetscReal trial_rho(PetscReal rho, PetscReal dc, PetscReal dv, PetscReal mask,
                    PetscReal lambda, const OptimizerOptions &opt) {
  if (h8_mask_is_fixed_solid(mask)) return 1.0;
  if (!h8_mask_is_design(mask)) return 0.0;
  if (dv <= 0.0) {
    // 闭包伴随后，部分被其它峰值覆盖的单元可能对闭包后体积导数为 0；
    // 这类退化点仍必须遵守 move limit，不能一步跳到 rho_min。
    return PetscMax(opt.rho_min, rho - opt.move);
  }
  const PetscReal safe_lambda = PetscMax(lambda, 1.0e-300);
  const PetscReal ratio = PetscMax(1.0e-16, -dc / (safe_lambda * dv));
  PetscReal value = rho * PetscSqrtReal(ratio);
  value = PetscMax(rho - opt.move, PetscMin(rho + opt.move, value));
  return PetscMax(opt.rho_min, PetscMin(1.0, value));
}

PetscErrorCode h8_design_weight_sum(DM eda, Vec mask, Vec dv,
                                    PetscReal *weighted_sum) {
  PetscScalar ***m = nullptr, ***v = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_sum = 0.0;

  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAVecGetArrayRead(eda, dv, &v));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (h8_mask_is_design(PetscRealPart(m[k][j][i]))) {
          local_sum += PetscRealPart(v[k][j][i]);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCall(DMDAVecRestoreArrayRead(eda, dv, &v));
  PetscCallMPI(MPI_Allreduce(&local_sum, weighted_sum, 1, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  return 0;
}

PetscErrorCode h8_trial_volume_sum(DM eda, Vec rho, Vec mask, Vec dc, Vec dv,
                                   PetscReal lambda,
                                   const OptimizerOptions &opt,
                                   PetscReal *weighted_sum) {
  PetscScalar ***r = nullptr, ***m = nullptr, ***d = nullptr, ***v = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_sum = 0.0;

  PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAVecGetArrayRead(eda, dc, &d));
  PetscCall(DMDAVecGetArrayRead(eda, dv, &v));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (h8_mask_is_design(PetscRealPart(m[k][j][i]))) {
          const PetscReal xnew =
              trial_rho(PetscRealPart(r[k][j][i]),
                        PetscRealPart(d[k][j][i]),
                        PetscRealPart(v[k][j][i]),
                        PetscRealPart(m[k][j][i]), lambda, opt);
          local_sum += PetscRealPart(v[k][j][i]) * xnew;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCall(DMDAVecRestoreArrayRead(eda, dc, &d));
  PetscCall(DMDAVecRestoreArrayRead(eda, dv, &v));
  PetscCallMPI(MPI_Allreduce(&local_sum, weighted_sum, 1, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  return 0;
}

PetscErrorCode h8_write_trial_design(DM eda, Vec rho, Vec mask, Vec dc, Vec dv,
                                     PetscReal lambda,
                                     const DensityOptions &density_options,
                                     const OptimizerOptions &opt,
                                     Vec trial,
                                     PetscReal *max_change) {
  (void)density_options;
  PetscScalar ***r = nullptr, ***m = nullptr, ***d = nullptr, ***v = nullptr;
  PetscScalar ***t = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_change = 0.0;

  PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAVecGetArrayRead(eda, dc, &d));
  PetscCall(DMDAVecGetArrayRead(eda, dv, &v));
  PetscCall(DMDAVecGetArray(eda, trial, &t));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal old_value = PetscRealPart(r[k][j][i]);
        PetscReal new_value = old_value;
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (h8_mask_is_fixed_solid(mask_value)) {
          new_value = 1.0;
        } else if (h8_mask_is_design(mask_value)) {
          new_value = trial_rho(old_value, PetscRealPart(d[k][j][i]),
                                PetscRealPart(v[k][j][i]), mask_value,
                                lambda, opt);
        }
        t[k][j][i] = new_value;
        local_change =
            PetscMax(local_change, PetscAbsReal(new_value - old_value));
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
  PetscCall(DMDAVecRestoreArrayRead(eda, dc, &d));
  PetscCall(DMDAVecRestoreArrayRead(eda, dv, &v));
  PetscCall(DMDAVecRestoreArray(eda, trial, &t));
  if (max_change != nullptr) {
    PetscCallMPI(MPI_Allreduce(&local_change, max_change, 1, MPIU_REAL,
                               MPI_MAX, PETSC_COMM_WORLD));
  }
  return 0;
}

PetscErrorCode h8_projected_trial_volume(DM eda, Vec rho, Vec phys_mask,
                                         Vec update_mask, Vec dc, Vec dv,
                                         Vec filter_denom,
                                         PetscReal lambda, PetscReal beta,
                                         const DensityOptions &density_options,
                                         const OptimizerOptions &opt,
                                         PetscBool include_fixed_solid_volume,
                                         Vec trial_design,
                                         Vec trial_filtered,
                                         Vec trial_phys,
                                         PetscReal *volume) {
  PetscCall(h8_write_trial_design(eda, rho, update_mask, dc, dv, lambda,
                                  density_options, opt, trial_design,
                                  nullptr));
  PetscCall(build_h8_physical_density(eda, trial_design, phys_mask,
                                      filter_denom, density_options, opt,
                                      beta, trial_filtered, trial_phys));
  PetscCall(compute_masked_volume_fraction_with_fixed(
      eda, trial_phys, phys_mask, include_fixed_solid_volume, volume));
  return 0;
}

PetscErrorCode oc_update_projected_volume(DM eda, Vec rho, Vec phys_mask,
                                          Vec update_mask, Vec dc, Vec dv,
                                          Vec filter_denom,
                                          PetscReal beta,
                                          const DensityOptions &density_options,
                                          const OptimizerOptions &opt,
                                          PetscBool include_fixed_solid_volume,
                                          PetscReal *max_change) {
  Vec trial_design = nullptr;
  Vec trial_filtered = nullptr;
  Vec trial_phys = nullptr;
  const PetscReal target = opt.volfrac;
  PetscReal l1 = 1.0e-300;
  PetscReal l2 = 1.0e-24;
  PetscReal best_lambda = l1;
  PetscReal best_error = PETSC_MAX_REAL;

  PetscCall(VecDuplicate(rho, &trial_design));
  PetscCall(VecDuplicate(rho, &trial_filtered));
  PetscCall(VecDuplicate(rho, &trial_phys));

  auto consider = [&](PetscReal lambda, PetscReal volume) {
    const PetscReal error = PetscAbsReal(volume - target);
    if (error < best_error) {
      best_error = error;
      best_lambda = lambda;
    }
  };

  PetscReal v1 = 0.0;
  PetscReal v2 = 0.0;
  PetscCall(h8_projected_trial_volume(eda, rho, phys_mask, update_mask, dc, dv,
                                      filter_denom, l1, beta, density_options,
                                      opt, include_fixed_solid_volume,
                                      trial_design, trial_filtered, trial_phys,
                                      &v1));
  consider(l1, v1);
  PetscCall(h8_projected_trial_volume(eda, rho, phys_mask, update_mask, dc, dv,
                                      filter_denom, l2, beta, density_options,
                                      opt, include_fixed_solid_volume,
                                      trial_design, trial_filtered, trial_phys,
                                      &v2));
  consider(l2, v2);
  while (v2 > target && l2 < 1.0e100) {
    l1 = l2;
    v1 = v2;
    l2 *= 10.0;
    PetscCall(h8_projected_trial_volume(
        eda, rho, phys_mask, update_mask, dc, dv, filter_denom, l2, beta,
        density_options, opt, include_fixed_solid_volume, trial_design,
        trial_filtered, trial_phys, &v2));
    consider(l2, v2);
  }

  if (v1 > target && v2 <= target) {
    for (PetscInt it = 0; it < 40; ++it) {
      const PetscReal lambda = 0.5 * (l1 + l2);
      PetscReal volume = 0.0;
      PetscCall(h8_projected_trial_volume(
          eda, rho, phys_mask, update_mask, dc, dv, filter_denom, lambda, beta,
          density_options, opt, include_fixed_solid_volume, trial_design,
          trial_filtered, trial_phys, &volume));
      consider(lambda, volume);
      if (volume > target) {
        l1 = lambda;
      } else {
        l2 = lambda;
      }
    }
  }

  const PetscReal final_lambda = best_lambda;
  // 拔模闭包可能让 trial 体积呈现离散跳变；选离目标最近的 trial，
  // 允许体积在目标附近轻微上下摆动，避免单侧欠用体积造成反复震荡。
  PetscCall(h8_write_trial_design(eda, rho, update_mask, dc, dv, final_lambda,
                                  density_options, opt, trial_design,
                                  max_change));
  PetscCall(VecCopy(trial_design, rho));
  PetscCall(VecDestroy(&trial_phys));
  PetscCall(VecDestroy(&trial_filtered));
  PetscCall(VecDestroy(&trial_design));
  return 0;
}

PetscErrorCode oc_update(DM eda, Vec rho, Vec mask, Vec dc, Vec dv,
                         const DensityOptions &density_options,
                         const OptimizerOptions &opt,
                         PetscReal *max_change) {
  (void)density_options;
  PetscReal l1 = 1.0e-300, l2 = 1.0e-300;
  PetscReal design_weight_sum = 0.0;
  PetscCall(h8_design_weight_sum(eda, mask, dv, &design_weight_sum));
  const PetscReal target_volume_sum = opt.volfrac * design_weight_sum;
  PetscReal low_sum = 0.0, high_sum = 0.0;

  PetscCall(h8_trial_volume_sum(eda, rho, mask, dc, dv, l1, opt, &low_sum));
  l2 = 1.0e-24;
  PetscCall(h8_trial_volume_sum(eda, rho, mask, dc, dv, l2, opt, &high_sum));
  while (high_sum > target_volume_sum && l2 < 1.0e100) {
    l1 = l2;
    low_sum = high_sum;
    l2 *= 10.0;
    PetscCall(h8_trial_volume_sum(eda, rho, mask, dc, dv, l2, opt, &high_sum));
  }

  if (low_sum < target_volume_sum) {
    l2 = l1;
  } else if (high_sum > target_volume_sum) {
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "Warning: OC update could not reduce trial volume to target within lambda range; using largest bracket lambda.\n"));
  } else {
    for (PetscInt it = 0; it < 100; ++it) {
      const PetscReal lambda = 0.5 * (l1 + l2);
      PetscReal global_sum = 0.0;
      PetscCall(h8_trial_volume_sum(eda, rho, mask, dc, dv, lambda, opt,
                                    &global_sum));
      if (global_sum > target_volume_sum) l1 = lambda;
      else l2 = lambda;
    }
  }

  {
    const PetscReal lambda = 0.5 * (l1 + l2);
    PetscScalar ***r = nullptr, ***m = nullptr, ***d = nullptr, ***v = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscReal local_change = 0.0;
    PetscCall(DMDAVecGetArray(eda, rho, &r));
    PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
    PetscCall(DMDAVecGetArrayRead(eda, dc, &d));
    PetscCall(DMDAVecGetArrayRead(eda, dv, &v));
    PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          const PetscReal old_value = PetscRealPart(r[k][j][i]);
          PetscReal new_value = old_value;
          const PetscReal mask_value = PetscRealPart(m[k][j][i]);
          if (h8_mask_is_fixed_solid(mask_value)) {
            new_value = 1.0;
          } else if (h8_mask_is_design(mask_value)) {
            new_value = trial_rho(old_value, PetscRealPart(d[k][j][i]),
                                  PetscRealPart(v[k][j][i]),
                                  mask_value, lambda, opt);
          }
          r[k][j][i] = new_value;
          local_change = PetscMax(local_change, PetscAbsReal(new_value - old_value));
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(eda, rho, &r));
    PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
    PetscCall(DMDAVecRestoreArrayRead(eda, dc, &d));
    PetscCall(DMDAVecRestoreArrayRead(eda, dv, &v));
    PetscCallMPI(MPI_Allreduce(&local_change, max_change, 1, MPIU_REAL, MPI_MAX,
                               PETSC_COMM_WORLD));
  }
  return 0;
}

PetscErrorCode write_h8_opt_vtk(DM uda, DM eda, const char *path,
                                const Grid &grid, Vec u, Vec rho, Vec mask) {
  PetscMPIInt rank = 0;
  Vec u_nat = nullptr, rho_nat = nullptr, mask_nat = nullptr;
  Vec u_seq = nullptr, rho_seq = nullptr, mask_seq = nullptr;
  VecScatter su = nullptr, sr = nullptr, sm = nullptr;
  const PetscScalar *ua = nullptr, *ra = nullptr, *ma = nullptr;
  const PetscInt nelem = (grid.nx - 1) * (grid.ny - 1) * (grid.nz - 1);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(DMDACreateNaturalVector(uda, &u_nat));
  PetscCall(DMDACreateNaturalVector(eda, &rho_nat));
  PetscCall(DMDACreateNaturalVector(eda, &mask_nat));
  PetscCall(DMDAGlobalToNaturalBegin(uda, u, INSERT_VALUES, u_nat));
  PetscCall(DMDAGlobalToNaturalEnd(uda, u, INSERT_VALUES, u_nat));
  PetscCall(DMDAGlobalToNaturalBegin(eda, rho, INSERT_VALUES, rho_nat));
  PetscCall(DMDAGlobalToNaturalEnd(eda, rho, INSERT_VALUES, rho_nat));
  PetscCall(DMDAGlobalToNaturalBegin(eda, mask, INSERT_VALUES, mask_nat));
  PetscCall(DMDAGlobalToNaturalEnd(eda, mask, INSERT_VALUES, mask_nat));
  PetscCall(VecScatterCreateToZero(u_nat, &su, &u_seq));
  PetscCall(VecScatterCreateToZero(rho_nat, &sr, &rho_seq));
  PetscCall(VecScatterCreateToZero(mask_nat, &sm, &mask_seq));
  PetscCall(VecScatterBegin(su, u_nat, u_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(su, u_nat, u_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(sr, rho_nat, rho_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sr, rho_nat, rho_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(sm, mask_nat, mask_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sm, mask_nat, mask_seq, INSERT_VALUES, SCATTER_FORWARD));

  if (rank == 0) {
    FILE *fp = std::fopen(path, "w");
    PetscCheck(fp != nullptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Cannot open H8 optimizer VTK: %s", path);
    PetscCall(VecGetArrayRead(u_seq, &ua));
    PetscCall(VecGetArrayRead(rho_seq, &ra));
    PetscCall(VecGetArrayRead(mask_seq, &ma));
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "C++ PETSc H8 matrix-free optimization result\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    std::fprintf(fp, "DIMENSIONS %lld %lld %lld\n", static_cast<long long>(grid.nx),
                 static_cast<long long>(grid.ny), static_cast<long long>(grid.nz));
    std::fprintf(fp, "ORIGIN 0 0 0\n");
    std::fprintf(fp, "SPACING 1 1 1\n");
    std::fprintf(fp, "POINT_DATA %lld\n", static_cast<long long>(node_count(grid)));
    std::fprintf(fp, "VECTORS displacement double\n");
    for (PetscInt id = 0; id < node_count(grid); ++id) {
      std::fprintf(fp, "%.17e %.17e %.17e\n",
                   static_cast<double>(PetscRealPart(ua[3 * id + 0])),
                   static_cast<double>(PetscRealPart(ua[3 * id + 1])),
                   static_cast<double>(PetscRealPart(ua[3 * id + 2])));
    }
    std::fprintf(fp, "CELL_DATA %lld\n", static_cast<long long>(nelem));
    std::fprintf(fp, "SCALARS rho double 1\nLOOKUP_TABLE default\n");
    for (PetscInt id = 0; id < nelem; ++id) {
      std::fprintf(fp, "%.17e\n", static_cast<double>(PetscRealPart(ra[id])));
    }
    std::fprintf(fp, "SCALARS design_mask double 1\nLOOKUP_TABLE default\n");
    for (PetscInt id = 0; id < nelem; ++id) {
      std::fprintf(fp, "%.17e\n", static_cast<double>(PetscRealPart(ma[id])));
    }
    std::fprintf(fp, "SCALARS fixed_solid_mask double 1\nLOOKUP_TABLE default\n");
    for (PetscInt id = 0; id < nelem; ++id) {
      std::fprintf(fp, "%.17e\n",
                   h8_mask_is_fixed_solid(PetscRealPart(ma[id])) ? 1.0 : 0.0);
    }
    PetscCall(VecRestoreArrayRead(u_seq, &ua));
    PetscCall(VecRestoreArrayRead(rho_seq, &ra));
    PetscCall(VecRestoreArrayRead(mask_seq, &ma));
    std::fclose(fp);
  }

  PetscCall(VecScatterDestroy(&su));
  PetscCall(VecScatterDestroy(&sr));
  PetscCall(VecScatterDestroy(&sm));
  PetscCall(VecDestroy(&u_seq));
  PetscCall(VecDestroy(&rho_seq));
  PetscCall(VecDestroy(&mask_seq));
  PetscCall(VecDestroy(&u_nat));
  PetscCall(VecDestroy(&rho_nat));
  PetscCall(VecDestroy(&mask_nat));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Wrote H8 optimization VTK: %s\n", path));
  return 0;
}

PetscErrorCode write_downsampled_h8_density_vtk(DM eda, const char *path,
                                                const Grid &grid, Vec rho,
                                                Vec mask, PetscInt stride,
                                                PetscInt max_samples) {
  PetscMPIInt rank = 0;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscScalar ***r = nullptr;
  PetscScalar ***m = nullptr;
  const PetscInt ex = grid.nx - 1;
  const PetscInt ey = grid.ny - 1;
  const PetscInt ez = grid.nz - 1;
  const PetscInt s = PetscMax(1, stride);
  const PetscInt ncx = PetscMax(1, ceil_div(ex, s));
  const PetscInt ncy = PetscMax(1, ceil_div(ey, s));
  const PetscInt ncz = PetscMax(1, ceil_div(ez, s));
  const PetscInt sample_count = ncx * ncy * ncz;
  PetscMPIInt reduce_count = 0;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(max_samples <= 0 || sample_count <= max_samples, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_OUTOFRANGE,
             "Downsampled VTK would contain %lld cells, above -post_max_samples=%lld; increase -post_stride",
             static_cast<long long>(sample_count),
             static_cast<long long>(max_samples));
  PetscCall(PetscMPIIntCast(sample_count, &reduce_count));

  std::vector<PetscReal> local_rho(static_cast<std::size_t>(sample_count), -1.0);
  std::vector<PetscReal> local_mask(static_cast<std::size_t>(sample_count), -1.0);
  std::vector<PetscReal> global_rho;
  std::vector<PetscReal> global_mask;
  if (rank == 0) {
    global_rho.assign(static_cast<std::size_t>(sample_count), -1.0);
    global_mask.assign(static_cast<std::size_t>(sample_count), -1.0);
  }

  PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt cz = 0; cz < ncz; ++cz) {
    const PetscInt k = PetscMin(ez - 1, cz * s + s / 2);
    if (k < zs || k >= zs + zm) continue;
    for (PetscInt cy = 0; cy < ncy; ++cy) {
      const PetscInt j = PetscMin(ey - 1, cy * s + s / 2);
      if (j < ys || j >= ys + ym) continue;
      for (PetscInt cx = 0; cx < ncx; ++cx) {
        const PetscInt i = PetscMin(ex - 1, cx * s + s / 2);
        if (i < xs || i >= xs + xm) continue;
        const PetscInt idx = (cz * ncy + cy) * ncx + cx;
        local_rho[static_cast<std::size_t>(idx)] = PetscRealPart(r[k][j][i]);
        local_mask[static_cast<std::size_t>(idx)] = PetscRealPart(m[k][j][i]);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));

  PetscCallMPI(MPI_Reduce(local_rho.data(),
                          rank == 0 ? global_rho.data() : nullptr,
                          reduce_count, MPIU_REAL, MPI_MAX, 0, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(local_mask.data(),
                          rank == 0 ? global_mask.data() : nullptr,
                          reduce_count, MPIU_REAL, MPI_MAX, 0, PETSC_COMM_WORLD));

  if (rank == 0) {
    FILE *fp = std::fopen(path, "w");
    PetscCheck(fp != nullptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Could not open %s for writing", path);
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "H8 checkpoint density downsample\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    std::fprintf(fp, "DIMENSIONS %lld %lld %lld\n",
                 static_cast<long long>(ncx + 1),
                 static_cast<long long>(ncy + 1),
                 static_cast<long long>(ncz + 1));
    std::fprintf(fp, "ORIGIN 0 0 0\n");
    std::fprintf(fp, "SPACING %.17g %.17g %.17g\n",
                 static_cast<double>(domain_length(grid) / static_cast<PetscReal>(ncx)),
                 static_cast<double>(domain_width(grid) / static_cast<PetscReal>(ncy)),
                 static_cast<double>(domain_height(grid) / static_cast<PetscReal>(ncz)));
    std::fprintf(fp, "CELL_DATA %lld\n", static_cast<long long>(sample_count));

    std::fprintf(fp, "SCALARS rho double 1\nLOOKUP_TABLE default\n");
    for (PetscInt idx = 0; idx < sample_count; ++idx) {
      const PetscReal value = PetscMax(0.0, global_rho[static_cast<std::size_t>(idx)]);
      std::fprintf(fp, "%.12e\n", static_cast<double>(value));
    }
    std::fprintf(fp, "SCALARS rho_masked double 1\nLOOKUP_TABLE default\n");
    for (PetscInt idx = 0; idx < sample_count; ++idx) {
      const PetscReal rho_value =
          PetscMax(0.0, global_rho[static_cast<std::size_t>(idx)]);
      const PetscReal mask_value =
          PetscMax(0.0, global_mask[static_cast<std::size_t>(idx)]);
      std::fprintf(fp, "%.12e\n",
                   static_cast<double>(mask_value > 0.5 ? rho_value : 0.0));
    }
    std::fprintf(fp, "SCALARS design_mask double 1\nLOOKUP_TABLE default\n");
    for (PetscInt idx = 0; idx < sample_count; ++idx) {
      const PetscReal value = PetscMax(0.0, global_mask[static_cast<std::size_t>(idx)]);
      std::fprintf(fp, "%.12e\n", static_cast<double>(value));
    }
    std::fprintf(fp, "SCALARS fixed_solid_mask double 1\nLOOKUP_TABLE default\n");
    for (PetscInt idx = 0; idx < sample_count; ++idx) {
      const PetscReal value = PetscMax(0.0, global_mask[static_cast<std::size_t>(idx)]);
      std::fprintf(fp, "%.12e\n",
                   static_cast<double>(h8_mask_is_fixed_solid(value) ? 1.0 : 0.0));
    }
    std::fprintf(fp, "SCALARS fixed_c_ring double 1\nLOOKUP_TABLE default\n");
    for (PetscInt cz = 0; cz < ncz; ++cz) {
      const PetscInt k = PetscMin(ez - 1, cz * s + s / 2);
      for (PetscInt cy = 0; cy < ncy; ++cy) {
        const PetscInt j = PetscMin(ey - 1, cy * s + s / 2);
        for (PetscInt cx = 0; cx < ncx; ++cx) {
          const PetscInt i = PetscMin(ex - 1, cx * s + s / 2);
          std::fprintf(fp, "%.12e\n",
                       h8_cell_has_c_support_node(i, j, k, grid) ? 1.0 : 0.0);
        }
      }
    }
    std::fprintf(fp, "SCALARS load_ab_rings double 1\nLOOKUP_TABLE default\n");
    for (PetscInt cz = 0; cz < ncz; ++cz) {
      const PetscInt k = PetscMin(ez - 1, cz * s + s / 2);
      for (PetscInt cy = 0; cy < ncy; ++cy) {
        const PetscInt j = PetscMin(ey - 1, cy * s + s / 2);
        for (PetscInt cx = 0; cx < ncx; ++cx) {
          const PetscInt i = PetscMin(ex - 1, cx * s + s / 2);
          std::fprintf(fp, "%.12e\n",
                       (h8_cell_has_ab_load_node(i, j, k, grid, 1) ||
                        h8_cell_has_ab_load_node(i, j, k, grid, 2))
                           ? 1.0
                           : 0.0);
        }
      }
    }
    std::fprintf(fp, "SCALARS spring_load_mount double 1\nLOOKUP_TABLE default\n");
    for (PetscInt cz = 0; cz < ncz; ++cz) {
      const PetscInt k = PetscMin(ez - 1, cz * s + s / 2);
      for (PetscInt cy = 0; cy < ncy; ++cy) {
        const PetscInt j = PetscMin(ey - 1, cy * s + s / 2);
        for (PetscInt cx = 0; cx < ncx; ++cx) {
          const PetscInt i = PetscMin(ex - 1, cx * s + s / 2);
          std::fprintf(fp, "%.12e\n",
                       h8_cell_has_spring_load_node(i, j, k, grid) ? 1.0 : 0.0);
        }
      }
    }
    std::fclose(fp);
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Wrote downsampled H8 density VTK: %s stride=%lld cells=%lld (%lld x %lld x %lld)\n",
                        path, static_cast<long long>(s),
                        static_cast<long long>(sample_count),
                        static_cast<long long>(ncx),
                        static_cast<long long>(ncy),
                        static_cast<long long>(ncz)));
  return 0;
}

void split_output_path(const char *path, char *dir, std::size_t dir_size,
                       char *stem, std::size_t stem_size) {
  const char *slash = std::strrchr(path, '/');
  const char *backslash = std::strrchr(path, '\\');
  const char *last_sep = slash;
  if (backslash && (!last_sep || backslash > last_sep)) last_sep = backslash;
  const char *base = last_sep ? last_sep + 1 : path;
  if (last_sep) {
    const std::size_t len =
        PetscMin(static_cast<std::size_t>(last_sep - path), dir_size - 1);
    std::memcpy(dir, path, len);
    dir[len] = '\0';
  } else {
    PetscSNPrintf(dir, dir_size, ".");
  }

  PetscSNPrintf(stem, stem_size, "%s", base);
  char *dot = std::strrchr(stem, '.');
  if (dot && dot != stem) *dot = '\0';
}

PetscErrorCode write_full_h8_parallel_pvti(DM uda, DM eda, const char *path,
                                           const Grid &grid, Vec u, Vec rho,
                                           Vec mask) {
  PetscMPIInt rank = 0, ranks = 1;
  PetscInt exs = 0, eys = 0, ezs = 0, exm = 0, eym = 0, ezm = 0;
  PetscInt local_extent[6] = {0, 0, 0, 0, 0, 0};
  std::vector<PetscInt> all_extents;
  char out_dir[PETSC_MAX_PATH_LEN];
  char stem[PETSC_MAX_PATH_LEN];
  char piece_dir_name[PETSC_MAX_PATH_LEN];
  char piece_dir_path[PETSC_MAX_PATH_LEN];
  char piece_path[PETSC_MAX_PATH_LEN];

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCall(DMDAGetCorners(eda, &exs, &eys, &ezs, &exm, &eym, &ezm));
  local_extent[0] = exs;
  local_extent[1] = exs + exm;
  local_extent[2] = eys;
  local_extent[3] = eys + eym;
  local_extent[4] = ezs;
  local_extent[5] = ezs + ezm;

  if (rank == 0) {
    all_extents.resize(static_cast<std::size_t>(6 * ranks));
  }
  PetscCallMPI(MPI_Gather(local_extent, 6, MPIU_INT,
                          rank == 0 ? all_extents.data() : nullptr,
                          6, MPIU_INT, 0, PETSC_COMM_WORLD));

  split_output_path(path, out_dir, sizeof(out_dir), stem, sizeof(stem));
  PetscCall(PetscSNPrintf(piece_dir_name, sizeof(piece_dir_name), "%s_pieces", stem));
  if (std::strcmp(out_dir, ".") == 0) {
    PetscCall(PetscSNPrintf(piece_dir_path, sizeof(piece_dir_path), "%s",
                            piece_dir_name));
  } else {
    PetscCall(PetscSNPrintf(piece_dir_path, sizeof(piece_dir_path), "%s/%s",
                            out_dir, piece_dir_name));
  }
  if (rank == 0) {
    PetscCall(PetscMkdir(piece_dir_path));
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  if (std::strcmp(out_dir, ".") == 0) {
    PetscCall(PetscSNPrintf(piece_path, sizeof(piece_path), "%s/%s_rank%06d.vti",
                            piece_dir_name, stem, static_cast<int>(rank)));
  } else {
    PetscCall(PetscSNPrintf(piece_path, sizeof(piece_path),
                            "%s/%s_rank%06d.vti", piece_dir_path, stem,
                            static_cast<int>(rank)));
  }

  {
    Vec local_u = nullptr;
    PetscScalar ****ug = nullptr;
    PetscScalar ***r = nullptr;
    PetscScalar ***m = nullptr;
    FILE *fp = nullptr;
    const PetscReal hx =
        domain_length(grid) / static_cast<PetscReal>(grid.nx - 1);
    const PetscReal hy =
        domain_width(grid) / static_cast<PetscReal>(grid.ny - 1);
    const PetscReal hz =
        domain_height(grid) / static_cast<PetscReal>(grid.nz - 1);

    PetscCall(DMCreateLocalVector(uda, &local_u));
    PetscCall(DMGlobalToLocalBegin(uda, u, INSERT_VALUES, local_u));
    PetscCall(DMGlobalToLocalEnd(uda, u, INSERT_VALUES, local_u));
    PetscCall(DMDAVecGetArrayDOFRead(uda, local_u, &ug));
    PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
    PetscCall(DMDAVecGetArrayRead(eda, mask, &m));

    fp = std::fopen(piece_path, "w");
    PetscCheck(fp != nullptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Could not open %s for writing", piece_path);
    std::fprintf(fp, "<?xml version=\"1.0\"?>\n");
    std::fprintf(fp,
                 "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    std::fprintf(fp,
                 "  <ImageData WholeExtent=\"%lld %lld %lld %lld %lld %lld\" Origin=\"0 0 0\" Spacing=\"%.17g %.17g %.17g\">\n",
                 static_cast<long long>(local_extent[0]),
                 static_cast<long long>(local_extent[1]),
                 static_cast<long long>(local_extent[2]),
                 static_cast<long long>(local_extent[3]),
                 static_cast<long long>(local_extent[4]),
                 static_cast<long long>(local_extent[5]),
                 static_cast<double>(hx), static_cast<double>(hy),
                 static_cast<double>(hz));
    std::fprintf(fp,
                 "    <Piece Extent=\"%lld %lld %lld %lld %lld %lld\">\n",
                 static_cast<long long>(local_extent[0]),
                 static_cast<long long>(local_extent[1]),
                 static_cast<long long>(local_extent[2]),
                 static_cast<long long>(local_extent[3]),
                 static_cast<long long>(local_extent[4]),
                 static_cast<long long>(local_extent[5]));

    std::fprintf(fp, "      <PointData Vectors=\"displacement\" Scalars=\"displacement_magnitude\">\n");
    std::fprintf(fp,
                 "        <DataArray type=\"Float64\" Name=\"displacement\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (PetscInt k = ezs; k <= ezs + ezm; ++k) {
      for (PetscInt j = eys; j <= eys + eym; ++j) {
        for (PetscInt i = exs; i <= exs + exm; ++i) {
          std::fprintf(fp, "%.12e %.12e %.12e\n",
                       static_cast<double>(PetscRealPart(ug[k][j][i][0])),
                       static_cast<double>(PetscRealPart(ug[k][j][i][1])),
                       static_cast<double>(PetscRealPart(ug[k][j][i][2])));
        }
      }
    }
    std::fprintf(fp, "        </DataArray>\n");
    std::fprintf(fp,
                 "        <DataArray type=\"Float64\" Name=\"displacement_magnitude\" NumberOfComponents=\"1\" format=\"ascii\">\n");
    for (PetscInt k = ezs; k <= ezs + ezm; ++k) {
      for (PetscInt j = eys; j <= eys + eym; ++j) {
        for (PetscInt i = exs; i <= exs + exm; ++i) {
          const PetscReal ux = PetscRealPart(ug[k][j][i][0]);
          const PetscReal uy = PetscRealPart(ug[k][j][i][1]);
          const PetscReal uz = PetscRealPart(ug[k][j][i][2]);
          std::fprintf(fp, "%.12e\n",
                       static_cast<double>(PetscSqrtReal(ux * ux + uy * uy + uz * uz)));
        }
      }
    }
    std::fprintf(fp, "        </DataArray>\n");
    std::fprintf(fp, "      </PointData>\n");

    std::fprintf(fp, "      <CellData Scalars=\"rho\">\n");
    std::fprintf(fp,
                 "        <DataArray type=\"Float64\" Name=\"rho\" NumberOfComponents=\"1\" format=\"ascii\">\n");
    for (PetscInt k = ezs; k < ezs + ezm; ++k) {
      for (PetscInt j = eys; j < eys + eym; ++j) {
        for (PetscInt i = exs; i < exs + exm; ++i) {
          std::fprintf(fp, "%.12e\n",
                       static_cast<double>(PetscRealPart(r[k][j][i])));
        }
      }
    }
    std::fprintf(fp, "        </DataArray>\n");
    std::fprintf(fp,
                 "        <DataArray type=\"Float64\" Name=\"rho_masked\" NumberOfComponents=\"1\" format=\"ascii\">\n");
    for (PetscInt k = ezs; k < ezs + ezm; ++k) {
      for (PetscInt j = eys; j < eys + eym; ++j) {
        for (PetscInt i = exs; i < exs + exm; ++i) {
          const PetscReal mask_value = PetscRealPart(m[k][j][i]);
          const PetscReal rho_value = PetscRealPart(r[k][j][i]);
          std::fprintf(fp, "%.12e\n",
                       static_cast<double>(mask_value > 0.5 ? rho_value : 0.0));
        }
      }
    }
    std::fprintf(fp, "        </DataArray>\n");
    std::fprintf(fp,
                 "        <DataArray type=\"Float64\" Name=\"design_mask\" NumberOfComponents=\"1\" format=\"ascii\">\n");
    for (PetscInt k = ezs; k < ezs + ezm; ++k) {
      for (PetscInt j = eys; j < eys + eym; ++j) {
        for (PetscInt i = exs; i < exs + exm; ++i) {
          std::fprintf(fp, "%.12e\n",
                       static_cast<double>(PetscRealPart(m[k][j][i])));
        }
      }
    }
    std::fprintf(fp, "        </DataArray>\n");
    std::fprintf(fp,
                 "        <DataArray type=\"Float64\" Name=\"fixed_solid_mask\" NumberOfComponents=\"1\" format=\"ascii\">\n");
    for (PetscInt k = ezs; k < ezs + ezm; ++k) {
      for (PetscInt j = eys; j < eys + eym; ++j) {
        for (PetscInt i = exs; i < exs + exm; ++i) {
          const PetscReal mask_value = PetscRealPart(m[k][j][i]);
          std::fprintf(fp, "%.12e\n",
                       static_cast<double>(h8_mask_is_fixed_solid(mask_value)
                                               ? 1.0
                                               : 0.0));
        }
      }
    }
    std::fprintf(fp, "        </DataArray>\n");
    std::fprintf(fp, "      </CellData>\n");
    std::fprintf(fp, "    </Piece>\n");
    std::fprintf(fp, "  </ImageData>\n");
    std::fprintf(fp, "</VTKFile>\n");
    std::fclose(fp);

    PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
    PetscCall(DMDAVecRestoreArrayRead(eda, rho, &r));
    PetscCall(DMDAVecRestoreArrayDOFRead(uda, local_u, &ug));
    PetscCall(VecDestroy(&local_u));
  }

  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  if (rank == 0) {
    FILE *fp = std::fopen(path, "w");
    const PetscReal hx =
        domain_length(grid) / static_cast<PetscReal>(grid.nx - 1);
    const PetscReal hy =
        domain_width(grid) / static_cast<PetscReal>(grid.ny - 1);
    const PetscReal hz =
        domain_height(grid) / static_cast<PetscReal>(grid.nz - 1);
    PetscCheck(fp != nullptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Could not open %s for writing", path);
    std::fprintf(fp, "<?xml version=\"1.0\"?>\n");
    std::fprintf(fp,
                 "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    std::fprintf(fp,
                 "  <PImageData WholeExtent=\"0 %lld 0 %lld 0 %lld\" GhostLevel=\"0\" Origin=\"0 0 0\" Spacing=\"%.17g %.17g %.17g\">\n",
                 static_cast<long long>(grid.nx - 1),
                 static_cast<long long>(grid.ny - 1),
                 static_cast<long long>(grid.nz - 1),
                 static_cast<double>(hx), static_cast<double>(hy),
                 static_cast<double>(hz));
    std::fprintf(fp, "    <PPointData Vectors=\"displacement\" Scalars=\"displacement_magnitude\">\n");
    std::fprintf(fp,
                 "      <PDataArray type=\"Float64\" Name=\"displacement\" NumberOfComponents=\"3\"/>\n");
    std::fprintf(fp,
                 "      <PDataArray type=\"Float64\" Name=\"displacement_magnitude\" NumberOfComponents=\"1\"/>\n");
    std::fprintf(fp, "    </PPointData>\n");
    std::fprintf(fp, "    <PCellData Scalars=\"rho\">\n");
    std::fprintf(fp,
                 "      <PDataArray type=\"Float64\" Name=\"rho\" NumberOfComponents=\"1\"/>\n");
    std::fprintf(fp,
                 "      <PDataArray type=\"Float64\" Name=\"rho_masked\" NumberOfComponents=\"1\"/>\n");
    std::fprintf(fp,
                 "      <PDataArray type=\"Float64\" Name=\"design_mask\" NumberOfComponents=\"1\"/>\n");
    std::fprintf(fp,
                 "      <PDataArray type=\"Float64\" Name=\"fixed_solid_mask\" NumberOfComponents=\"1\"/>\n");
    std::fprintf(fp, "    </PCellData>\n");
    for (PetscMPIInt pr = 0; pr < ranks; ++pr) {
      const PetscInt *e = &all_extents[static_cast<std::size_t>(6 * pr)];
      std::fprintf(fp,
                   "    <Piece Extent=\"%lld %lld %lld %lld %lld %lld\" Source=\"%s/%s_rank%06d.vti\"/>\n",
                   static_cast<long long>(e[0]),
                   static_cast<long long>(e[1]),
                   static_cast<long long>(e[2]),
                   static_cast<long long>(e[3]),
                   static_cast<long long>(e[4]),
                   static_cast<long long>(e[5]),
                   piece_dir_name, stem, static_cast<int>(pr));
    }
    std::fprintf(fp, "  </PImageData>\n");
    std::fprintf(fp, "</VTKFile>\n");
    std::fclose(fp);
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Wrote full H8 parallel VTK: %s plus %d piece files in %s\n",
                        path, static_cast<int>(ranks), piece_dir_path));
  return 0;
}

PetscErrorCode write_h8_summary(const char *output_prefix, const Grid &grid,
                                const OptimizerOptions &opt,
                                PetscInt final_iter, PetscReal compliance,
                                PetscReal volume, PetscReal change,
                                PetscReal total_time_s) {
  PetscViewer viewer = nullptr;
  char path[PETSC_MAX_PATH_LEN];
  PetscInt load_case = 0;
  PetscBool include_spring_load = PETSC_TRUE;
  PetscLogDouble mem_current = 0.0, mem_peak = 0.0;
  PetscLogDouble mem_current_max = 0.0, mem_peak_max = 0.0;
  PetscCall(h8_get_load_options(&load_case, &include_spring_load));
  PetscCall(PetscMemoryGetCurrentUsage(&mem_current));
  PetscCall(PetscMemoryGetMaximumUsage(&mem_peak));
  PetscCallMPI(MPI_Reduce(&mem_current, &mem_current_max, 1, MPI_DOUBLE, MPI_MAX,
                          0, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&mem_peak, &mem_peak_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                          PETSC_COMM_WORLD));
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s_opt_summary.txt", output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, path, &viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "mode=optimize\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "operator=h8_matrix_free\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "nx=%lld\nny=%lld\nnz=%lld\n",
                                   static_cast<long long>(grid.nx),
                                   static_cast<long long>(grid.ny),
                                   static_cast<long long>(grid.nz)));
  PetscCall(PetscViewerASCIIPrintf(viewer,
                                   "domain_length_m=%.12e\ndomain_width_m=%.12e\ndomain_height_m=%.12e\n",
                                   static_cast<double>(domain_length(grid)),
                                   static_cast<double>(domain_width(grid)),
                                   static_cast<double>(domain_height(grid))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "dof=%lld\n",
                                   static_cast<long long>(dof_count(grid))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "elements=%lld\n",
                                   static_cast<long long>((grid.nx - 1) *
                                                          (grid.ny - 1) *
                                                          (grid.nz - 1))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_iter=%lld\n",
                                   static_cast<long long>(opt.max_iter)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "volfrac_target=%.12e\n",
                                   static_cast<double>(opt.volfrac)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "filter_radius=%.12e\n",
                                   static_cast<double>(opt.filter_radius)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "heaviside_projection=%s\n",
                                   opt.heaviside_projection ? "true"
                                                            : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "heaviside_eta=%.12e\n",
                                   static_cast<double>(opt.heaviside_eta)));
  PetscCall(PetscViewerASCIIPrintf(viewer,
                                   "heaviside_beta_initial=%.12e\n",
                                   static_cast<double>(
                                       opt.heaviside_beta_initial)));
  PetscCall(PetscViewerASCIIPrintf(viewer,
                                   "heaviside_beta_max=%.12e\n",
                                   static_cast<double>(
                                       opt.heaviside_beta_max)));
  PetscCall(PetscViewerASCIIPrintf(viewer,
                                   "heaviside_beta_interval=%lld\n",
                                   static_cast<long long>(
                                       opt.heaviside_beta_interval)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_closure=%s\n",
                                   opt.z_draft_closure ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_axis=%s\n",
                                   opt.draft_axis));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_axes=%s\n",
                                   opt.draft_axes));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_eta=%.12e\n",
                                   static_cast<double>(opt.z_draft_eta)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_beta=%.12e\n",
                                   static_cast<double>(opt.draft_beta)));
  PetscCall(PetscViewerASCIIPrintf(viewer,
                                   "projected_volume_correction=%s\n",
                                   opt.projected_volume_correction ? "true"
                                                                   : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "use_mma=%s\n",
                                   opt.use_mma ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "matlab_z_projection=%s\n",
                                   opt.matlab_z_projection ? "true"
                                                           : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "z_draft_closure=%s\n",
                                   opt.z_draft_closure ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "z_draft_eta=%.12e\n",
                                   static_cast<double>(opt.z_draft_eta)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "benchmark_case=%s\n",
                                   opt.benchmark_case));
  PetscCall(PetscViewerASCIIPrintf(viewer, "h8_load_case=%lld\n",
                                   static_cast<long long>(load_case)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "h8_include_spring_load=%s\n",
                                   include_spring_load ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_iter=%lld\n",
                                   static_cast<long long>(final_iter)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_compliance=%.12e\n",
                                   static_cast<double>(compliance)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_volume=%.12e\n",
                                   static_cast<double>(volume)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_change=%.12e\n",
                                   static_cast<double>(change)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "total_wall_time_s=%.12e\n",
                                   static_cast<double>(total_time_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_current_memory_bytes=%.0f\n",
                                   static_cast<double>(mem_current_max)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_peak_memory_bytes=%.0f\n",
                                   static_cast<double>(mem_peak_max)));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "H8 optimization summary: %s\n", path));
  return 0;
}

PetscErrorCode write_vec_binary(Vec v, const char *path) {
  PetscViewer viewer = nullptr;
  PetscCall(PetscViewerBinaryOpen(PetscObjectComm(reinterpret_cast<PetscObject>(v)),
                                  path, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(v, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  return 0;
}

PetscErrorCode write_h8_checkpoint(const OptimizerOptions &opt, PetscInt iter,
                                   PetscBool final, Vec rho_design,
                                   Vec rho_phys, Vec mask) {
  char design_path[PETSC_MAX_PATH_LEN];
  char phys_path[PETSC_MAX_PATH_LEN];
  char mask_path[PETSC_MAX_PATH_LEN];
  const char *tag = final ? "final" : "iter";

  if (!opt.write_checkpoint) {
    return 0;
  }

  if (final) {
    PetscCall(PetscSNPrintf(design_path, sizeof(design_path),
                            "%s_final_rho_design.petscbin",
                            opt.checkpoint_prefix));
    PetscCall(PetscSNPrintf(phys_path, sizeof(phys_path),
                            "%s_final_rho_phys.petscbin",
                            opt.checkpoint_prefix));
    PetscCall(PetscSNPrintf(mask_path, sizeof(mask_path),
                            "%s_final_mask.petscbin",
                            opt.checkpoint_prefix));
  } else {
    PetscCall(PetscSNPrintf(design_path, sizeof(design_path),
                            "%s_iter_%06lld_rho_design.petscbin",
                            opt.checkpoint_prefix,
                            static_cast<long long>(iter)));
    PetscCall(PetscSNPrintf(phys_path, sizeof(phys_path),
                            "%s_iter_%06lld_rho_phys.petscbin",
                            opt.checkpoint_prefix,
                            static_cast<long long>(iter)));
    PetscCall(PetscSNPrintf(mask_path, sizeof(mask_path),
                            "%s_iter_%06lld_mask.petscbin",
                            opt.checkpoint_prefix,
                            static_cast<long long>(iter)));
  }

  PetscCall(write_vec_binary(rho_design, design_path));
  PetscCall(write_vec_binary(rho_phys, phys_path));
  if (final || iter == 0) {
    PetscCall(write_vec_binary(mask, mask_path));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Wrote H8 checkpoint (%s): %s, %s\n",
                        tag, design_path, phys_path));
  return 0;
}

} // namespace

PetscErrorCode run_h8_density_postprocess(const Grid &grid,
                                          const OptimizerOptions &optimizer_options,
                                          const char *density_file,
                                          const char *mask_file,
                                          const char *vtk_file,
                                          PetscInt stride,
                                          PetscInt max_samples) {
  DM uda = nullptr, eda = nullptr;
  Vec rho = nullptr, mask = nullptr;
  PetscBool have_mask_file = PETSC_FALSE;
  const PetscInt filter_stencil =
      PetscMax(1, static_cast<PetscInt>(PetscCeilReal(optimizer_options.filter_radius)));

  PetscCheck(density_file != nullptr && density_file[0] != '\0',
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "-density_file is required for -mode h8_postprocess");
  PetscCheck(vtk_file != nullptr && vtk_file[0] != '\0',
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "-post_vtk_file is required for -mode h8_postprocess");
  PetscCheck(stride >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "-post_stride must be at least 1");

  PetscCall(create_h8_dms(grid, filter_stencil, &uda, &eda));
  PetscCall(DMCreateGlobalVector(eda, &rho));
  PetscCall(VecDuplicate(rho, &mask));

  PetscCall(load_dmda_natural_binary_checkpoint(eda, rho, density_file));

  if (mask_file != nullptr && mask_file[0] != '\0') {
    have_mask_file = PETSC_TRUE;
    PetscCall(load_dmda_natural_binary_checkpoint(eda, mask, mask_file));
  } else {
    PetscCall(VecSet(mask, 1.0));
  }
  if (have_mask_file) {
    PetscCall(apply_h8_draft_closure(eda, mask, optimizer_options, rho));
  }

  PetscCall(write_downsampled_h8_density_vtk(eda, vtk_file, grid, rho, mask,
                                             stride, max_samples));

  PetscCall(VecDestroy(&mask));
  PetscCall(VecDestroy(&rho));
  PetscCall(DMDestroy(&eda));
  PetscCall(DMDestroy(&uda));
  return 0;
}

PetscErrorCode run_h8_initial_vtk(const Grid &grid,
                                  const DensityOptions &density_options,
                                  const OptimizerOptions &optimizer_options,
                                  const char *vtk_file,
                                  PetscInt stride,
                                  PetscInt max_samples) {
  DM uda = nullptr, eda = nullptr;
  Vec mask = nullptr, rho_design = nullptr, rho_filtered = nullptr;
  Vec rho_phys = nullptr;
  Vec filter_denom = nullptr;
  const PetscInt filter_stencil =
      PetscMax(1, static_cast<PetscInt>(PetscCeilReal(optimizer_options.filter_radius)));

  PetscCheck(vtk_file != nullptr && vtk_file[0] != '\0',
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "-post_vtk_file is required for -mode h8_initial_vtk");
  PetscCheck(stride >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "-post_stride must be at least 1");

  PetscCall(create_h8_dms(grid, filter_stencil, &uda, &eda));
  PetscCall(create_element_mask_and_density(eda, grid, density_options,
                                            optimizer_options, &mask,
                                            &rho_design));
  PetscCall(VecDuplicate(rho_design, &rho_filtered));
  PetscCall(VecDuplicate(rho_design, &rho_phys));
  PetscCall(VecDuplicate(rho_design, &filter_denom));
  PetscCall(compute_filter_denominator(eda, mask,
                                       optimizer_options.filter_radius,
                                       filter_denom));
  PetscBool matlab_z_projection = PETSC_FALSE;
  PetscCall(h8_uses_matlab_z_projection(optimizer_options,
                                        &matlab_z_projection));
  const OptimizerOptions projection_options =
      h8_projection_options_for_iter(optimizer_options, matlab_z_projection, 1);
  const PetscReal projection_beta =
      matlab_z_projection
          ? h8_matlab_outer_beta_for_iter(optimizer_options, 1)
          : h8_heaviside_beta_for_iter(optimizer_options, 1);
  PetscCall(build_h8_physical_density(
      eda, rho_design, mask, filter_denom, density_options, projection_options,
      projection_beta, rho_filtered, rho_phys));
  PetscCall(write_downsampled_h8_density_vtk(eda, vtk_file, grid, rho_phys,
                                             mask, stride, max_samples));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Wrote H8 initial design/mask VTK: %s\n", vtk_file));

  PetscCall(VecDestroy(&filter_denom));
  PetscCall(VecDestroy(&rho_phys));
  PetscCall(VecDestroy(&rho_filtered));
  PetscCall(VecDestroy(&rho_design));
  PetscCall(VecDestroy(&mask));
  PetscCall(DMDestroy(&eda));
  PetscCall(DMDestroy(&uda));
  return 0;
}

PetscErrorCode run_h8_full_vtk_postprocess(const Grid &grid,
                                           const DensityOptions &density_options,
                                           const OptimizerOptions &optimizer_options,
                                           const char *density_file,
                                           const char *mask_file,
                                           const char *vtk_file) {
  DM uda = nullptr, eda = nullptr;
  Vec rho = nullptr, mask = nullptr, u = nullptr, b = nullptr;
  PetscBool have_mask_file = PETSC_FALSE;
  Mat A = nullptr;
  Mat P = nullptr;
  KSP ksp = nullptr;
  PetscInt its = 0;
  PetscReal rnorm = 0.0;
  PetscLogDouble total_start = 0.0, solve_start = 0.0, write_start = 0.0;
  PetscReal solve_time = 0.0, write_time = 0.0, total_time = 0.0;
  char h8_pc_type[64];
  PetscBool use_aux_matrix = PETSC_FALSE;
  PetscBool use_elastic_aux_matrix = PETSC_FALSE;
  const PetscInt filter_stencil =
      PetscMax(1, static_cast<PetscInt>(PetscCeilReal(optimizer_options.filter_radius)));

  PetscCheck(density_file != nullptr && density_file[0] != '\0',
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "-density_file is required for -mode h8_full_vtk");
  PetscCheck(vtk_file != nullptr && vtk_file[0] != '\0',
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "-post_vtk_file is required for -mode h8_full_vtk");

  PetscCall(PetscTime(&total_start));
  PetscCall(create_h8_dms(grid, filter_stencil, &uda, &eda));
  PetscCall(DMCreateGlobalVector(eda, &rho));
  PetscCall(VecDuplicate(rho, &mask));

  PetscCall(load_dmda_natural_binary_checkpoint(eda, rho, density_file));

  if (mask_file != nullptr && mask_file[0] != '\0') {
    have_mask_file = PETSC_TRUE;
    PetscCall(load_dmda_natural_binary_checkpoint(eda, mask, mask_file));
  } else {
    PetscCall(VecSet(mask, 1.0));
  }
  if (have_mask_file) {
    PetscCall(apply_h8_draft_closure(eda, mask, optimizer_options, rho));
  }

  PetscCall(DMCreateGlobalVector(uda, &u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(fill_h8_load(uda, eda, mask, grid, density_options,
                         optimizer_options, b));
  PetscCall(create_h8_shell_matrix(uda, eda, rho, grid, density_options, &A));
  PetscCall(get_h8_pc_type_option(h8_pc_type, sizeof(h8_pc_type),
                                  &use_aux_matrix));
  use_elastic_aux_matrix = h8_pc_type_uses_elastic_aux_matrix(h8_pc_type);
  if (use_aux_matrix) {
    PetscCall(log_h8_memory_stage("full_vtk before auxiliary matrix"));
    if (use_elastic_aux_matrix) {
      PetscCall(create_h8_aux_elasticity_matrix(uda, eda, rho, grid,
                                                density_options, &P));
    } else {
      PetscCall(create_h8_aux_laplacian_matrix(uda, eda, rho, grid,
                                               density_options, &P));
    }
    PetscCall(log_h8_memory_stage("full_vtk after auxiliary matrix"));
  }
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(configure_h8_ksp(ksp, A, P, uda, eda, rho, grid, density_options,
                             optimizer_options));
  PetscCall(log_h8_memory_stage("full_vtk after KSP configure"));

  PetscCall(PetscTime(&solve_start));
  PetscCall(VecSet(u, 0.0));
  PetscCall(set_h8_ksp_operators(ksp, A, P, uda, eda, rho, grid,
                                 density_options, PETSC_FALSE,
                                 use_elastic_aux_matrix));
  PetscCall(log_h8_memory_stage("full_vtk before KSPSolve"));
  PetscCall(KSPSolve(ksp, b, u));
  PetscCall(log_h8_memory_stage("full_vtk after KSPSolve"));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCall(KSPGetResidualNorm(ksp, &rnorm));
  PetscCall(elapsed_max(solve_start, &solve_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Full VTK displacement solve finished: ksp_it=%lld ksp_reason=%d residual=%.3e linear_solve_s=%.6g\n",
                        static_cast<long long>(its),
                        static_cast<int>(reason),
                        static_cast<double>(rnorm),
                        static_cast<double>(solve_time)));

  PetscCall(PetscTime(&write_start));
  PetscCall(write_full_h8_parallel_pvti(uda, eda, vtk_file, grid, u, rho, mask));
  PetscCall(elapsed_max(write_start, &write_time));
  PetscCall(elapsed_max(total_start, &total_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Full VTK postprocess timings: linear_solve_s=%.6g write_s=%.6g total_s=%.6g\n",
                        static_cast<double>(solve_time),
                        static_cast<double>(write_time),
                        static_cast<double>(total_time)));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&mask));
  PetscCall(VecDestroy(&rho));
  PetscCall(DMDestroy(&eda));
  PetscCall(DMDestroy(&uda));
  return 0;
}

PetscErrorCode run_h8_optimizer(const Grid &grid,
                                const DensityOptions &density_options,
                                const OptimizerOptions &optimizer_options,
                                const char *output_prefix,
  const char *final_vtk_file) {
  DM uda = nullptr, eda = nullptr;
  Vec rho_design = nullptr, rho_filtered = nullptr, rho_phys = nullptr;
  Vec mask = nullptr, update_mask = nullptr;
  Vec u = nullptr, b = nullptr;
  Vec dc_phys = nullptr, dc_filtered = nullptr, dc_design = nullptr;
  Vec dv_design = nullptr;
  Vec filter_denom = nullptr;
  Mat A = nullptr;
  Mat P = nullptr;
  KSP ksp = nullptr;
  PetscViewer hist = nullptr;
  PetscViewer mma_trace = nullptr;
  PetscViewer mma_vector_trace = nullptr;
  char hist_path[PETSC_MAX_PATH_LEN];
  char mma_trace_path[PETSC_MAX_PATH_LEN];
  char mma_vector_trace_path[PETSC_MAX_PATH_LEN];
  char h8_pc_type[64];
  std::vector<ObjectiveVolumePoint> objective_history;
  PetscReal ke[24 * 24]{};
  PetscReal compliance = 0.0, volume = 0.0, change = 0.0;
  PetscReal rhs_norm = 0.0;
  PetscLogDouble total_start = 0.0;
  PetscInt final_iter = 0;
  PetscBool reuse_initial_guess = PETSC_TRUE;
  PetscBool stop_on_ksp_divergence = PETSC_TRUE;
  PetscBool stopped_on_ksp_divergence = PETSC_FALSE;
  PetscBool use_aux_matrix = PETSC_FALSE;
  PetscBool use_elastic_aux_matrix = PETSC_FALSE;
  PetscBool matlab_z_projection = PETSC_FALSE;
  PetscBool write_mma_vectors = PETSC_FALSE;
  H8MatlabMMAState matlab_mma_state;
  H8MatlabMMAState general_mma_state;
  PetscInt aux_rebuild_interval = 1;
  PetscInt h8_load_case = 0;
  PetscBool h8_include_spring_load = PETSC_TRUE;
  const PetscInt filter_stencil =
      PetscMax(1, static_cast<PetscInt>(PetscCeilReal(optimizer_options.filter_radius)));

  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-h8_reuse_initial_guess",
                                &reuse_initial_guess, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_stop_on_ksp_divergence",
                                &stop_on_ksp_divergence, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-h8_aux_rebuild_interval",
                               &aux_rebuild_interval, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-opt_write_mma_vectors",
                                &write_mma_vectors, nullptr));
  PetscCall(h8_get_load_options(&h8_load_case, &h8_include_spring_load));
  PetscCall(h8_uses_matlab_z_projection(optimizer_options,
                                        &matlab_z_projection));
  const PetscBool include_fixed_solid_volume =
      (!density_options.use_control_arm_mask &&
       benchmark_has_matlab_load_elements(optimizer_options.benchmark_case))
          ? PETSC_TRUE
          : PETSC_FALSE;
  aux_rebuild_interval = PetscMax(1, aux_rebuild_interval);
  PetscCall(PetscTime(&total_start));
  compute_h8_element_stiffness(grid, density_options, ke);
  PetscCall(create_h8_dms(grid, filter_stencil, &uda, &eda));
  PetscCall(create_element_mask_and_density(eda, grid, density_options,
                                            optimizer_options, &mask, &rho_design));
  PetscCall(VecDuplicate(mask, &update_mask));
  if (matlab_z_projection) {
    PetscCall(create_h8_z_surface_update_mask(eda, mask, update_mask));
  } else {
    PetscCall(VecCopy(mask, update_mask));
  }
  PetscCall(VecDuplicate(rho_design, &rho_filtered));
  PetscCall(VecDuplicate(rho_design, &rho_phys));
  PetscCall(VecDuplicate(rho_design, &filter_denom));
  PetscReal current_beta =
      matlab_z_projection
          ? h8_matlab_outer_beta_for_iter(optimizer_options, 1)
          : h8_heaviside_beta_for_iter(optimizer_options, 1);
  OptimizerOptions projection_options =
      h8_projection_options_for_iter(optimizer_options, matlab_z_projection, 1);
  PetscCall(compute_filter_denominator(eda, mask, optimizer_options.filter_radius,
                                       filter_denom));
  PetscCall(build_h8_physical_density(eda, rho_design, mask, filter_denom,
                                      density_options, projection_options,
                                      current_beta, rho_filtered, rho_phys));
  PetscCall(VecDuplicate(rho_design, &dv_design));
  if (matlab_z_projection) {
    PetscCall(VecSet(dv_design, 0.0));
  } else {
    PetscCall(apply_sensitivity_filter_adjoint(eda, mask, mask, filter_denom,
                                               optimizer_options.filter_radius,
                                               dv_design));
  }
  PetscCall(DMCreateGlobalVector(uda, &u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(DMCreateGlobalVector(eda, &dc_phys));
  PetscCall(VecDuplicate(dc_phys, &dc_filtered));
  PetscCall(VecDuplicate(dc_phys, &dc_design));
  PetscCall(fill_h8_load(uda, eda, mask, grid, density_options,
                         optimizer_options, b));
  PetscCall(VecNorm(b, NORM_2, &rhs_norm));
  PetscCall(create_h8_shell_matrix(uda, eda, rho_phys, grid, density_options, &A));
  PetscCall(get_h8_pc_type_option(h8_pc_type, sizeof(h8_pc_type),
                                  &use_aux_matrix));
  use_elastic_aux_matrix = h8_pc_type_uses_elastic_aux_matrix(h8_pc_type);
  if (use_aux_matrix) {
    PetscCall(log_h8_memory_stage("optimizer before auxiliary matrix"));
    if (use_elastic_aux_matrix) {
      PetscCall(create_h8_aux_elasticity_matrix(uda, eda, rho_phys, grid,
                                                density_options, &P));
    } else {
      PetscCall(create_h8_aux_laplacian_matrix(uda, eda, rho_phys, grid,
                                               density_options, &P));
    }
    PetscCall(log_h8_memory_stage("optimizer after auxiliary matrix"));
  }
  PetscCall(write_h8_checkpoint(optimizer_options, 0, PETSC_FALSE,
                                rho_design, rho_phys, mask));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(configure_h8_ksp(ksp, A, P, uda, eda, rho_phys, grid, density_options,
                             optimizer_options));
  PetscCall(log_h8_memory_stage("optimizer after KSP configure"));

  PetscCall(PetscSNPrintf(hist_path, sizeof(hist_path), "%s_history.csv",
                          output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, hist_path, &hist));
  PetscCall(PetscViewerASCIIPrintf(hist,
                                   "iter,compliance,volume,raw_design_volume,volume_target_for_update,projected_volume_gap,heaviside_beta,change,ksp_iterations,ksp_reason,ksp_reason_name,ksp_residual,rhs_norm,ksp_relative_residual,filter_s,linear_solve_s,sensitivity_s,update_s,checkpoint_s,iter_total_s\n"));
  if (optimizer_options.use_mma) {
    PetscCall(PetscSNPrintf(mma_trace_path, sizeof(mma_trace_path),
                            "%s_mma_trace.csv", output_prefix));
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, mma_trace_path,
                                   &mma_trace));
    PetscCall(PetscViewerASCIIPrintf(
        mma_trace,
        "iter,f0val,fval,volume,beta1,beta2,move,c_scale,x_change,x_min,x_max,x_sum,x_l1,xmma_min,xmma_max,xmma_sum,xmma_l1,df0dx_min,df0dx_max,df0dx_sum,df0dx_l1,dfdx_min,dfdx_max,dfdx_sum,dfdx_l1\n"));
    if (write_mma_vectors) {
      PetscCall(PetscSNPrintf(mma_vector_trace_path,
                              sizeof(mma_vector_trace_path),
                              "%s_mma_vectors.csv", output_prefix));
      PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, mma_vector_trace_path,
                                     &mma_vector_trace));
      PetscCall(PetscViewerASCIIPrintf(
          mma_vector_trace, "iter,var,x,xmma,df0dx,dfdx\n"));
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Optimize mode: h8_matrix_free nx=%lld ny=%lld nz=%lld size=%.4gm x %.4gm x %.4gm max_iter=%lld volfrac=%g filter_radius=%g\n",
                        static_cast<long long>(grid.nx),
                        static_cast<long long>(grid.ny),
                        static_cast<long long>(grid.nz),
                        static_cast<double>(domain_length(grid)),
                        static_cast<double>(domain_width(grid)),
                        static_cast<double>(domain_height(grid)),
                        static_cast<long long>(optimizer_options.max_iter),
                        static_cast<double>(optimizer_options.volfrac),
                        static_cast<double>(optimizer_options.filter_radius)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 KSP reuse initial guess: %s\n",
                        reuse_initial_guess ? "true" : "false"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 stop on KSP divergence: %s\n",
                        stop_on_ksp_divergence ? "true" : "false"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 Heaviside projection: %s beta0=%g beta_max=%g beta_interval=%lld\n",
                        optimizer_options.heaviside_projection ? "true"
                                                               : "false",
                        static_cast<double>(
                            optimizer_options.heaviside_beta_initial),
                        static_cast<double>(
                            optimizer_options.heaviside_beta_max),
                        static_cast<long long>(
                            optimizer_options.heaviside_beta_interval)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 optimizer update: %s%s\n",
                        (optimizer_options.use_mma && matlab_z_projection)
                            ? "MATLAB-style MMA"
                            : (optimizer_options.use_mma ? "MMA" : "OC"),
                        (optimizer_options.use_mma && !matlab_z_projection)
                            ? " (general H8 design variables)"
                            : ""));
  if (density_options.use_control_arm_mask) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 hard regions/BC: A/B/C rings and spring mount are locked; C ring is fixed; A/B rings plus spring mount are loaded; draft_closure=%s axes=%s eta=%g\n",
                          optimizer_options.z_draft_closure ? "true" : "false",
                          optimizer_options.draft_axes,
                          static_cast<double>(optimizer_options.z_draft_eta)));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 load options: load_case=%lld (%s), include_spring_load=%s\n",
                          static_cast<long long>(h8_load_case),
                          h8_load_case == 4
                              ? "symmetric vertical A/B/spring case"
                              : ((h8_load_case >= 1 && h8_load_case <= 3)
                                     ? "single legacy case"
                                     : "weighted legacy cases 1-3"),
                          h8_include_spring_load ? "true" : "false"));
  } else {
    if (benchmark_is_bridge(optimizer_options.benchmark_case)) {
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "H8 rectangular benchmark: case=bridge, bottom left/right support lines fixed, top face loaded; draft_closure=%s axes=%s eta=%g\n",
          optimizer_options.z_draft_closure ? "true" : "false",
          optimizer_options.draft_axes,
          static_cast<double>(optimizer_options.z_draft_eta)));
    } else if (benchmark_is_mbb(optimizer_options.benchmark_case)) {
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "H8 rectangular benchmark: case=mbb, left face fixed, right-bottom line loaded; draft_closure=%s axes=%s eta=%g\n",
          optimizer_options.z_draft_closure ? "true" : "false",
          optimizer_options.draft_axes,
          static_cast<double>(optimizer_options.z_draft_eta)));
    } else if (benchmark_is_tri(optimizer_options.benchmark_case)) {
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "H8 rectangular benchmark: case=tri, fixed nodes match MATLAB [1,nely+1,(nely+1)*(nelx+1)], z-top y-high x-left corner loaded in -x/-y; draft_closure=%s axes=%s eta=%g\n",
          optimizer_options.z_draft_closure ? "true" : "false",
          optimizer_options.draft_axes,
          static_cast<double>(optimizer_options.z_draft_eta)));
    } else if (benchmark_is_torsion_edge(optimizer_options.benchmark_case)) {
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "H8 rectangular benchmark: case=%s, left face fixed, right-end edge ring loaded tangentially for torque Mx=opt_load; draft_closure=%s axes=%s eta=%g\n",
          optimizer_options.benchmark_case,
          optimizer_options.z_draft_closure ? "true" : "false",
          optimizer_options.draft_axes,
          static_cast<double>(optimizer_options.z_draft_eta)));
    } else if (benchmark_is_torsion(optimizer_options.benchmark_case)) {
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "H8 rectangular benchmark: case=%s, left face fixed, right face loaded tangentially for torque Mx=opt_load; draft_closure=%s axes=%s eta=%g\n",
          optimizer_options.benchmark_case,
          optimizer_options.z_draft_closure ? "true" : "false",
          optimizer_options.draft_axes,
          static_cast<double>(optimizer_options.z_draft_eta)));
    } else if (benchmark_is_tip_center_cantilever(
                   optimizer_options.benchmark_case)) {
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "H8 rectangular benchmark: case=%s, left face fixed, right-end face center point loaded in -Z; draft_closure=%s axes=%s eta=%g\n",
          optimizer_options.benchmark_case,
          optimizer_options.z_draft_closure ? "true" : "false",
          optimizer_options.draft_axes,
          static_cast<double>(optimizer_options.z_draft_eta)));
    } else if (benchmark_is_bottom_point_cantilever(
                   optimizer_options.benchmark_case)) {
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "H8 rectangular benchmark: case=%s, left face fixed, right-end bottom-center point loaded in -Z; draft_closure=%s axes=%s eta=%g\n",
          optimizer_options.benchmark_case,
          optimizer_options.z_draft_closure ? "true" : "false",
          optimizer_options.draft_axes,
          static_cast<double>(optimizer_options.z_draft_eta)));
    } else {
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "H8 rectangular benchmark: case=%s, left face fixed, right face loaded; draft_closure=%s axes=%s eta=%g\n",
          optimizer_options.benchmark_case,
          optimizer_options.z_draft_closure ? "true" : "false",
          optimizer_options.draft_axes,
          static_cast<double>(optimizer_options.z_draft_eta)));
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 RHS norm ||b||_2=%.6e. History logs both absolute KSP residual and residual/||b||.\n",
                        static_cast<double>(rhs_norm)));
  if (use_aux_matrix) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "H8 auxiliary preconditioner rebuild interval: %lld iteration(s)\n",
                          static_cast<long long>(aux_rebuild_interval)));
  }

  for (PetscInt iter = 1; iter <= optimizer_options.max_iter; ++iter) {
    PetscInt its = 0;
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscReal rnorm = 0.0;
    PetscReal rel_rnorm = 0.0;
    PetscReal filter_time = 0.0;
    PetscReal solve_time = 0.0;
    PetscReal sensitivity_time = 0.0;
    PetscReal update_time = 0.0;
    PetscReal checkpoint_time = 0.0;
    PetscReal iter_time = 0.0;
    PetscScalar dot = 0.0;
    PetscLogDouble iter_start = 0.0;
    PetscLogDouble stage_start = 0.0;
    char memory_stage[128];

    PetscCall(PetscTime(&iter_start));
    current_beta =
        matlab_z_projection
            ? h8_matlab_outer_beta_for_iter(optimizer_options, iter)
            : h8_heaviside_beta_for_iter(optimizer_options, iter);
    projection_options =
        h8_projection_options_for_iter(optimizer_options,
                                       matlab_z_projection, iter);
    PetscCall(PetscTime(&stage_start));
    PetscCall(build_h8_physical_density(eda, rho_design, mask, filter_denom,
                                        density_options, projection_options,
                                        current_beta, rho_filtered, rho_phys));
    PetscCall(elapsed_max(stage_start, &filter_time));

    PetscCall(PetscTime(&stage_start));
    if (iter == 1 || !reuse_initial_guess) {
      PetscCall(VecSet(u, 0.0));
      PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
    } else {
      PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    }
    {
      const PetscBool rebuild_aux =
          (use_aux_matrix && iter > 1 &&
           ((iter - 1) % aux_rebuild_interval == 0)) ? PETSC_TRUE
                                                     : PETSC_FALSE;
      PetscCall(set_h8_ksp_operators(ksp, A, P, uda, eda, rho_phys, grid,
                                     density_options, rebuild_aux,
                                     use_elastic_aux_matrix));
    }
    PetscCall(PetscSNPrintf(memory_stage, sizeof(memory_stage),
                            "optimizer iter %lld before KSPSolve",
                            static_cast<long long>(iter)));
    PetscCall(log_h8_memory_stage(memory_stage));
    PetscCall(KSPSolve(ksp, b, u));
    PetscCall(PetscSNPrintf(memory_stage, sizeof(memory_stage),
                            "optimizer iter %lld after KSPSolve",
                            static_cast<long long>(iter)));
    PetscCall(log_h8_memory_stage(memory_stage));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    rel_rnorm = rhs_norm > 0.0 ? rnorm / rhs_norm : rnorm;
    PetscCall(elapsed_max(stage_start, &solve_time));

    PetscCall(VecDot(b, u, &dot));
    compliance = PetscRealPart(dot);
    PetscReal raw_design_volume = 0.0;
    PetscReal projected_volume_gap = 0.0;
    PetscReal effective_raw_volfrac = optimizer_options.volfrac;
    const PetscBool ksp_failed = reason < 0 ? PETSC_TRUE : PETSC_FALSE;

    if (ksp_failed) {
      change = 0.0;
      PetscCall(compute_masked_volume_fraction_with_fixed(
          eda, rho_phys, mask, include_fixed_solid_volume, &volume));
      PetscCall(compute_masked_volume_fraction(eda, rho_design, update_mask,
                                               &raw_design_volume));
      projected_volume_gap = volume - raw_design_volume;
      PetscCall(elapsed_max(iter_start, &iter_time));
      PetscCall(PetscViewerASCIIPrintf(hist, "%lld,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%lld,%d,%s,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                                       static_cast<long long>(iter),
                                       static_cast<double>(compliance),
                                       static_cast<double>(volume),
                                       static_cast<double>(raw_design_volume),
                                       static_cast<double>(effective_raw_volfrac),
                                       static_cast<double>(projected_volume_gap),
                                       static_cast<double>(current_beta),
                                       static_cast<double>(change),
                                       static_cast<long long>(its),
                                       static_cast<int>(reason),
                                       ksp_reason_name(reason),
                                       static_cast<double>(rnorm),
                                       static_cast<double>(rhs_norm),
                                       static_cast<double>(rel_rnorm),
                                       static_cast<double>(filter_time),
                                       static_cast<double>(solve_time),
                                       0.0, 0.0, 0.0,
                                       static_cast<double>(iter_time)));
      PetscCall(PetscViewerFlush(hist));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "it=%lld compliance=%.6e volume=%.6f beta=%.3g change=%.3e ksp_it=%lld ksp_reason=%d(%s) residual=%.3e rel_res=%.3e time[s]: filter=%.3g linear_solve=%.3g total=%.3g\n",
                            static_cast<long long>(iter),
                            static_cast<double>(compliance),
                            static_cast<double>(volume),
                            static_cast<double>(current_beta),
                            static_cast<double>(change),
                            static_cast<long long>(its),
                            static_cast<int>(reason),
                            ksp_reason_name(reason),
                            static_cast<double>(rnorm),
                            static_cast<double>(rel_rnorm),
                            static_cast<double>(filter_time),
                            static_cast<double>(solve_time),
                            static_cast<double>(iter_time)));
      if (stop_on_ksp_divergence) {
        stopped_on_ksp_divergence = PETSC_TRUE;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "Stopping H8 optimization because the linear solve did not converge; no OC update was applied for this step.\n"));
        break;
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "Skipping this H8 OC update because the linear solve did not converge.\n"));
      continue;
    }

    PetscCall(PetscTime(&stage_start));
    PetscCall(compute_h8_sensitivity(uda, eda, u, rho_phys, mask, grid,
                                     density_options, ke,
                                     include_fixed_solid_volume, dc_phys,
                                     &volume));
    if (matlab_z_projection) {
      PetscCall(apply_h8_matlab_z_surface_sensitivity(
          eda, dc_phys, mask, rho_design, density_options, projection_options,
          current_beta, dc_design, dv_design));
    } else {
      PetscCall(apply_h8_heaviside_sensitivity(
          eda, dc_phys, rho_filtered, mask, optimizer_options, current_beta,
          dc_filtered));
      PetscCall(apply_sensitivity_filter_adjoint(
          eda, dc_filtered, mask, filter_denom, optimizer_options.filter_radius,
          dc_design));
    }
    PetscCall(elapsed_max(stage_start, &sensitivity_time));

    PetscCall(PetscTime(&stage_start));
    PetscCall(compute_masked_volume_fraction(eda, rho_design, update_mask,
                                             &raw_design_volume));
    projected_volume_gap = volume - raw_design_volume;
    OptimizerOptions step_options = projection_options;
    H8MatlabMMADiagnostics mma_diag;
    if (optimizer_options.projected_volume_correction || matlab_z_projection) {
      // 直接用“滤波 -> Heaviside -> 拔模闭包”后的物理体积选择 OC
      // lambda，避免把闭包体积差线性折算成 raw 目标后产生来回跳变。
      // Match the MATLAB schedule: target = max(0.9 * current physical
      // volume, final volfrac), so the early iterations do not collapse.
      effective_raw_volfrac =
          PetscMax(optimizer_options.volfrac, 0.9 * volume);
      step_options.volfrac = effective_raw_volfrac;
      if (optimizer_options.use_mma && matlab_z_projection) {
        PetscCall(h8_mma_update_matlab_z(
            eda, rho_design, mask, dc_phys, density_options, step_options,
            current_beta, iter, compliance, volume, &matlab_mma_state,
            &mma_diag, &change));
      } else if (optimizer_options.use_mma) {
        mma_diag.beta2 = current_beta;
        PetscCall(h8_mma_update_general(
            eda, rho_design, update_mask, dc_design, dv_design, step_options,
            current_beta, iter, compliance, volume, &general_mma_state,
            &mma_diag, &change));
      } else {
        PetscCall(oc_update_projected_volume(
            eda, rho_design, mask, update_mask, dc_design, dv_design,
            filter_denom, current_beta, density_options, step_options,
            include_fixed_solid_volume, &change));
      }
    } else {
      step_options.volfrac = effective_raw_volfrac;
      PetscCall(oc_update(eda, rho_design, mask, dc_design, dv_design,
                          density_options, step_options, &change));
    }
    PetscCall(elapsed_max(stage_start, &update_time));
    if (mma_trace != nullptr) {
      PetscCall(PetscViewerASCIIPrintf(
          mma_trace,
          "%lld,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
          static_cast<long long>(iter),
          static_cast<double>(mma_diag.f0val),
          static_cast<double>(mma_diag.fval),
          static_cast<double>(volume),
          static_cast<double>(mma_diag.beta1),
          static_cast<double>(mma_diag.beta2),
          static_cast<double>(mma_diag.move),
          static_cast<double>(mma_diag.c_scale),
          static_cast<double>(mma_diag.x_change),
          static_cast<double>(mma_diag.x.min),
          static_cast<double>(mma_diag.x.max),
          static_cast<double>(mma_diag.x.sum),
          static_cast<double>(mma_diag.x.l1),
          static_cast<double>(mma_diag.xmma.min),
          static_cast<double>(mma_diag.xmma.max),
          static_cast<double>(mma_diag.xmma.sum),
          static_cast<double>(mma_diag.xmma.l1),
          static_cast<double>(mma_diag.df0dx.min),
          static_cast<double>(mma_diag.df0dx.max),
          static_cast<double>(mma_diag.df0dx.sum),
          static_cast<double>(mma_diag.df0dx.l1),
          static_cast<double>(mma_diag.dfdx.min),
          static_cast<double>(mma_diag.dfdx.max),
          static_cast<double>(mma_diag.dfdx.sum),
          static_cast<double>(mma_diag.dfdx.l1)));
      PetscCall(PetscViewerFlush(mma_trace));
    }
    if (mma_vector_trace != nullptr) {
      const std::size_t nvars = std::min(
          std::min(mma_diag.x_values.size(), mma_diag.xmma_values.size()),
          std::min(mma_diag.df0dx_values.size(), mma_diag.dfdx_values.size()));
      for (std::size_t q = 0; q < nvars; ++q) {
        PetscCall(PetscViewerASCIIPrintf(
            mma_vector_trace,
            "%lld,%lld,%.16e,%.16e,%.16e,%.16e\n",
            static_cast<long long>(iter), static_cast<long long>(q + 1),
            static_cast<double>(mma_diag.x_values[q]),
            static_cast<double>(mma_diag.xmma_values[q]),
            static_cast<double>(mma_diag.df0dx_values[q]),
            static_cast<double>(mma_diag.dfdx_values[q])));
      }
      PetscCall(PetscViewerFlush(mma_vector_trace));
    }

    PetscCall(PetscTime(&stage_start));
    if (optimizer_options.write_checkpoint &&
        optimizer_options.checkpoint_interval > 0 &&
        iter % optimizer_options.checkpoint_interval == 0) {
      PetscCall(build_h8_physical_density(eda, rho_design, mask, filter_denom,
                                          density_options, projection_options,
                                          current_beta, rho_filtered,
                                          rho_phys));
      PetscCall(write_h8_checkpoint(optimizer_options, iter, PETSC_FALSE,
                                    rho_design, rho_phys, mask));
    }
    PetscCall(elapsed_max(stage_start, &checkpoint_time));
    PetscCall(elapsed_max(iter_start, &iter_time));
    final_iter = iter;
    PetscCall(PetscViewerASCIIPrintf(hist, "%lld,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%lld,%d,%s,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                                     static_cast<long long>(iter),
                                     static_cast<double>(compliance),
                                     static_cast<double>(volume),
                                     static_cast<double>(raw_design_volume),
                                     static_cast<double>(effective_raw_volfrac),
                                     static_cast<double>(projected_volume_gap),
                                     static_cast<double>(current_beta),
                                     static_cast<double>(change),
                                     static_cast<long long>(its),
                                     static_cast<int>(reason),
                                     ksp_reason_name(reason),
                                     static_cast<double>(rnorm),
                                     static_cast<double>(rhs_norm),
                                     static_cast<double>(rel_rnorm),
                                     static_cast<double>(filter_time),
                                     static_cast<double>(solve_time),
                                     static_cast<double>(sensitivity_time),
                                     static_cast<double>(update_time),
                                     static_cast<double>(checkpoint_time),
                                     static_cast<double>(iter_time)));
    PetscCall(PetscViewerFlush(hist));
    objective_history.push_back({iter, compliance, volume});
    PetscCall(write_objective_volume_history(output_prefix, objective_history,
                                             optimizer_options.volfrac));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "it=%lld compliance=%.6e volume=%.6f raw_volume=%.6f volume_target=%.6f projected_gap=%.3e beta=%.3g change=%.3e ksp_it=%lld ksp_reason=%d(%s) residual=%.3e rel_res=%.3e time[s]: filter=%.3g linear_solve=%.3g sens=%.3g update=%.3g checkpoint=%.3g total=%.3g\n",
                          static_cast<long long>(iter),
                          static_cast<double>(compliance),
                          static_cast<double>(volume),
                          static_cast<double>(raw_design_volume),
                          static_cast<double>(effective_raw_volfrac),
                          static_cast<double>(projected_volume_gap),
                          static_cast<double>(current_beta),
                          static_cast<double>(change),
                          static_cast<long long>(its),
                          static_cast<int>(reason),
                          ksp_reason_name(reason),
                          static_cast<double>(rnorm),
                          static_cast<double>(rel_rnorm),
                          static_cast<double>(filter_time),
                          static_cast<double>(solve_time),
                          static_cast<double>(sensitivity_time),
                          static_cast<double>(update_time),
                          static_cast<double>(checkpoint_time),
                          static_cast<double>(iter_time)));
  }
  PetscCall(PetscViewerDestroy(&mma_vector_trace));
  PetscCall(PetscViewerDestroy(&mma_trace));
  PetscCall(PetscViewerDestroy(&hist));

  if (!stopped_on_ksp_divergence) {
    PetscScalar dot = 0.0;
    PetscInt its = 0;
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscReal rnorm = 0.0;
    PetscReal rel_rnorm = 0.0;
    current_beta =
        matlab_z_projection
            ? h8_matlab_outer_beta_for_iter(optimizer_options,
                                            PetscMax(1, final_iter))
            : h8_heaviside_beta_for_iter(optimizer_options,
                                         PetscMax(1, final_iter));
    projection_options =
        h8_projection_options_for_iter(optimizer_options,
                                       matlab_z_projection,
                                       PetscMax(1, final_iter));
    PetscCall(build_h8_physical_density(eda, rho_design, mask, filter_denom,
                                        density_options, projection_options,
                                        current_beta, rho_filtered, rho_phys));
    PetscCall(KSPSetInitialGuessNonzero(ksp, reuse_initial_guess));
    if (!reuse_initial_guess) PetscCall(VecSet(u, 0.0));
    PetscCall(set_h8_ksp_operators(ksp, A, P, uda, eda, rho_phys, grid,
                                   density_options, use_aux_matrix,
                                   use_elastic_aux_matrix));
    PetscCall(KSPSolve(ksp, b, u));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    rel_rnorm = rhs_norm > 0.0 ? rnorm / rhs_norm : rnorm;
    PetscCall(VecDot(b, u, &dot));
    compliance = PetscRealPart(dot);
    PetscCall(compute_h8_sensitivity(uda, eda, u, rho_phys, mask, grid,
                                     density_options, ke,
                                     include_fixed_solid_volume, dc_phys,
                                     &volume));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "final compliance=%.6e volume=%.6f beta=%.3g ksp_it=%lld ksp_reason=%d(%s) residual=%.3e rel_res=%.3e\n",
                          static_cast<double>(compliance),
                          static_cast<double>(volume),
                          static_cast<double>(current_beta),
                          static_cast<long long>(its),
                          static_cast<int>(reason),
                          ksp_reason_name(reason),
                          static_cast<double>(rnorm),
                          static_cast<double>(rel_rnorm)));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Final verification solve skipped because optimization stopped on KSP divergence.\n"));
  }

  if (optimizer_options.write_final_vtk && !stopped_on_ksp_divergence) {
    PetscCheck(node_count(grid) <= optimizer_options.vtk_max_points,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "H8 optimization VTK output exceeds -opt_vtk_max_points; disable -opt_write_final_vtk for production runs");
    PetscCall(write_h8_opt_vtk(uda, eda, final_vtk_file, grid, u, rho_phys, mask));
  } else if (optimizer_options.write_final_vtk && stopped_on_ksp_divergence) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Final VTK skipped because the last displacement field came from a divergent linear solve.\n"));
  }
  PetscCall(write_h8_checkpoint(optimizer_options, final_iter, PETSC_TRUE,
                                rho_design, rho_phys, mask));
  PetscReal total_time = 0.0;
  PetscCall(elapsed_max(total_start, &total_time));
  PetscCall(write_h8_summary(output_prefix, grid, optimizer_options, final_iter,
                             compliance, volume, change, total_time));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&dc_design));
  PetscCall(VecDestroy(&dc_filtered));
  PetscCall(VecDestroy(&dc_phys));
  PetscCall(VecDestroy(&dv_design));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&filter_denom));
  PetscCall(VecDestroy(&rho_phys));
  PetscCall(VecDestroy(&rho_filtered));
  PetscCall(VecDestroy(&rho_design));
  PetscCall(VecDestroy(&update_mask));
  PetscCall(VecDestroy(&mask));
  PetscCall(DMDestroy(&uda));
  PetscCall(DMDestroy(&eda));
  return 0;
}

} // namespace control_arm
