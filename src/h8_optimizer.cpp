#include "control_arm/h8_optimizer.hpp"

#include "control_arm/petsc_utils.hpp"

#include <petscdmda.h>
#include <petscviewer.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

struct ElementGrid {
  PetscInt ex = 0;
  PetscInt ey = 0;
  PetscInt ez = 0;
};

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
                           const Grid &grid) {
  return h8_is_c_support_node(i, j, k, grid);
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
                                  const DensityOptions &options) {
  if (!options.use_control_arm_mask) return PETSC_FALSE;

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

void compute_h8_element_stiffness(const Grid &grid,
                                  const DensityOptions &options,
                                  PetscReal *ke) {
  const PetscReal nu = 0.30;
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

PetscErrorCode compute_masked_volume_fraction(DM eda, Vec rho, Vec mask,
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
        if (h8_mask_is_design(mask_value)) {
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

PetscErrorCode create_h8_dms(const Grid &grid, PetscInt element_stencil_width,
                             DM *uda, DM *eda) {
  PetscMPIInt ranks = 1;
  const PetscInt estencil = PetscMax(1, element_stencil_width);
  PetscInt px = 1, py = 1, pz = 1;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCall(choose_h8_process_grid(grid, estencil, ranks, &px, &py, &pz));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 optimizer DMDA process grid: px=%lld py=%lld pz=%lld, element_stencil=%lld\n",
                        static_cast<long long>(px),
                        static_cast<long long>(py),
                        static_cast<long long>(pz),
                        static_cast<long long>(estencil)));

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
                         px, py, pz, 1, estencil,
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

PetscErrorCode apply_z_draft_closure(DM eda, Vec mask, PetscReal eta, Vec rho) {
  PetscScalar ***r = nullptr;
  PetscScalar ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  PetscMPIInt reduce_count = 0;

  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  const PetscInt nxy = ex * ey;
  PetscCall(PetscMPIIntCast(nxy, &reduce_count));
  const PetscReal threshold = PetscMax(0.0, PetscMin(1.0, eta));
  std::vector<PetscInt> local_min(static_cast<std::size_t>(nxy), ez);
  std::vector<PetscInt> local_max(static_cast<std::size_t>(nxy), -1);
  std::vector<PetscInt> global_min(static_cast<std::size_t>(nxy), ez);
  std::vector<PetscInt> global_max(static_cast<std::size_t>(nxy), -1);
  std::vector<PetscReal> local_peak(static_cast<std::size_t>(nxy), 0.0);
  std::vector<PetscReal> global_peak(static_cast<std::size_t>(nxy), 0.0);

  PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        const PetscReal rho_value = PetscRealPart(r[k][j][i]);
        if (!h8_mask_is_active(mask_value) || rho_value < threshold) continue;
        const PetscInt id = j * ex + i;
        local_min[static_cast<std::size_t>(id)] =
            PetscMin(local_min[static_cast<std::size_t>(id)], k);
        local_max[static_cast<std::size_t>(id)] =
            PetscMax(local_max[static_cast<std::size_t>(id)], k);
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
        const PetscInt id = j * ex + i;
        const PetscInt kmin = global_min[static_cast<std::size_t>(id)];
        const PetscInt kmax = global_max[static_cast<std::size_t>(id)];
        if (kmin <= k && k <= kmax) {
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
  PetscCall(DMCreateGlobalVector(eda, mask));
  PetscCall(VecDuplicate(*mask, rho));
  PetscCall(DMDAVecGetArray(eda, *mask, &m));
  PetscCall(DMDAVecGetArray(eda, *rho, &r));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscBool forced_solid =
            h8_cell_is_forced_solid(i, j, k, grid, density_options);
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
          r[k][j][i] = opt.volfrac;
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
          if (h8_is_fixed_node(i, j, k, g)) {
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
                  if (h8_is_fixed_node(ni, nj, nk, g)) continue;
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
          if (h8_is_fixed_node(i, j, k, g)) {
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
        if (h8_is_fixed_node(i, j, k, g)) {
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

          if (h8_is_fixed_node(i, j, k, grid)) {
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
            if (!h8_is_fixed_node(ii, jj, kk, grid)) {
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
                                         INSERT_VALUES));
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

          if (h8_is_fixed_node(i, j, k, grid)) {
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
                  if (h8_is_fixed_node(ni, nj, nk, grid)) continue;
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
          std::strcmp(h8_pc_type, "aux_elastic_hypre") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

PetscBool h8_pc_type_uses_elastic_aux_matrix(const char *h8_pc_type) {
  return (std::strcmp(h8_pc_type, "aux_elastic_gamg") == 0 ||
          std::strcmp(h8_pc_type, "aux_elastic_hypre") == 0)
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
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-h8_pc_type",
                                  h8_pc_type, sizeof(h8_pc_type),
                                  &has_h8_pc_type));
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
  } else {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "-h8_pc_type must be block_jacobi, jacobi, petsc, aux_gamg, aux_hypre, aux_elastic_gamg, or aux_elastic_hypre");
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

PetscReal h8_control_arm_case_weight(PetscInt load_case) {
  static const PetscReal weights[3] = {0.06579, 0.78539, 0.14882};
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
                          0.0, 0.0, 0.0};
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
        if (h8_is_fixed_node(i, j, k, grid)) continue;
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

PetscErrorCode fill_h8_load(DM uda, DM eda, Vec mask, const Grid &grid,
                            PetscReal load, Vec b) {
  PetscInt load_case = 0;
  PetscBool include_spring_load = PETSC_TRUE;
  PetscCall(h8_get_load_options(&load_case, &include_spring_load));
  PetscCall(VecSet(b, 0.0));
  if (load_case >= 1 && load_case <= 3) {
    PetscCall(add_h8_control_arm_load_case(uda, eda, mask, grid,
                                           load, load_case, 1.0,
                                           include_spring_load, b));
  } else {
    for (PetscInt c = 1; c <= 3; ++c) {
      PetscCall(add_h8_control_arm_load_case(uda, eda, mask, grid, load, c,
                                             h8_control_arm_case_weight(c),
                                             include_spring_load, b));
    }
  }
  return 0;
}

PetscErrorCode compute_h8_sensitivity(DM uda, DM eda, Vec u, Vec rho, Vec mask,
                                      const Grid &grid,
                                      const DensityOptions &density_options,
                                      const PetscReal *ke, Vec dc,
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

        if (h8_mask_is_design(mask_value)) {
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
                h8_is_fixed_node(ni, nj, nk, grid)
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
  if (dv <= 0.0) return opt.rho_min;
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

PetscErrorCode oc_update(DM eda, Vec rho, Vec mask, Vec dc, Vec dv,
                         const DensityOptions &density_options,
                         const OptimizerOptions &opt,
                         PetscReal *max_change) {
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
          PetscReal new_value = density_options.void_density;
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
  PetscCall(PetscViewerASCIIPrintf(viewer, "z_draft_closure=%s\n",
                                   opt.z_draft_closure ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "z_draft_eta=%.12e\n",
                                   static_cast<double>(opt.z_draft_eta)));
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

std::uint32_t read_be32(const unsigned char *b) {
  return (static_cast<std::uint32_t>(b[0]) << 24) |
         (static_cast<std::uint32_t>(b[1]) << 16) |
         (static_cast<std::uint32_t>(b[2]) << 8) |
         static_cast<std::uint32_t>(b[3]);
}

std::uint64_t read_be64(const unsigned char *b) {
  std::uint64_t value = 0;
  for (PetscInt i = 0; i < 8; ++i) {
    value = (value << 8) | static_cast<std::uint64_t>(b[i]);
  }
  return value;
}

PetscReal read_be_double(const unsigned char *b) {
  const std::uint64_t bits = read_be64(b);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return static_cast<PetscReal>(value);
}

PetscErrorCode load_vec_binary_checkpoint(Vec v, const char *path) {
  static const std::uint64_t kPetscVecFileClassId = 1211214ULL;
  MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(v));
  MPI_File fh;
  MPI_Status status;
  unsigned char header[16] = {};
  PetscInt vec_size = 0;
  PetscInt lo = 0, hi = 0;
  PetscMPIInt byte_count = 0;
  PetscMPIInt rank = 0;
  std::uint64_t file_class = 0;
  std::uint64_t file_size = 0;
  MPI_Offset header_bytes = 0;

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_File_open(comm, const_cast<char *>(path), MPI_MODE_RDONLY,
                             MPI_INFO_NULL, &fh));
  PetscCallMPI(MPI_File_read_at_all(fh, 0, header, 16, MPI_BYTE, &status));

  if (read_be32(header) == kPetscVecFileClassId) {
    file_class = read_be32(header);
    file_size = read_be32(header + 4);
    header_bytes = 8;
  } else if (read_be64(header) == kPetscVecFileClassId) {
    file_class = read_be64(header);
    file_size = read_be64(header + 8);
    header_bytes = 16;
  }

  PetscCheck(file_class == kPetscVecFileClassId, comm, PETSC_ERR_FILE_UNEXPECTED,
             "PETSc binary checkpoint is not a Vec file or has an unsupported header: %s",
             path);
  PetscCall(VecGetSize(v, &vec_size));
  PetscCheck(file_size == static_cast<std::uint64_t>(vec_size), comm,
             PETSC_ERR_FILE_UNEXPECTED,
             "PETSc binary Vec length mismatch for %s: file has %llu entries, expected %lld",
             path, static_cast<unsigned long long>(file_size),
             static_cast<long long>(vec_size));
  PetscCall(VecGetOwnershipRange(v, &lo, &hi));

  const PetscInt local_n = hi - lo;
  const std::size_t raw_bytes = static_cast<std::size_t>(local_n) * sizeof(double);
  PetscCheck(raw_bytes <= static_cast<std::size_t>(std::numeric_limits<PetscMPIInt>::max()),
             comm, PETSC_ERR_SUP,
             "Local PETSc checkpoint read is too large for one MPI-IO call");
  std::vector<unsigned char> raw(raw_bytes);
  byte_count = static_cast<PetscMPIInt>(raw_bytes);
  if (byte_count > 0) {
    const MPI_Offset offset =
        header_bytes + static_cast<MPI_Offset>(lo) * static_cast<MPI_Offset>(sizeof(double));
    PetscCallMPI(MPI_File_read_at_all(fh, offset, raw.data(), byte_count,
                                      MPI_BYTE, &status));
  }
  PetscCallMPI(MPI_File_close(&fh));

  PetscScalar *arr = nullptr;
  PetscCall(VecGetArray(v, &arr));
  for (PetscInt i = 0; i < local_n; ++i) {
    arr[i] = read_be_double(raw.data() + static_cast<std::size_t>(i) * sizeof(double));
  }
  PetscCall(VecRestoreArray(v, &arr));

  if (rank == 0) {
    PetscCall(PetscPrintf(comm,
                          "Loaded PETSc binary Vec checkpoint: %s entries=%llu header_bytes=%lld\n",
                          path, static_cast<unsigned long long>(file_size),
                          static_cast<long long>(header_bytes)));
  }
  return 0;
}

PetscErrorCode load_dmda_natural_binary_checkpoint(DM da, Vec v,
                                                   const char *path) {
  Vec natural = nullptr;
  PetscCall(DMDACreateNaturalVector(da, &natural));
  PetscCall(load_vec_binary_checkpoint(natural, path));
  PetscCall(DMDANaturalToGlobalBegin(da, natural, INSERT_VALUES, v));
  PetscCall(DMDANaturalToGlobalEnd(da, natural, INSERT_VALUES, v));
  PetscCall(VecDestroy(&natural));
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
  if (have_mask_file && optimizer_options.z_draft_closure) {
    PetscCall(apply_z_draft_closure(eda, mask, optimizer_options.z_draft_eta,
                                    rho));
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
  Vec mask = nullptr, rho_design = nullptr, rho_phys = nullptr;
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
  PetscCall(VecDuplicate(rho_design, &rho_phys));
  PetscCall(VecDuplicate(rho_design, &filter_denom));
  PetscCall(compute_filter_denominator(eda, mask,
                                       optimizer_options.filter_radius,
                                       filter_denom));
  PetscCall(apply_density_filter(eda, rho_design, mask, filter_denom,
                                 optimizer_options.filter_radius,
                                 density_options.void_density, rho_phys));
  if (optimizer_options.z_draft_closure) {
    PetscCall(apply_z_draft_closure(eda, mask, optimizer_options.z_draft_eta,
                                    rho_phys));
  }
  PetscCall(write_downsampled_h8_density_vtk(eda, vtk_file, grid, rho_phys,
                                             mask, stride, max_samples));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Wrote H8 initial design/mask VTK: %s\n", vtk_file));

  PetscCall(VecDestroy(&filter_denom));
  PetscCall(VecDestroy(&rho_phys));
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
  if (have_mask_file && optimizer_options.z_draft_closure) {
    PetscCall(apply_z_draft_closure(eda, mask, optimizer_options.z_draft_eta,
                                    rho));
  }

  PetscCall(DMCreateGlobalVector(uda, &u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(fill_h8_load(uda, eda, mask, grid, optimizer_options.load, b));
  PetscCall(create_h8_shell_matrix(uda, eda, rho, grid, density_options, &A));
  PetscCall(get_h8_pc_type_option(h8_pc_type, sizeof(h8_pc_type),
                                  &use_aux_matrix));
  use_elastic_aux_matrix = h8_pc_type_uses_elastic_aux_matrix(h8_pc_type);
  if (use_aux_matrix) {
    if (use_elastic_aux_matrix) {
      PetscCall(create_h8_aux_elasticity_matrix(uda, eda, rho, grid,
                                                density_options, &P));
    } else {
      PetscCall(create_h8_aux_laplacian_matrix(uda, eda, rho, grid,
                                               density_options, &P));
    }
  }
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(configure_h8_ksp(ksp, A, P, uda, eda, rho, grid, density_options,
                             optimizer_options));

  PetscCall(PetscTime(&solve_start));
  PetscCall(VecSet(u, 0.0));
  PetscCall(set_h8_ksp_operators(ksp, A, P, uda, eda, rho, grid,
                                 density_options, PETSC_FALSE,
                                 use_elastic_aux_matrix));
  PetscCall(KSPSolve(ksp, b, u));
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
  Vec rho_design = nullptr, rho_phys = nullptr, mask = nullptr;
  Vec u = nullptr, b = nullptr;
  Vec dc_phys = nullptr, dc_design = nullptr, dv_design = nullptr;
  Vec filter_denom = nullptr;
  Mat A = nullptr;
  Mat P = nullptr;
  KSP ksp = nullptr;
  PetscViewer hist = nullptr;
  char hist_path[PETSC_MAX_PATH_LEN];
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
  PetscCall(h8_get_load_options(&h8_load_case, &h8_include_spring_load));
  aux_rebuild_interval = PetscMax(1, aux_rebuild_interval);
  PetscCall(PetscTime(&total_start));
  compute_h8_element_stiffness(grid, density_options, ke);
  PetscCall(create_h8_dms(grid, filter_stencil, &uda, &eda));
  PetscCall(create_element_mask_and_density(eda, grid, density_options,
                                            optimizer_options, &mask, &rho_design));
  PetscCall(VecDuplicate(rho_design, &rho_phys));
  PetscCall(VecDuplicate(rho_design, &filter_denom));
  PetscCall(compute_filter_denominator(eda, mask, optimizer_options.filter_radius,
                                       filter_denom));
  PetscCall(apply_density_filter(eda, rho_design, mask, filter_denom,
                                 optimizer_options.filter_radius,
                                 density_options.void_density, rho_phys));
  if (optimizer_options.z_draft_closure) {
    PetscCall(apply_z_draft_closure(eda, mask, optimizer_options.z_draft_eta,
                                    rho_phys));
  }
  PetscCall(VecDuplicate(rho_design, &dv_design));
  PetscCall(apply_sensitivity_filter_adjoint(eda, mask, mask, filter_denom,
                                             optimizer_options.filter_radius,
                                             dv_design));
  PetscCall(DMCreateGlobalVector(uda, &u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(DMCreateGlobalVector(eda, &dc_phys));
  PetscCall(VecDuplicate(dc_phys, &dc_design));
  PetscCall(fill_h8_load(uda, eda, mask, grid, optimizer_options.load, b));
  PetscCall(VecNorm(b, NORM_2, &rhs_norm));
  PetscCall(create_h8_shell_matrix(uda, eda, rho_phys, grid, density_options, &A));
  PetscCall(get_h8_pc_type_option(h8_pc_type, sizeof(h8_pc_type),
                                  &use_aux_matrix));
  use_elastic_aux_matrix = h8_pc_type_uses_elastic_aux_matrix(h8_pc_type);
  if (use_aux_matrix) {
    if (use_elastic_aux_matrix) {
      PetscCall(create_h8_aux_elasticity_matrix(uda, eda, rho_phys, grid,
                                                density_options, &P));
    } else {
      PetscCall(create_h8_aux_laplacian_matrix(uda, eda, rho_phys, grid,
                                               density_options, &P));
    }
  }
  PetscCall(write_h8_checkpoint(optimizer_options, 0, PETSC_FALSE,
                                rho_design, rho_phys, mask));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(configure_h8_ksp(ksp, A, P, uda, eda, rho_phys, grid, density_options,
                             optimizer_options));

  PetscCall(PetscSNPrintf(hist_path, sizeof(hist_path), "%s_history.csv",
                          output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, hist_path, &hist));
  PetscCall(PetscViewerASCIIPrintf(hist,
                                   "iter,compliance,volume,change,ksp_iterations,ksp_reason,ksp_reason_name,ksp_residual,rhs_norm,ksp_relative_residual,filter_s,linear_solve_s,sensitivity_s,update_s,checkpoint_s,iter_total_s\n"));
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
                        "H8 hard regions/BC: A/B/C rings and spring mount are locked; C ring is fixed; A/B rings plus spring mount are loaded; z_draft_closure=%s eta=%g\n",
                        optimizer_options.z_draft_closure ? "true" : "false",
                        static_cast<double>(optimizer_options.z_draft_eta)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "H8 load options: load_case=%lld (%s), include_spring_load=%s\n",
                        static_cast<long long>(h8_load_case),
                        (h8_load_case >= 1 && h8_load_case <= 3)
                            ? "single case"
                            : "AHP weighted combined cases",
                        h8_include_spring_load ? "true" : "false"));
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

    PetscCall(PetscTime(&iter_start));
    PetscCall(PetscTime(&stage_start));
    PetscCall(apply_density_filter(eda, rho_design, mask, filter_denom,
                                   optimizer_options.filter_radius,
                                   density_options.void_density, rho_phys));
    if (optimizer_options.z_draft_closure) {
      PetscCall(apply_z_draft_closure(eda, mask, optimizer_options.z_draft_eta,
                                      rho_phys));
    }
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
    PetscCall(KSPSolve(ksp, b, u));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    rel_rnorm = rhs_norm > 0.0 ? rnorm / rhs_norm : rnorm;
    PetscCall(elapsed_max(stage_start, &solve_time));

    PetscCall(VecDot(b, u, &dot));
    compliance = PetscRealPart(dot);
    if (reason < 0) {
      PetscCall(compute_masked_volume_fraction(eda, rho_phys, mask, &volume));
      PetscCall(elapsed_max(iter_start, &iter_time));
      PetscCall(PetscViewerASCIIPrintf(hist, "%lld,%.12e,%.12e,%.12e,%lld,%d,%s,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                                       static_cast<long long>(iter),
                                       static_cast<double>(compliance),
                                       static_cast<double>(volume),
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
                            "it=%lld compliance=%.6e volume=%.6f change=%.3e ksp_it=%lld ksp_reason=%d(%s) residual=%.3e rel_res=%.3e time[s]: filter=%.3g linear_solve=%.3g total=%.3g\n",
                            static_cast<long long>(iter),
                            static_cast<double>(compliance),
                            static_cast<double>(volume),
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
                              "Stopping H8 optimization because the linear solve did not converge. The density update for this failed iteration was skipped; tune KSP/preconditioner/material continuation and restart from the last checkpoint.\n"));
        break;
      }
    }

    PetscCall(PetscTime(&stage_start));
    PetscCall(compute_h8_sensitivity(uda, eda, u, rho_phys, mask, grid,
                                     density_options, ke, dc_phys, &volume));
    PetscCall(apply_sensitivity_filter_adjoint(eda, dc_phys, mask, filter_denom,
                                               optimizer_options.filter_radius,
                                               dc_design));
    PetscCall(elapsed_max(stage_start, &sensitivity_time));

    PetscCall(PetscTime(&stage_start));
    PetscCall(oc_update(eda, rho_design, mask, dc_design, dv_design, density_options,
                        optimizer_options, &change));
    PetscCall(elapsed_max(stage_start, &update_time));

    PetscCall(PetscTime(&stage_start));
    if (optimizer_options.write_checkpoint &&
        optimizer_options.checkpoint_interval > 0 &&
        iter % optimizer_options.checkpoint_interval == 0) {
      PetscCall(apply_density_filter(eda, rho_design, mask, filter_denom,
                                     optimizer_options.filter_radius,
                                     density_options.void_density, rho_phys));
      if (optimizer_options.z_draft_closure) {
        PetscCall(apply_z_draft_closure(eda, mask, optimizer_options.z_draft_eta,
                                        rho_phys));
      }
      PetscCall(write_h8_checkpoint(optimizer_options, iter, PETSC_FALSE,
                                    rho_design, rho_phys, mask));
    }
    PetscCall(elapsed_max(stage_start, &checkpoint_time));
    PetscCall(elapsed_max(iter_start, &iter_time));
    final_iter = iter;
    PetscCall(PetscViewerASCIIPrintf(hist, "%lld,%.12e,%.12e,%.12e,%lld,%d,%s,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                                     static_cast<long long>(iter),
                                     static_cast<double>(compliance),
                                     static_cast<double>(volume),
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
                          "it=%lld compliance=%.6e volume=%.6f change=%.3e ksp_it=%lld ksp_reason=%d(%s) residual=%.3e rel_res=%.3e time[s]: filter=%.3g linear_solve=%.3g sens=%.3g update=%.3g checkpoint=%.3g total=%.3g\n",
                          static_cast<long long>(iter),
                          static_cast<double>(compliance),
                          static_cast<double>(volume),
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
  PetscCall(PetscViewerDestroy(&hist));

  if (!stopped_on_ksp_divergence) {
    PetscScalar dot = 0.0;
    PetscInt its = 0;
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    PetscReal rnorm = 0.0;
    PetscReal rel_rnorm = 0.0;
    PetscCall(apply_density_filter(eda, rho_design, mask, filter_denom,
                                   optimizer_options.filter_radius,
                                   density_options.void_density, rho_phys));
    if (optimizer_options.z_draft_closure) {
      PetscCall(apply_z_draft_closure(eda, mask, optimizer_options.z_draft_eta,
                                      rho_phys));
    }
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
                                     density_options, ke, dc_phys, &volume));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "final compliance=%.6e volume=%.6f ksp_it=%lld ksp_reason=%d(%s) residual=%.3e rel_res=%.3e\n",
                          static_cast<double>(compliance),
                          static_cast<double>(volume),
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
  PetscCall(VecDestroy(&dc_phys));
  PetscCall(VecDestroy(&dv_design));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&filter_denom));
  PetscCall(VecDestroy(&rho_phys));
  PetscCall(VecDestroy(&rho_design));
  PetscCall(VecDestroy(&mask));
  PetscCall(DMDestroy(&uda));
  PetscCall(DMDestroy(&eda));
  return 0;
}

} // namespace control_arm
