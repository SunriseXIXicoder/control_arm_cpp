#include "control_arm/h8_optimizer.hpp"

#include <petscdmda.h>
#include <petscviewer.h>

#include <cstdio>

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

PetscErrorCode create_h8_dms(const Grid &grid, PetscInt element_stencil_width,
                             DM *uda, DM *eda) {
  PetscMPIInt ranks = 1;
  const PetscInt estencil = PetscMax(1, element_stencil_width);
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCheck(ranks <= grid.nz, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "H8 optimizer uses z-slab decomposition and needs ranks <= nz");
  PetscCheck((grid.nz - 1) >= ranks * estencil, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_OUTOFRANGE,
             "H8 optimizer needs each element z-slab at least as thick as the density filter stencil; increase nz or reduce ranks/filter radius");

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         grid.nx, grid.ny, grid.nz,
                         1, 1, ranks, 3, 1,
                         nullptr, nullptr, nullptr, uda));
  PetscCall(DMSetFromOptions(*uda));
  PetscCall(DMSetUp(*uda));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         grid.nx - 1, grid.ny - 1, grid.nz - 1,
                         1, 1, ranks, 1, estencil,
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
    PetscCall(VecCopy(rho_design, rho_phys));
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
        if (PetscRealPart(m[k][j][i]) <= 0.5) {
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
              if (w > 0.0 && PetscRealPart(m[kk][jj][ii]) > 0.5) {
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

PetscErrorCode apply_sensitivity_filter_adjoint(DM eda, Vec dc_phys, Vec mask,
                                                Vec denom, PetscReal radius,
                                                Vec dc_design) {
  if (radius <= 0.0) {
    PetscCall(VecCopy(dc_phys, dc_design));
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
        if (PetscRealPart(m[k][j][i]) <= 0.5) {
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
              if (PetscRealPart(m[kk][jj][ii]) <= 0.5) continue;
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
        const PetscBool design =
            cell_density(i, j, k, grid, density_options) >=
                    density_options.mask_threshold
                ? PETSC_TRUE
                : PETSC_FALSE;
        m[k][j][i] = design ? 1.0 : 0.0;
        r[k][j][i] = design ? opt.volfrac : density_options.void_density;
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
          if (i == 0) {
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
                  if (ni == 0) continue;
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
          if (i == 0) {
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

PetscErrorCode fill_h8_load(DM uda, const Grid &grid, PetscReal load, Vec b) {
  PetscScalar ****bg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(VecSet(b, 0.0));
  PetscCall(DMDAVecGetArrayDOF(uda, b, &bg));
  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (i == grid.nx - 1) {
          bg[k][j][i][2] = -load / static_cast<PetscReal>(grid.ny * grid.nz);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(uda, b, &bg));
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

        local_mask_sum += mask_value;
        local_rho_sum += mask_value * rho_value;
        if (mask_value <= 0.5) {
          d[ez][ey][ex] = 0.0;
          continue;
        }

        for (PetscInt node = 0; node < 8; ++node) {
          PetscInt ni = 0, nj = 0, nk = 0;
          h8_node_coords(ex, ey, ez, node, &ni, &nj, &nk);
          for (PetscInt c = 0; c < 3; ++c) {
            ue[3 * node + c] = ni == 0 ? 0.0 : PetscRealPart(ug[nk][nj][ni][c]);
          }
        }

        for (PetscInt a = 0; a < 24; ++a) {
          for (PetscInt b = 0; b < 24; ++b) {
            energy += ue[a] * ke[24 * a + b] * ue[b];
          }
        }
        d[ez][ey][ex] = -simp_derivative(rho_value, density_options) * energy;
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
  if (mask <= 0.5) return 0.0;
  if (dv <= 0.0) return opt.rho_min;
  const PetscReal ratio = PetscMax(1.0e-16, -dc / (lambda * dv));
  PetscReal value = rho * PetscSqrtReal(ratio);
  value = PetscMax(rho - opt.move, PetscMin(rho + opt.move, value));
  return PetscMax(opt.rho_min, PetscMin(1.0, value));
}

PetscErrorCode oc_update(DM eda, Vec rho, Vec mask, Vec dc, Vec dv,
                         const DensityOptions &density_options,
                         const OptimizerOptions &opt,
                         PetscReal *max_change) {
  PetscReal l1 = 1.0e-16, l2 = 1.0e16;
  PetscScalar mask_sum_scalar = 0.0;
  PetscReal mask_sum = 0.0;
  PetscCall(VecSum(mask, &mask_sum_scalar));
  mask_sum = PetscRealPart(mask_sum_scalar);
  const PetscReal target_volume_sum = opt.volfrac * mask_sum;

  for (PetscInt it = 0; it < 80; ++it) {
    const PetscReal lambda = 0.5 * (l1 + l2);
    PetscScalar ***r = nullptr, ***m = nullptr, ***d = nullptr, ***v = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscReal local_sum = 0.0, global_sum = 0.0;
    PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
    PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
    PetscCall(DMDAVecGetArrayRead(eda, dc, &d));
    PetscCall(DMDAVecGetArrayRead(eda, dv, &v));
    PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (PetscRealPart(m[k][j][i]) > 0.5) {
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
    PetscCallMPI(MPI_Allreduce(&local_sum, &global_sum, 1, MPIU_REAL, MPI_SUM,
                               PETSC_COMM_WORLD));
    if (global_sum > target_volume_sum) l1 = lambda;
    else l2 = lambda;
  }

  {
    const PetscReal lambda = l2;
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
          if (PetscRealPart(m[k][j][i]) > 0.5) {
            new_value = trial_rho(old_value, PetscRealPart(d[k][j][i]),
                                  PetscRealPart(v[k][j][i]),
                                  PetscRealPart(m[k][j][i]), lambda, opt);
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

PetscErrorCode write_h8_summary(const char *output_prefix, const Grid &grid,
                                const OptimizerOptions &opt,
                                PetscInt final_iter, PetscReal compliance,
                                PetscReal volume, PetscReal change) {
  PetscViewer viewer = nullptr;
  char path[PETSC_MAX_PATH_LEN];
  PetscLogDouble mem_current = 0.0, mem_peak = 0.0;
  PetscLogDouble mem_current_max = 0.0, mem_peak_max = 0.0;
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
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_iter=%lld\n",
                                   static_cast<long long>(final_iter)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_compliance=%.12e\n",
                                   static_cast<double>(compliance)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_volume=%.12e\n",
                                   static_cast<double>(volume)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_change=%.12e\n",
                                   static_cast<double>(change)));
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
  KSP ksp = nullptr;
  PetscViewer hist = nullptr;
  char hist_path[PETSC_MAX_PATH_LEN];
  PetscReal ke[24 * 24]{};
  PetscReal compliance = 0.0, volume = 0.0, change = 0.0;
  PetscInt final_iter = 0;
  const PetscInt filter_stencil =
      PetscMax(1, static_cast<PetscInt>(PetscCeilReal(optimizer_options.filter_radius)));

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
  PetscCall(VecDuplicate(rho_design, &dv_design));
  PetscCall(apply_sensitivity_filter_adjoint(eda, mask, mask, filter_denom,
                                             optimizer_options.filter_radius,
                                             dv_design));
  PetscCall(DMCreateGlobalVector(uda, &u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(DMCreateGlobalVector(eda, &dc_phys));
  PetscCall(VecDuplicate(dc_phys, &dc_design));
  PetscCall(fill_h8_load(uda, grid, optimizer_options.load, b));
  PetscCall(create_h8_shell_matrix(uda, eda, rho_phys, grid, density_options, &A));
  PetscCall(write_h8_checkpoint(optimizer_options, 0, PETSC_FALSE,
                                rho_design, rho_phys, mask));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPCG));
  {
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
  }
  PetscCall(KSPSetTolerances(ksp, optimizer_options.ksp_rtol, PETSC_DEFAULT,
                             PETSC_DEFAULT, optimizer_options.ksp_max_it));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(PetscSNPrintf(hist_path, sizeof(hist_path), "%s_history.csv",
                          output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, hist_path, &hist));
  PetscCall(PetscViewerASCIIPrintf(hist,
                                   "iter,compliance,volume,change,ksp_iterations,ksp_residual\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Optimize mode: h8_matrix_free nx=%lld ny=%lld nz=%lld max_iter=%lld volfrac=%g filter_radius=%g\n",
                        static_cast<long long>(grid.nx),
                        static_cast<long long>(grid.ny),
                        static_cast<long long>(grid.nz),
                        static_cast<long long>(optimizer_options.max_iter),
                        static_cast<double>(optimizer_options.volfrac),
                        static_cast<double>(optimizer_options.filter_radius)));

  for (PetscInt iter = 1; iter <= optimizer_options.max_iter; ++iter) {
    PetscInt its = 0;
    PetscReal rnorm = 0.0;
    PetscScalar dot = 0.0;
    PetscCall(apply_density_filter(eda, rho_design, mask, filter_denom,
                                   optimizer_options.filter_radius,
                                   density_options.void_density, rho_phys));
    PetscCall(VecSet(u, 0.0));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, u));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    PetscCall(VecDot(b, u, &dot));
    compliance = PetscRealPart(dot);
    PetscCall(compute_h8_sensitivity(uda, eda, u, rho_phys, mask, grid,
                                     density_options, ke, dc_phys, &volume));
    PetscCall(apply_sensitivity_filter_adjoint(eda, dc_phys, mask, filter_denom,
                                               optimizer_options.filter_radius,
                                               dc_design));
    PetscCall(oc_update(eda, rho_design, mask, dc_design, dv_design, density_options,
                        optimizer_options, &change));
    if (optimizer_options.write_checkpoint &&
        optimizer_options.checkpoint_interval > 0 &&
        iter % optimizer_options.checkpoint_interval == 0) {
      PetscCall(apply_density_filter(eda, rho_design, mask, filter_denom,
                                     optimizer_options.filter_radius,
                                     density_options.void_density, rho_phys));
      PetscCall(write_h8_checkpoint(optimizer_options, iter, PETSC_FALSE,
                                    rho_design, rho_phys, mask));
    }
    final_iter = iter;
    PetscCall(PetscViewerASCIIPrintf(hist, "%lld,%.12e,%.12e,%.12e,%lld,%.12e\n",
                                     static_cast<long long>(iter),
                                     static_cast<double>(compliance),
                                     static_cast<double>(volume),
                                     static_cast<double>(change),
                                     static_cast<long long>(its),
                                     static_cast<double>(rnorm)));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "it=%lld compliance=%.6e volume=%.6f change=%.3e ksp_it=%lld\n",
                          static_cast<long long>(iter),
                          static_cast<double>(compliance),
                          static_cast<double>(volume),
                          static_cast<double>(change),
                          static_cast<long long>(its)));
  }
  PetscCall(PetscViewerDestroy(&hist));

  {
    PetscScalar dot = 0.0;
    PetscInt its = 0;
    PetscReal rnorm = 0.0;
    PetscCall(apply_density_filter(eda, rho_design, mask, filter_denom,
                                   optimizer_options.filter_radius,
                                   density_options.void_density, rho_phys));
    PetscCall(VecSet(u, 0.0));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, u));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    PetscCall(VecDot(b, u, &dot));
    compliance = PetscRealPart(dot);
    PetscCall(compute_h8_sensitivity(uda, eda, u, rho_phys, mask, grid,
                                     density_options, ke, dc_phys, &volume));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "final compliance=%.6e volume=%.6f ksp_it=%lld residual=%.3e\n",
                          static_cast<double>(compliance),
                          static_cast<double>(volume),
                          static_cast<long long>(its),
                          static_cast<double>(rnorm)));
  }

  if (optimizer_options.write_final_vtk) {
    PetscCheck(node_count(grid) <= optimizer_options.vtk_max_points,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "H8 optimization VTK output exceeds -opt_vtk_max_points; disable -opt_write_final_vtk for production runs");
    PetscCall(write_h8_opt_vtk(uda, eda, final_vtk_file, grid, u, rho_phys, mask));
  }
  PetscCall(write_h8_checkpoint(optimizer_options, final_iter, PETSC_TRUE,
                                rho_design, rho_phys, mask));
  PetscCall(write_h8_summary(output_prefix, grid, optimizer_options, final_iter,
                             compliance, volume, change));

  PetscCall(KSPDestroy(&ksp));
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
