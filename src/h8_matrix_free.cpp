#include "control_arm/h8_matrix_free.hpp"

#include "control_arm/vtk.hpp"

#include <petscmath.h>

namespace control_arm {
namespace {

struct H8Context {
  DM da = nullptr;
  Vec local_x = nullptr;
  Grid grid;
  DensityOptions options;
  PetscReal ke[24 * 24]{};
};

PetscInt h8_local_node(PetscInt lx, PetscInt ly, PetscInt lz) {
  if (lz == 0) {
    if (ly == 0) {
      return lx == 0 ? 0 : 1;
    }
    return lx == 0 ? 3 : 2;
  }
  if (ly == 0) {
    return lx == 0 ? 4 : 5;
  }
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

PetscReal h8_cell_scale(PetscInt ex, PetscInt ey, PetscInt ez,
                        const Grid &grid,
                        const DensityOptions &options) {
  const PetscReal rho = cell_density(ex, ey, ez, grid, options);
  return options.emin + (1.0 - options.emin) * PetscPowReal(rho, options.penal);
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

  for (PetscInt i = 0; i < 24 * 24; ++i) {
    ke[i] = 0.0;
  }

  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      D[i][j] = lambda;
    }
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

PetscErrorCode h8_mult(Mat A, Vec x, Vec y) {
  H8Context *ctx = nullptr;
  PetscScalar ****xg = nullptr;
  PetscScalar ****yg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;

  PetscCall(MatShellGetContext(A, &ctx));
  const Grid &g = ctx->grid;
  const DensityOptions &options = ctx->options;

  PetscCall(DMGlobalToLocalBegin(ctx->da, x, INSERT_VALUES, ctx->local_x));
  PetscCall(DMGlobalToLocalEnd(ctx->da, x, INSERT_VALUES, ctx->local_x));
  PetscCall(DMDAVecGetArrayDOFRead(ctx->da, ctx->local_x, &xg));
  PetscCall(DMDAVecGetArrayDOF(ctx->da, y, &yg));
  PetscCall(DMDAGetCorners(ctx->da, &xs, &ys, &zs, &xm, &ym, &zm));

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
                const PetscReal scale = h8_cell_scale(ex, ey, ez, g, options);

                for (PetscInt col_node = 0; col_node < 8; ++col_node) {
                  PetscInt ni = 0, nj = 0, nk = 0;
                  h8_node_coords(ex, ey, ez, col_node, &ni, &nj, &nk);
                  if (ni == 0) {
                    continue;
                  }
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

  PetscCall(DMDAVecRestoreArrayDOFRead(ctx->da, ctx->local_x, &xg));
  PetscCall(DMDAVecRestoreArrayDOF(ctx->da, y, &yg));
  return 0;
}

PetscErrorCode h8_get_diagonal(Mat A, Vec diag_vec) {
  H8Context *ctx = nullptr;
  PetscScalar ****diag = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;

  PetscCall(MatShellGetContext(A, &ctx));
  const Grid &g = ctx->grid;
  const DensityOptions &options = ctx->options;

  PetscCall(DMDAVecGetArrayDOF(ctx->da, diag_vec, &diag));
  PetscCall(DMDAGetCorners(ctx->da, &xs, &ys, &zs, &xm, &ym, &zm));

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
                value += h8_cell_scale(ex, ey, ez, g, options) *
                         ctx->ke[24 * row + row];
              }
            }
          }
          diag[k][j][i][c] = value;
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(ctx->da, diag_vec, &diag));
  return 0;
}

PetscErrorCode h8_destroy(Mat A) {
  H8Context *ctx = nullptr;
  PetscCall(MatShellGetContext(A, &ctx));
  if (ctx) {
    PetscCall(VecDestroy(&ctx->local_x));
    PetscCall(DMDestroy(&ctx->da));
    delete ctx;
  }
  return 0;
}

PetscErrorCode fill_h8_load(DM da, const Grid &grid, PetscReal load, Vec b) {
  PetscScalar ****bg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;

  PetscCall(VecSet(b, 0.0));
  PetscCall(DMDAVecGetArrayDOF(da, b, &bg));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (i == grid.nx - 1) {
          bg[k][j][i][2] = -load / static_cast<PetscReal>(grid.ny * grid.nz);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(da, b, &bg));
  return 0;
}

} // namespace

PetscErrorCode create_h8_matrix_free_system(const Grid &grid,
                                            const DensityOptions &options,
                                            PetscReal load,
                                            Mat *A,
                                            Vec *u,
                                            Vec *b) {
  H8Context *ctx = new H8Context();
  PetscInt local_size = 0;

  ctx->grid = grid;
  ctx->options = options;
  compute_h8_element_stiffness(grid, options, ctx->ke);

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         grid.nx, grid.ny, grid.nz,
                         PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                         3, 1, nullptr, nullptr, nullptr, &ctx->da));
  PetscCall(DMSetFromOptions(ctx->da));
  PetscCall(DMSetUp(ctx->da));
  PetscCall(DMCreateGlobalVector(ctx->da, u));
  PetscCall(VecDuplicate(*u, b));
  PetscCall(DMCreateLocalVector(ctx->da, &ctx->local_x));
  PetscCall(VecGetLocalSize(*u, &local_size));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD, local_size, local_size,
                           dof_count(grid), dof_count(grid), ctx, A));
  PetscCall(MatShellSetOperation(*A, MATOP_MULT,
                                 reinterpret_cast<void (*)(void)>(h8_mult)));
  PetscCall(MatShellSetOperation(*A, MATOP_GET_DIAGONAL,
                                 reinterpret_cast<void (*)(void)>(h8_get_diagonal)));
  PetscCall(MatShellSetOperation(*A, MATOP_DESTROY,
                                 reinterpret_cast<void (*)(void)>(h8_destroy)));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(fill_h8_load(ctx->da, grid, load, *b));
  PetscCall(VecSet(*u, 0.0));
  return 0;
}

PetscErrorCode write_h8_solution_vtk(Mat A, Vec displacement,
                                     const char *path,
                                     const Grid &grid,
                                     const DensityOptions &options) {
  H8Context *ctx = nullptr;
  Vec natural = nullptr;

  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(DMDACreateNaturalVector(ctx->da, &natural));
  PetscCall(DMDAGlobalToNaturalBegin(ctx->da, displacement, INSERT_VALUES, natural));
  PetscCall(DMDAGlobalToNaturalEnd(ctx->da, displacement, INSERT_VALUES, natural));
  PetscCall(write_structured_solution_vtk(natural, path, grid, options));
  PetscCall(VecDestroy(&natural));
  return 0;
}

} // namespace control_arm
