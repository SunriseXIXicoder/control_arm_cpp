#include "control_arm/low_order_optimizer.hpp"

#include <petscdmda.h>

#include <cstdio>

namespace control_arm {
namespace {

struct LowOrderOptContext {
  DM uda = nullptr;
  DM sda = nullptr;
  Vec local_u = nullptr;
  Vec local_rho = nullptr;
  Vec rho = nullptr;
  Grid grid;
  DensityOptions density_options;
};

PetscErrorCode create_dms(const Grid &grid, DM *uda, DM *sda) {
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                         grid.nx, grid.ny, grid.nz,
                         PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                         3, 1, nullptr, nullptr, nullptr, uda));
  PetscCall(DMSetFromOptions(*uda));
  PetscCall(DMSetUp(*uda));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                         grid.nx, grid.ny, grid.nz,
                         PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                         1, 1, nullptr, nullptr, nullptr, sda));
  PetscCall(DMSetFromOptions(*sda));
  PetscCall(DMSetUp(*sda));
  return 0;
}

PetscErrorCode create_mask_and_density(DM sda, const Grid &grid,
                                       const DensityOptions &density_options,
                                       const OptimizerOptions &opt,
                                       Vec *mask, Vec *rho) {
  PetscScalar ***m = nullptr;
  PetscScalar ***r = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;

  PetscCall(DMCreateGlobalVector(sda, mask));
  PetscCall(VecDuplicate(*mask, rho));
  PetscCall(DMDAVecGetArray(sda, *mask, &m));
  PetscCall(DMDAVecGetArray(sda, *rho, &r));
  PetscCall(DMDAGetCorners(sda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscBool in_design =
            (i > 0 && node_density(i, j, k, grid, density_options) >=
                          density_options.mask_threshold)
                ? PETSC_TRUE
                : PETSC_FALSE;
        m[k][j][i] = in_design ? 1.0 : 0.0;
        r[k][j][i] = in_design ? opt.volfrac : density_options.void_density;
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(sda, *mask, &m));
  PetscCall(DMDAVecRestoreArray(sda, *rho, &r));
  return 0;
}

PetscErrorCode low_order_mult(Mat A, Vec x, Vec y) {
  LowOrderOptContext *ctx = nullptr;
  PetscScalar ****xg = nullptr;
  PetscScalar ****yg = nullptr;
  PetscScalar ***rho = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const PetscInt di[6] = {-1, 1, 0, 0, 0, 0};
  const PetscInt dj[6] = {0, 0, -1, 1, 0, 0};
  const PetscInt dk[6] = {0, 0, 0, 0, -1, 1};

  PetscCall(MatShellGetContext(A, &ctx));
  const Grid &g = ctx->grid;
  const DensityOptions &opts = ctx->density_options;
  const PetscReal hx = 1.0 / static_cast<PetscReal>(g.nx - 1);
  const PetscReal hy = 1.0 / static_cast<PetscReal>(g.ny - 1);
  const PetscReal hz = 1.0 / static_cast<PetscReal>(g.nz - 1);
  const PetscReal weight[6] = {1.0 / (hx * hx), 1.0 / (hx * hx),
                               1.0 / (hy * hy), 1.0 / (hy * hy),
                               1.0 / (hz * hz), 1.0 / (hz * hz)};

  PetscCall(DMGlobalToLocalBegin(ctx->uda, x, INSERT_VALUES, ctx->local_u));
  PetscCall(DMGlobalToLocalEnd(ctx->uda, x, INSERT_VALUES, ctx->local_u));
  PetscCall(DMGlobalToLocalBegin(ctx->sda, ctx->rho, INSERT_VALUES,
                                 ctx->local_rho));
  PetscCall(DMGlobalToLocalEnd(ctx->sda, ctx->rho, INSERT_VALUES,
                               ctx->local_rho));
  PetscCall(DMDAVecGetArrayDOFRead(ctx->uda, ctx->local_u, &xg));
  PetscCall(DMDAVecGetArrayDOF(ctx->uda, y, &yg));
  PetscCall(DMDAVecGetArrayRead(ctx->sda, ctx->local_rho, &rho));
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
          for (PetscInt q = 0; q < 6; ++q) {
            const PetscInt ii = i + di[q];
            const PetscInt jj = j + dj[q];
            const PetscInt kk = k + dk[q];
            if (ii < 0 || ii >= g.nx || jj < 0 || jj >= g.ny || kk < 0 ||
                kk >= g.nz) {
              continue;
            }
            const PetscReal kij =
                weight[q] * edge_stiffness(PetscRealPart(rho[k][j][i]),
                                            PetscRealPart(rho[kk][jj][ii]), opts);
            value += kij * xg[k][j][i][c];
            if (ii != 0) {
              value -= kij * xg[kk][jj][ii][c];
            }
          }
          yg[k][j][i][c] = value;
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(ctx->uda, ctx->local_u, &xg));
  PetscCall(DMDAVecRestoreArrayDOF(ctx->uda, y, &yg));
  PetscCall(DMDAVecRestoreArrayRead(ctx->sda, ctx->local_rho, &rho));
  return 0;
}

PetscErrorCode low_order_get_diagonal(Mat A, Vec diag_vec) {
  LowOrderOptContext *ctx = nullptr;
  PetscScalar ****diag = nullptr;
  PetscScalar ***rho = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const PetscInt di[6] = {-1, 1, 0, 0, 0, 0};
  const PetscInt dj[6] = {0, 0, -1, 1, 0, 0};
  const PetscInt dk[6] = {0, 0, 0, 0, -1, 1};

  PetscCall(MatShellGetContext(A, &ctx));
  const Grid &g = ctx->grid;
  const DensityOptions &opts = ctx->density_options;
  const PetscReal hx = 1.0 / static_cast<PetscReal>(g.nx - 1);
  const PetscReal hy = 1.0 / static_cast<PetscReal>(g.ny - 1);
  const PetscReal hz = 1.0 / static_cast<PetscReal>(g.nz - 1);
  const PetscReal weight[6] = {1.0 / (hx * hx), 1.0 / (hx * hx),
                               1.0 / (hy * hy), 1.0 / (hy * hy),
                               1.0 / (hz * hz), 1.0 / (hz * hz)};

  PetscCall(DMGlobalToLocalBegin(ctx->sda, ctx->rho, INSERT_VALUES,
                                 ctx->local_rho));
  PetscCall(DMGlobalToLocalEnd(ctx->sda, ctx->rho, INSERT_VALUES,
                               ctx->local_rho));
  PetscCall(DMDAVecGetArrayDOF(ctx->uda, diag_vec, &diag));
  PetscCall(DMDAVecGetArrayRead(ctx->sda, ctx->local_rho, &rho));
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
          for (PetscInt q = 0; q < 6; ++q) {
            const PetscInt ii = i + di[q];
            const PetscInt jj = j + dj[q];
            const PetscInt kk = k + dk[q];
            if (ii < 0 || ii >= g.nx || jj < 0 || jj >= g.ny || kk < 0 ||
                kk >= g.nz) {
              continue;
            }
            value += weight[q] *
                     edge_stiffness(PetscRealPart(rho[k][j][i]),
                                    PetscRealPart(rho[kk][jj][ii]), opts);
          }
          diag[k][j][i][c] = value;
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(ctx->uda, diag_vec, &diag));
  PetscCall(DMDAVecRestoreArrayRead(ctx->sda, ctx->local_rho, &rho));
  return 0;
}

PetscErrorCode low_order_destroy(Mat A) {
  LowOrderOptContext *ctx = nullptr;
  PetscCall(MatShellGetContext(A, &ctx));
  if (ctx) {
    PetscCall(VecDestroy(&ctx->local_u));
    PetscCall(VecDestroy(&ctx->local_rho));
    delete ctx;
  }
  return 0;
}

PetscErrorCode create_shell_matrix(DM uda, DM sda, Vec rho,
                                  const Grid &grid,
                                  const DensityOptions &density_options,
                                  Mat *A) {
  LowOrderOptContext *ctx = new LowOrderOptContext();
  Vec template_vec = nullptr;
  PetscInt local_size = 0;

  ctx->uda = uda;
  ctx->sda = sda;
  ctx->rho = rho;
  ctx->grid = grid;
  ctx->density_options = density_options;
  PetscCall(DMCreateLocalVector(uda, &ctx->local_u));
  PetscCall(DMCreateLocalVector(sda, &ctx->local_rho));
  PetscCall(DMCreateGlobalVector(uda, &template_vec));
  PetscCall(VecGetLocalSize(template_vec, &local_size));
  PetscCall(VecDestroy(&template_vec));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD, local_size, local_size,
                           dof_count(grid), dof_count(grid), ctx, A));
  PetscCall(MatShellSetOperation(*A, MATOP_MULT,
                                 reinterpret_cast<void (*)(void)>(low_order_mult)));
  PetscCall(MatShellSetOperation(*A, MATOP_GET_DIAGONAL,
                                 reinterpret_cast<void (*)(void)>(low_order_get_diagonal)));
  PetscCall(MatShellSetOperation(*A, MATOP_DESTROY,
                                 reinterpret_cast<void (*)(void)>(low_order_destroy)));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  return 0;
}

PetscErrorCode fill_load(DM uda, const Grid &grid, PetscReal load, Vec b) {
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

PetscReal edge_sensitivity_factor(PetscReal rho0, PetscReal rho1,
                                  const DensityOptions &opts) {
  const PetscReal avg = PetscMax(1.0e-12, PetscMin(1.0, 0.5 * (rho0 + rho1)));
  return 0.5 * opts.young_modulus * (1.0 - opts.emin) * opts.penal *
         PetscPowReal(avg, opts.penal - 1.0);
}

PetscErrorCode compute_sensitivity(DM uda, DM sda, Vec u, Vec rho,
                                   Vec mask, const Grid &grid,
                                   const DensityOptions &density_options,
                                   Vec dc, PetscReal *volume_fraction) {
  Vec local_u = nullptr;
  Vec local_rho = nullptr;
  PetscScalar ****ug = nullptr;
  PetscScalar ***rg = nullptr;
  PetscScalar ***mg = nullptr;
  PetscScalar ***dcg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_rho_sum = 0.0;
  PetscReal local_mask_sum = 0.0;
  PetscReal global_rho_sum = 0.0;
  PetscReal global_mask_sum = 0.0;
  const PetscInt di[6] = {-1, 1, 0, 0, 0, 0};
  const PetscInt dj[6] = {0, 0, -1, 1, 0, 0};
  const PetscInt dk[6] = {0, 0, 0, 0, -1, 1};
  const PetscReal hx = 1.0 / static_cast<PetscReal>(grid.nx - 1);
  const PetscReal hy = 1.0 / static_cast<PetscReal>(grid.ny - 1);
  const PetscReal hz = 1.0 / static_cast<PetscReal>(grid.nz - 1);
  const PetscReal weight[6] = {1.0 / (hx * hx), 1.0 / (hx * hx),
                               1.0 / (hy * hy), 1.0 / (hy * hy),
                               1.0 / (hz * hz), 1.0 / (hz * hz)};

  PetscCall(DMCreateLocalVector(uda, &local_u));
  PetscCall(DMCreateLocalVector(sda, &local_rho));
  PetscCall(DMGlobalToLocalBegin(uda, u, INSERT_VALUES, local_u));
  PetscCall(DMGlobalToLocalEnd(uda, u, INSERT_VALUES, local_u));
  PetscCall(DMGlobalToLocalBegin(sda, rho, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(sda, rho, INSERT_VALUES, local_rho));

  PetscCall(DMDAVecGetArrayDOFRead(uda, local_u, &ug));
  PetscCall(DMDAVecGetArrayRead(sda, local_rho, &rg));
  PetscCall(DMDAVecGetArrayRead(sda, mask, &mg));
  PetscCall(DMDAVecGetArray(sda, dc, &dcg));
  PetscCall(DMDAGetCorners(sda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal m = PetscRealPart(mg[k][j][i]);
        local_mask_sum += m;
        local_rho_sum += m * PetscRealPart(rg[k][j][i]);

        if (m <= 0.5) {
          dcg[k][j][i] = 0.0;
          continue;
        }

        PetscReal value = 0.0;
        for (PetscInt q = 0; q < 6; ++q) {
          const PetscInt ii = i + di[q];
          const PetscInt jj = j + dj[q];
          const PetscInt kk = k + dk[q];
          if (ii < 0 || ii >= grid.nx || jj < 0 || jj >= grid.ny ||
              kk < 0 || kk >= grid.nz) {
            continue;
          }
          const PetscReal dE =
              weight[q] * edge_sensitivity_factor(PetscRealPart(rg[k][j][i]),
                                                  PetscRealPart(rg[kk][jj][ii]),
                                                  density_options);
          PetscReal du2 = 0.0;
          for (PetscInt c = 0; c < 3; ++c) {
            const PetscReal up = PetscRealPart(ug[k][j][i][c]);
            const PetscReal uq = ii == 0 ? 0.0 : PetscRealPart(ug[kk][jj][ii][c]);
            const PetscReal du = up - uq;
            du2 += du * du;
          }
          value -= dE * du2;
        }
        dcg[k][j][i] = value;
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(uda, local_u, &ug));
  PetscCall(DMDAVecRestoreArrayRead(sda, local_rho, &rg));
  PetscCall(DMDAVecRestoreArrayRead(sda, mask, &mg));
  PetscCall(DMDAVecRestoreArray(sda, dc, &dcg));
  PetscCall(VecDestroy(&local_u));
  PetscCall(VecDestroy(&local_rho));

  PetscCallMPI(MPI_Allreduce(&local_rho_sum, &global_rho_sum, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_mask_sum, &global_mask_sum, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  *volume_fraction = global_mask_sum > 0.0 ? global_rho_sum / global_mask_sum : 0.0;
  return 0;
}

PetscReal trial_rho(PetscReal rho, PetscReal dc, PetscReal mask,
                    PetscReal lambda, const OptimizerOptions &opt) {
  if (mask <= 0.5) {
    return 0.0;
  }
  const PetscReal ratio = PetscMax(1.0e-16, -dc / lambda);
  PetscReal value = rho * PetscSqrtReal(ratio);
  value = PetscMax(rho - opt.move, PetscMin(rho + opt.move, value));
  return PetscMax(opt.rho_min, PetscMin(1.0, value));
}

PetscErrorCode oc_update(DM sda, Vec rho, Vec mask, Vec dc,
                         const DensityOptions &density_options,
                         const OptimizerOptions &opt,
                         PetscReal *max_change) {
  PetscReal l1 = 1.0e-16;
  PetscReal l2 = 1.0e16;
  PetscReal global_mask_sum = 0.0;
  PetscScalar mask_sum_scalar = 0.0;

  PetscCall(VecSum(mask, &mask_sum_scalar));
  global_mask_sum = PetscRealPart(mask_sum_scalar);

  for (PetscInt it = 0; it < 80; ++it) {
    const PetscReal lambda = 0.5 * (l1 + l2);
    PetscScalar ***r = nullptr;
    PetscScalar ***m = nullptr;
    PetscScalar ***d = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscReal local_sum = 0.0;
    PetscReal global_sum = 0.0;

    PetscCall(DMDAVecGetArrayRead(sda, rho, &r));
    PetscCall(DMDAVecGetArrayRead(sda, mask, &m));
    PetscCall(DMDAVecGetArrayRead(sda, dc, &d));
    PetscCall(DMDAGetCorners(sda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          const PetscReal mk = PetscRealPart(m[k][j][i]);
          if (mk > 0.5) {
            local_sum += trial_rho(PetscRealPart(r[k][j][i]),
                                   PetscRealPart(d[k][j][i]), mk, lambda, opt);
          }
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(sda, rho, &r));
    PetscCall(DMDAVecRestoreArrayRead(sda, mask, &m));
    PetscCall(DMDAVecRestoreArrayRead(sda, dc, &d));
    PetscCallMPI(MPI_Allreduce(&local_sum, &global_sum, 1, MPIU_REAL, MPI_SUM,
                               PETSC_COMM_WORLD));

    if (global_sum / global_mask_sum > opt.volfrac) {
      l1 = lambda;
    } else {
      l2 = lambda;
    }
  }

  {
    const PetscReal lambda = l2;
    PetscScalar ***r = nullptr;
    PetscScalar ***m = nullptr;
    PetscScalar ***d = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscReal local_change = 0.0;

    PetscCall(DMDAVecGetArray(sda, rho, &r));
    PetscCall(DMDAVecGetArrayRead(sda, mask, &m));
    PetscCall(DMDAVecGetArrayRead(sda, dc, &d));
    PetscCall(DMDAGetCorners(sda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          const PetscReal old_value = PetscRealPart(r[k][j][i]);
          PetscReal new_value = density_options.void_density;
          if (PetscRealPart(m[k][j][i]) > 0.5) {
            new_value = trial_rho(old_value, PetscRealPart(d[k][j][i]),
                                  PetscRealPart(m[k][j][i]), lambda, opt);
          }
          r[k][j][i] = new_value;
          local_change = PetscMax(local_change, PetscAbsReal(new_value - old_value));
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(sda, rho, &r));
    PetscCall(DMDAVecRestoreArrayRead(sda, mask, &m));
    PetscCall(DMDAVecRestoreArrayRead(sda, dc, &d));
    PetscCallMPI(MPI_Allreduce(&local_change, max_change, 1, MPIU_REAL, MPI_MAX,
                               PETSC_COMM_WORLD));
  }
  return 0;
}

PetscErrorCode write_optimizer_vtk(DM uda, DM sda, const char *path,
                                   const Grid &grid, Vec u, Vec rho, Vec mask) {
  PetscMPIInt rank = 0;
  Vec u_nat = nullptr, rho_nat = nullptr, mask_nat = nullptr;
  Vec u_seq = nullptr, rho_seq = nullptr, mask_seq = nullptr;
  VecScatter su = nullptr, sr = nullptr, sm = nullptr;
  const PetscScalar *ua = nullptr, *ra = nullptr, *ma = nullptr;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(DMDACreateNaturalVector(uda, &u_nat));
  PetscCall(DMDACreateNaturalVector(sda, &rho_nat));
  PetscCall(DMDACreateNaturalVector(sda, &mask_nat));
  PetscCall(DMDAGlobalToNaturalBegin(uda, u, INSERT_VALUES, u_nat));
  PetscCall(DMDAGlobalToNaturalEnd(uda, u, INSERT_VALUES, u_nat));
  PetscCall(DMDAGlobalToNaturalBegin(sda, rho, INSERT_VALUES, rho_nat));
  PetscCall(DMDAGlobalToNaturalEnd(sda, rho, INSERT_VALUES, rho_nat));
  PetscCall(DMDAGlobalToNaturalBegin(sda, mask, INSERT_VALUES, mask_nat));
  PetscCall(DMDAGlobalToNaturalEnd(sda, mask, INSERT_VALUES, mask_nat));
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
               "Cannot open optimizer VTK: %s", path);
    PetscCall(VecGetArrayRead(u_seq, &ua));
    PetscCall(VecGetArrayRead(rho_seq, &ra));
    PetscCall(VecGetArrayRead(mask_seq, &ma));
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "C++ PETSc low-order optimization result\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    std::fprintf(fp, "DIMENSIONS %lld %lld %lld\n", static_cast<long long>(grid.nx),
                 static_cast<long long>(grid.ny), static_cast<long long>(grid.nz));
    std::fprintf(fp, "ORIGIN 0 0 0\n");
    std::fprintf(fp, "SPACING 1 1 1\n");
    std::fprintf(fp, "POINT_DATA %lld\n", static_cast<long long>(node_count(grid)));
    std::fprintf(fp, "SCALARS rho double 1\nLOOKUP_TABLE default\n");
    for (PetscInt id = 0; id < node_count(grid); ++id) {
      std::fprintf(fp, "%.17e\n", static_cast<double>(PetscRealPart(ra[id])));
    }
    std::fprintf(fp, "SCALARS design_mask double 1\nLOOKUP_TABLE default\n");
    for (PetscInt id = 0; id < node_count(grid); ++id) {
      std::fprintf(fp, "%.17e\n", static_cast<double>(PetscRealPart(ma[id])));
    }
    std::fprintf(fp, "VECTORS displacement double\n");
    for (PetscInt id = 0; id < node_count(grid); ++id) {
      std::fprintf(fp, "%.17e %.17e %.17e\n",
                   static_cast<double>(PetscRealPart(ua[3 * id + 0])),
                   static_cast<double>(PetscRealPart(ua[3 * id + 1])),
                   static_cast<double>(PetscRealPart(ua[3 * id + 2])));
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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Wrote optimization VTK: %s\n", path));
  return 0;
}

PetscErrorCode write_summary(const char *output_prefix, const Grid &grid,
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
  PetscCall(PetscViewerASCIIPrintf(viewer, "operator=low_order_matrix_free\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "nx=%lld\nny=%lld\nnz=%lld\n",
                                   static_cast<long long>(grid.nx),
                                   static_cast<long long>(grid.ny),
                                   static_cast<long long>(grid.nz)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "dof=%lld\n",
                                   static_cast<long long>(dof_count(grid))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_iter=%lld\n",
                                   static_cast<long long>(opt.max_iter)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "volfrac_target=%.12e\n",
                                   static_cast<double>(opt.volfrac)));
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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Optimization summary: %s\n", path));
  return 0;
}

} // namespace

PetscErrorCode run_low_order_optimizer(const Grid &grid,
                                       const DensityOptions &density_options,
                                       const OptimizerOptions &optimizer_options,
                                       const char *output_prefix,
                                       const char *final_vtk_file) {
  DM uda = nullptr;
  DM sda = nullptr;
  Vec rho = nullptr, mask = nullptr, u = nullptr, b = nullptr, dc = nullptr;
  Mat A = nullptr;
  KSP ksp = nullptr;
  PetscViewer hist = nullptr;
  char hist_path[PETSC_MAX_PATH_LEN];
  PetscReal compliance = 0.0;
  PetscReal volume = 0.0;
  PetscReal change = 0.0;
  PetscInt final_iter = 0;

  PetscCall(create_dms(grid, &uda, &sda));
  PetscCall(create_mask_and_density(sda, grid, density_options,
                                    optimizer_options, &mask, &rho));
  PetscCall(DMCreateGlobalVector(uda, &u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(DMCreateGlobalVector(sda, &dc));
  PetscCall(fill_load(uda, grid, optimizer_options.load, b));
  PetscCall(create_shell_matrix(uda, sda, rho, grid, density_options, &A));

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
                        "Optimize mode: low_order_matrix_free nx=%lld ny=%lld nz=%lld max_iter=%lld volfrac=%g\n",
                        static_cast<long long>(grid.nx),
                        static_cast<long long>(grid.ny),
                        static_cast<long long>(grid.nz),
                        static_cast<long long>(optimizer_options.max_iter),
                        static_cast<double>(optimizer_options.volfrac)));

  for (PetscInt iter = 1; iter <= optimizer_options.max_iter; ++iter) {
    PetscInt its = 0;
    PetscReal rnorm = 0.0;
    PetscScalar dot = 0.0;
    char vtk_path[PETSC_MAX_PATH_LEN];

    PetscCall(VecSet(u, 0.0));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, u));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    PetscCall(VecDot(b, u, &dot));
    compliance = PetscRealPart(dot);
    PetscCall(compute_sensitivity(uda, sda, u, rho, mask, grid, density_options,
                                  dc, &volume));
    PetscCall(oc_update(sda, rho, mask, dc, density_options, optimizer_options,
                        &change));
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

    if (optimizer_options.vtk_interval > 0 &&
        iter % optimizer_options.vtk_interval == 0) {
      PetscCall(PetscSNPrintf(vtk_path, sizeof(vtk_path), "%s_iter_%04lld.vtk",
                              output_prefix, static_cast<long long>(iter)));
      PetscCall(write_optimizer_vtk(uda, sda, vtk_path, grid, u, rho, mask));
    }
  }

  PetscCall(PetscViewerDestroy(&hist));
  {
    PetscScalar final_dot = 0.0;
    PetscInt final_its = 0;
    PetscReal final_rnorm = 0.0;
    PetscCall(VecSet(u, 0.0));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, u));
    PetscCall(KSPGetIterationNumber(ksp, &final_its));
    PetscCall(KSPGetResidualNorm(ksp, &final_rnorm));
    PetscCall(VecDot(b, u, &final_dot));
    compliance = PetscRealPart(final_dot);
    PetscCall(compute_sensitivity(uda, sda, u, rho, mask, grid, density_options,
                                  dc, &volume));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "final compliance=%.6e volume=%.6f ksp_it=%lld residual=%.3e\n",
                          static_cast<double>(compliance),
                          static_cast<double>(volume),
                          static_cast<long long>(final_its),
                          static_cast<double>(final_rnorm)));
  }
  if (optimizer_options.write_final_vtk) {
    PetscCall(write_optimizer_vtk(uda, sda, final_vtk_file, grid, u, rho, mask));
  }
  PetscCall(write_summary(output_prefix, grid, optimizer_options, final_iter,
                          compliance, volume, change));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&rho));
  PetscCall(VecDestroy(&mask));
  PetscCall(DMDestroy(&uda));
  PetscCall(DMDestroy(&sda));
  return 0;
}

} // namespace control_arm
