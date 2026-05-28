#include "control_arm/density_pipeline.hpp"

#include <petscviewer.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace control_arm {
namespace {

enum class DraftDirection { PlusX, MinusX, PlusY, MinusY, PlusZ, MinusZ };
enum class DraftCombine { Product, Minimum, Maximum };

struct CellDM {
  DM da = nullptr;
  PetscInt ex = 0;
  PetscInt ey = 0;
  PetscInt ez = 0;
};

PetscReal clamp01(PetscReal value) {
  return PetscMax(0.0, PetscMin(1.0, value));
}

PetscReal smooth_heaviside(PetscReal x, PetscReal beta, PetscReal eta) {
  const PetscReal den = PetscTanhReal(beta * eta) +
                        PetscTanhReal(beta * (1.0 - eta));
  return (PetscTanhReal(beta * eta) + PetscTanhReal(beta * (x - eta))) / den;
}

PetscInt xy_id(PetscInt i, PetscInt j, PetscInt ex) {
  return i + ex * j;
}

PetscErrorCode parse_draft_dirs(const char *dirs,
                                std::vector<DraftDirection> *parsed) {
  parsed->clear();
  std::string s(dirs ? dirs : "");
  std::size_t start = 0;

  while (start < s.size()) {
    std::size_t end = s.find(',', start);
    if (end == std::string::npos) {
      end = s.size();
    }
    std::string token = s.substr(start, end - start);
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    std::transform(token.begin(), token.end(), token.begin(), ::tolower);

    if (!token.empty() && token != "none") {
      if (token == "+x") {
        parsed->push_back(DraftDirection::PlusX);
      } else if (token == "-x") {
        parsed->push_back(DraftDirection::MinusX);
      } else if (token == "+y") {
        parsed->push_back(DraftDirection::PlusY);
      } else if (token == "-y") {
        parsed->push_back(DraftDirection::MinusY);
      } else if (token == "+z") {
        parsed->push_back(DraftDirection::PlusZ);
      } else if (token == "-z") {
        parsed->push_back(DraftDirection::MinusZ);
      } else {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                "Unknown draft direction. Use comma-separated +x,-x,+y,-y,+z,-z or none");
      }
    }
    start = end + 1;
  }
  return 0;
}

const char *direction_name(DraftDirection dir) {
  switch (dir) {
    case DraftDirection::PlusX: return "+x";
    case DraftDirection::MinusX: return "-x";
    case DraftDirection::PlusY: return "+y";
    case DraftDirection::MinusY: return "-y";
    case DraftDirection::PlusZ: return "+z";
    case DraftDirection::MinusZ: return "-z";
  }
  return "?";
}

PetscErrorCode parse_draft_combine(const char *combine, DraftCombine *parsed) {
  std::string token(combine ? combine : "max");
  token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
  std::transform(token.begin(), token.end(), token.begin(), ::tolower);
  if (token == "product") {
    *parsed = DraftCombine::Product;
  } else if (token == "min" || token == "minimum") {
    *parsed = DraftCombine::Minimum;
  } else if (token == "max" || token == "maximum") {
    *parsed = DraftCombine::Maximum;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
            "-draft_combine must be max, min, or product");
  }
  return 0;
}

const char *combine_name(DraftCombine combine) {
  switch (combine) {
    case DraftCombine::Product: return "product";
    case DraftCombine::Minimum: return "min";
    case DraftCombine::Maximum: return "max";
  }
  return "?";
}

PetscErrorCode create_cell_dm(const Grid &grid, PetscInt stencil_width,
                              CellDM *cell_dm) {
  PetscMPIInt ranks = 1;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  cell_dm->ex = grid.nx - 1;
  cell_dm->ey = grid.ny - 1;
  cell_dm->ez = grid.nz - 1;

  PetscCheck(ranks <= cell_dm->ez, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "density mode currently uses z-slab decomposition and needs ranks <= nz-1");
  PetscCheck(stencil_width == 0 || cell_dm->ez >= ranks * stencil_width,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "density mode needs each z-slab at least as thick as the filter/draft stencil; increase nz or reduce MPI ranks/radius");

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         cell_dm->ex, cell_dm->ey, cell_dm->ez,
                         1, 1, ranks, 1, stencil_width,
                         nullptr, nullptr, nullptr, &cell_dm->da));
  PetscCall(DMSetFromOptions(cell_dm->da));
  PetscCall(DMSetUp(cell_dm->da));
  return 0;
}

PetscErrorCode create_initial_density(const CellDM &cell_dm, const Grid &grid,
                                      const DensityOptions &options, Vec *rho) {
  PetscScalar ***a = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;

  PetscCall(DMCreateGlobalVector(cell_dm.da, rho));
  PetscCall(DMDAVecGetArray(cell_dm.da, *rho, &a));
  PetscCall(DMDAGetCorners(cell_dm.da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        a[k][j][i] = cell_density(i, j, k, grid, options);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(cell_dm.da, *rho, &a));
  return 0;
}

PetscErrorCode apply_cone_filter(const CellDM &cell_dm, Vec rho, Vec filtered,
                                 PetscReal radius) {
  Vec local = nullptr;
  PetscScalar ***src = nullptr;
  PetscScalar ***dst = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const PetscInt r = PetscMax(0, static_cast<PetscInt>(PetscCeilReal(radius)));

  if (radius <= 0.0) {
    PetscCall(VecCopy(rho, filtered));
    return 0;
  }

  PetscCall(DMCreateLocalVector(cell_dm.da, &local));
  PetscCall(DMGlobalToLocalBegin(cell_dm.da, rho, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(cell_dm.da, rho, INSERT_VALUES, local));
  PetscCall(DMDAVecGetArrayRead(cell_dm.da, local, &src));
  PetscCall(DMDAVecGetArray(cell_dm.da, filtered, &dst));
  PetscCall(DMDAGetCorners(cell_dm.da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        PetscReal sum = 0.0;
        PetscReal weight_sum = 0.0;
        for (PetscInt dz = -r; dz <= r; ++dz) {
          const PetscInt kk = k + dz;
          if (kk < 0 || kk >= cell_dm.ez) continue;
          for (PetscInt dy = -r; dy <= r; ++dy) {
            const PetscInt jj = j + dy;
            if (jj < 0 || jj >= cell_dm.ey) continue;
            for (PetscInt dx = -r; dx <= r; ++dx) {
              const PetscInt ii = i + dx;
              if (ii < 0 || ii >= cell_dm.ex) continue;
              const PetscReal dist =
                  PetscSqrtReal(static_cast<PetscReal>(dx * dx + dy * dy + dz * dz));
              const PetscReal w = PetscMax(0.0, radius - dist);
              if (w > 0.0) {
                sum += w * PetscRealPart(src[kk][jj][ii]);
                weight_sum += w;
              }
            }
          }
        }
        dst[k][j][i] = weight_sum > 0.0 ? sum / weight_sum : src[k][j][i];
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(cell_dm.da, local, &src));
  PetscCall(DMDAVecRestoreArray(cell_dm.da, filtered, &dst));
  PetscCall(VecDestroy(&local));
  return 0;
}

PetscReal seed_power(PetscScalar rho, PetscReal q) {
  return PetscPowReal(clamp01(1.0 - PetscRealPart(rho)), q);
}

PetscReal rho_from_accum(PetscReal sum, PetscReal count,
                         const DensityPipelineOptions &opts) {
  if (count <= 0.0) {
    return 1.0;
  }
  const PetscReal mu = PetscPowReal(PetscMax(sum / count, 0.0),
                                    1.0 / opts.draft_pnorm);
  const PetscReal cut = clamp01(smooth_heaviside(mu, opts.draft_beta, opts.draft_eta));
  return clamp01(1.0 - cut);
}

void combine_rho(PetscScalar *accum, PetscReal rho_dir, DraftCombine combine) {
  if (combine == DraftCombine::Product) {
    *accum *= rho_dir;
  } else if (combine == DraftCombine::Minimum) {
    *accum = PetscMin(PetscRealPart(*accum), rho_dir);
  } else {
    *accum = PetscMax(PetscRealPart(*accum), rho_dir);
  }
}

PetscErrorCode apply_x_projection(const CellDM &cell_dm, Vec filtered, Vec product,
                                  DraftDirection dir,
                                  DraftCombine combine,
                                  const DensityPipelineOptions &opts) {
  Vec local = nullptr;
  PetscScalar ***src = nullptr;
  PetscScalar ***prod = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const PetscInt r = PetscMax(0, opts.draft_radius);
  const PetscReal r2 = static_cast<PetscReal>(r * r);

  PetscCall(DMCreateLocalVector(cell_dm.da, &local));
  PetscCall(DMGlobalToLocalBegin(cell_dm.da, filtered, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(cell_dm.da, filtered, INSERT_VALUES, local));
  PetscCall(DMDAVecGetArrayRead(cell_dm.da, local, &src));
  PetscCall(DMDAVecGetArray(cell_dm.da, product, &prod));
  PetscCall(DMDAGetCorners(cell_dm.da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = 0; j < cell_dm.ey; ++j) {
      PetscReal run_sum = 0.0;
      PetscReal run_count = 0.0;
      const PetscInt first = dir == DraftDirection::PlusX ? cell_dm.ex - 1 : 0;
      const PetscInt last = dir == DraftDirection::PlusX ? -1 : cell_dm.ex;
      const PetscInt step = dir == DraftDirection::PlusX ? -1 : 1;
      for (PetscInt i = first; i != last; i += step) {
        PetscReal plane_sum = 0.0;
        PetscReal plane_count = 0.0;
        for (PetscInt dz = -r; dz <= r; ++dz) {
          const PetscInt kk = k + dz;
          if (kk < 0 || kk >= cell_dm.ez) continue;
          for (PetscInt dy = -r; dy <= r; ++dy) {
            const PetscInt jj = j + dy;
            if (jj < 0 || jj >= cell_dm.ey) continue;
            if (static_cast<PetscReal>(dy * dy + dz * dz) > r2) continue;
            plane_sum += seed_power(src[kk][jj][i], opts.draft_pnorm);
            plane_count += 1.0;
          }
        }
        run_sum += plane_sum;
        run_count += plane_count;
        if (i >= xs && i < xs + xm && j >= ys && j < ys + ym) {
          combine_rho(&prod[k][j][i],
                      rho_from_accum(run_sum, run_count, opts),
                      combine);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(cell_dm.da, local, &src));
  PetscCall(DMDAVecRestoreArray(cell_dm.da, product, &prod));
  PetscCall(VecDestroy(&local));
  return 0;
}

PetscErrorCode apply_y_projection(const CellDM &cell_dm, Vec filtered, Vec product,
                                  DraftDirection dir,
                                  DraftCombine combine,
                                  const DensityPipelineOptions &opts) {
  Vec local = nullptr;
  PetscScalar ***src = nullptr;
  PetscScalar ***prod = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const PetscInt r = PetscMax(0, opts.draft_radius);
  const PetscReal r2 = static_cast<PetscReal>(r * r);

  PetscCall(DMCreateLocalVector(cell_dm.da, &local));
  PetscCall(DMGlobalToLocalBegin(cell_dm.da, filtered, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(cell_dm.da, filtered, INSERT_VALUES, local));
  PetscCall(DMDAVecGetArrayRead(cell_dm.da, local, &src));
  PetscCall(DMDAVecGetArray(cell_dm.da, product, &prod));
  PetscCall(DMDAGetCorners(cell_dm.da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt i = 0; i < cell_dm.ex; ++i) {
      PetscReal run_sum = 0.0;
      PetscReal run_count = 0.0;
      const PetscInt first = dir == DraftDirection::PlusY ? cell_dm.ey - 1 : 0;
      const PetscInt last = dir == DraftDirection::PlusY ? -1 : cell_dm.ey;
      const PetscInt step = dir == DraftDirection::PlusY ? -1 : 1;
      for (PetscInt j = first; j != last; j += step) {
        PetscReal plane_sum = 0.0;
        PetscReal plane_count = 0.0;
        for (PetscInt dz = -r; dz <= r; ++dz) {
          const PetscInt kk = k + dz;
          if (kk < 0 || kk >= cell_dm.ez) continue;
          for (PetscInt dx = -r; dx <= r; ++dx) {
            const PetscInt ii = i + dx;
            if (ii < 0 || ii >= cell_dm.ex) continue;
            if (static_cast<PetscReal>(dx * dx + dz * dz) > r2) continue;
            plane_sum += seed_power(src[kk][j][ii], opts.draft_pnorm);
            plane_count += 1.0;
          }
        }
        run_sum += plane_sum;
        run_count += plane_count;
        if (i >= xs && i < xs + xm && j >= ys && j < ys + ym) {
          combine_rho(&prod[k][j][i],
                      rho_from_accum(run_sum, run_count, opts),
                      combine);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(cell_dm.da, local, &src));
  PetscCall(DMDAVecRestoreArray(cell_dm.da, product, &prod));
  PetscCall(VecDestroy(&local));
  return 0;
}

void z_plane_contribution(PetscScalar ***src, const CellDM &cell_dm,
                          PetscInt seed_z, const DensityPipelineOptions &opts,
                          std::vector<PetscReal> *sum,
                          std::vector<PetscReal> *count) {
  const PetscInt r = PetscMax(0, opts.draft_radius);
  const PetscReal r2 = static_cast<PetscReal>(r * r);
  std::fill(sum->begin(), sum->end(), 0.0);
  std::fill(count->begin(), count->end(), 0.0);

  for (PetscInt j = 0; j < cell_dm.ey; ++j) {
    for (PetscInt i = 0; i < cell_dm.ex; ++i) {
      PetscReal s = 0.0;
      PetscReal c = 0.0;
      for (PetscInt dy = -r; dy <= r; ++dy) {
        const PetscInt jj = j + dy;
        if (jj < 0 || jj >= cell_dm.ey) continue;
        for (PetscInt dx = -r; dx <= r; ++dx) {
          const PetscInt ii = i + dx;
          if (ii < 0 || ii >= cell_dm.ex) continue;
          if (static_cast<PetscReal>(dx * dx + dy * dy) > r2) continue;
          s += seed_power(src[seed_z][jj][ii], opts.draft_pnorm);
          c += 1.0;
        }
      }
      const PetscInt id = xy_id(i, j, cell_dm.ex);
      (*sum)[id] = s;
      (*count)[id] = c;
    }
  }
}

PetscErrorCode exscan_array(MPI_Comm comm, const std::vector<PetscReal> &local,
                            std::vector<PetscReal> *incoming) {
  PetscMPIInt rank = 0;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Exscan(local.data(), incoming->data(),
                          static_cast<int>(local.size()), MPIU_REAL, MPI_SUM, comm));
  if (rank == 0) {
    std::fill(incoming->begin(), incoming->end(), 0.0);
  }
  return 0;
}

PetscErrorCode apply_z_projection(const CellDM &cell_dm, Vec filtered, Vec product,
                                  DraftDirection dir,
                                  DraftCombine combine,
                                  const DensityPipelineOptions &opts) {
  MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(cell_dm.da));
  Vec local = nullptr;
  PetscScalar ***src = nullptr;
  PetscScalar ***prod = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const PetscInt nxy = cell_dm.ex * cell_dm.ey;
  std::vector<PetscReal> plane_sum(nxy), plane_count(nxy);
  std::vector<PetscReal> local_total_sum(nxy, 0.0), local_total_count(nxy, 0.0);
  std::vector<PetscReal> incoming_sum(nxy, 0.0), incoming_count(nxy, 0.0);
  std::vector<PetscReal> running_sum(nxy, 0.0), running_count(nxy, 0.0);
  MPI_Comm scan_comm = MPI_COMM_NULL;

  PetscCall(DMCreateLocalVector(cell_dm.da, &local));
  PetscCall(DMGlobalToLocalBegin(cell_dm.da, filtered, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(cell_dm.da, filtered, INSERT_VALUES, local));
  PetscCall(DMDAVecGetArrayRead(cell_dm.da, local, &src));
  PetscCall(DMDAVecGetArray(cell_dm.da, product, &prod));
  PetscCall(DMDAGetCorners(cell_dm.da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    z_plane_contribution(src, cell_dm, k, opts, &plane_sum, &plane_count);
    for (PetscInt id = 0; id < nxy; ++id) {
      local_total_sum[id] += plane_sum[id];
      local_total_count[id] += plane_count[id];
    }
  }

  if (dir == DraftDirection::MinusZ) {
    PetscCall(exscan_array(comm, local_total_sum, &incoming_sum));
    PetscCall(exscan_array(comm, local_total_count, &incoming_count));
  } else {
    PetscMPIInt rank = 0;
    PetscMPIInt size = 1;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCallMPI(MPI_Comm_split(comm, 0, size - 1 - rank, &scan_comm));
    PetscCall(exscan_array(scan_comm, local_total_sum, &incoming_sum));
    PetscCall(exscan_array(scan_comm, local_total_count, &incoming_count));
    PetscCallMPI(MPI_Comm_free(&scan_comm));
  }

  running_sum = incoming_sum;
  running_count = incoming_count;

  const PetscInt first = dir == DraftDirection::MinusZ ? zs : zs + zm - 1;
  const PetscInt last = dir == DraftDirection::MinusZ ? zs + zm : zs - 1;
  const PetscInt step = dir == DraftDirection::MinusZ ? 1 : -1;
  for (PetscInt k = first; k != last; k += step) {
    z_plane_contribution(src, cell_dm, k, opts, &plane_sum, &plane_count);
    for (PetscInt id = 0; id < nxy; ++id) {
      running_sum[id] += plane_sum[id];
      running_count[id] += plane_count[id];
    }
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscInt id = xy_id(i, j, cell_dm.ex);
        combine_rho(&prod[k][j][i],
                    rho_from_accum(running_sum[id], running_count[id], opts),
                    combine);
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(cell_dm.da, local, &src));
  PetscCall(DMDAVecRestoreArray(cell_dm.da, product, &prod));
  PetscCall(VecDestroy(&local));
  return 0;
}

PetscErrorCode apply_draft_projection(const CellDM &cell_dm, Vec filtered,
                                      Vec projected,
                                      const DensityPipelineOptions &opts,
                                      const std::vector<DraftDirection> &dirs) {
  Vec product = nullptr;
  PetscScalar ***src = nullptr;
  PetscScalar ***prod = nullptr;
  PetscScalar ***dst = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  DraftCombine combine = DraftCombine::Maximum;

  if (dirs.empty()) {
    PetscCall(VecCopy(filtered, projected));
    return 0;
  }

  PetscCall(parse_draft_combine(opts.draft_combine, &combine));
  PetscCall(VecDuplicate(filtered, &product));
  PetscCall(VecSet(product, combine == DraftCombine::Maximum ? 0.0 : 1.0));

  for (DraftDirection dir : dirs) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Applying draft projection %s\n",
                          direction_name(dir)));
    if (dir == DraftDirection::PlusX || dir == DraftDirection::MinusX) {
      PetscCall(apply_x_projection(cell_dm, filtered, product, dir, combine, opts));
    } else if (dir == DraftDirection::PlusY || dir == DraftDirection::MinusY) {
      PetscCall(apply_y_projection(cell_dm, filtered, product, dir, combine, opts));
    } else {
      PetscCall(apply_z_projection(cell_dm, filtered, product, dir, combine, opts));
    }
  }

  PetscCall(DMDAVecGetArrayRead(cell_dm.da, filtered, &src));
  PetscCall(DMDAVecGetArrayRead(cell_dm.da, product, &prod));
  PetscCall(DMDAVecGetArray(cell_dm.da, projected, &dst));
  PetscCall(DMDAGetCorners(cell_dm.da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        dst[k][j][i] = PetscMin(PetscRealPart(src[k][j][i]),
                                PetscRealPart(prod[k][j][i]));
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(cell_dm.da, filtered, &src));
  PetscCall(DMDAVecRestoreArrayRead(cell_dm.da, product, &prod));
  PetscCall(DMDAVecRestoreArray(cell_dm.da, projected, &dst));
  PetscCall(VecDestroy(&product));
  return 0;
}

PetscErrorCode vec_average(Vec x, PetscReal *avg) {
  PetscScalar sum = 0.0;
  PetscInt n = 0;
  PetscCall(VecSum(x, &sum));
  PetscCall(VecGetSize(x, &n));
  *avg = n > 0 ? PetscRealPart(sum) / static_cast<PetscReal>(n) : 0.0;
  return 0;
}

PetscErrorCode write_density_pipeline_vtk(const CellDM &cell_dm,
                                          const char *path,
                                          Vec rho,
                                          Vec filtered,
                                          Vec projected) {
  PetscMPIInt rank = 0;
  Vec rho_nat = nullptr, filt_nat = nullptr, proj_nat = nullptr;
  Vec rho_seq = nullptr, filt_seq = nullptr, proj_seq = nullptr;
  VecScatter s1 = nullptr, s2 = nullptr, s3 = nullptr;
  const PetscScalar *r = nullptr, *f = nullptr, *p = nullptr;

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMDACreateNaturalVector(cell_dm.da, &rho_nat));
  PetscCall(VecDuplicate(rho_nat, &filt_nat));
  PetscCall(VecDuplicate(rho_nat, &proj_nat));
  PetscCall(DMDAGlobalToNaturalBegin(cell_dm.da, rho, INSERT_VALUES, rho_nat));
  PetscCall(DMDAGlobalToNaturalEnd(cell_dm.da, rho, INSERT_VALUES, rho_nat));
  PetscCall(DMDAGlobalToNaturalBegin(cell_dm.da, filtered, INSERT_VALUES, filt_nat));
  PetscCall(DMDAGlobalToNaturalEnd(cell_dm.da, filtered, INSERT_VALUES, filt_nat));
  PetscCall(DMDAGlobalToNaturalBegin(cell_dm.da, projected, INSERT_VALUES, proj_nat));
  PetscCall(DMDAGlobalToNaturalEnd(cell_dm.da, projected, INSERT_VALUES, proj_nat));
  PetscCall(VecScatterCreateToZero(rho_nat, &s1, &rho_seq));
  PetscCall(VecScatterCreateToZero(filt_nat, &s2, &filt_seq));
  PetscCall(VecScatterCreateToZero(proj_nat, &s3, &proj_seq));
  PetscCall(VecScatterBegin(s1, rho_nat, rho_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(s1, rho_nat, rho_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(s2, filt_nat, filt_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(s2, filt_nat, filt_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(s3, proj_nat, proj_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(s3, proj_nat, proj_seq, INSERT_VALUES, SCATTER_FORWARD));

  if (rank == 0) {
    FILE *fp = std::fopen(path, "w");
    PetscCheck(fp != nullptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Cannot open density pipeline VTK: %s", path);
    PetscCall(VecGetArrayRead(rho_seq, &r));
    PetscCall(VecGetArrayRead(filt_seq, &f));
    PetscCall(VecGetArrayRead(proj_seq, &p));

    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "C++ PETSc distributed density pipeline\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    std::fprintf(fp, "DIMENSIONS %lld %lld %lld\n",
                 static_cast<long long>(cell_dm.ex + 1),
                 static_cast<long long>(cell_dm.ey + 1),
                 static_cast<long long>(cell_dm.ez + 1));
    std::fprintf(fp, "ORIGIN 0 0 0\n");
    std::fprintf(fp, "SPACING 1 1 1\n");
    std::fprintf(fp, "CELL_DATA %lld\n",
                 static_cast<long long>(cell_dm.ex * cell_dm.ey * cell_dm.ez));
    auto write_scalar = [&](const char *name, const PetscScalar *data) {
      std::fprintf(fp, "SCALARS %s double 1\n", name);
      std::fprintf(fp, "LOOKUP_TABLE default\n");
      for (PetscInt id = 0; id < cell_dm.ex * cell_dm.ey * cell_dm.ez; ++id) {
        std::fprintf(fp, "%.17e\n", static_cast<double>(PetscRealPart(data[id])));
      }
    };
    write_scalar("rho_initial", r);
    write_scalar("rho_filtered", f);
    write_scalar("rho_projected", p);

    PetscCall(VecRestoreArrayRead(rho_seq, &r));
    PetscCall(VecRestoreArrayRead(filt_seq, &f));
    PetscCall(VecRestoreArrayRead(proj_seq, &p));
    std::fclose(fp);
  }

  PetscCall(VecScatterDestroy(&s1));
  PetscCall(VecScatterDestroy(&s2));
  PetscCall(VecScatterDestroy(&s3));
  PetscCall(VecDestroy(&rho_seq));
  PetscCall(VecDestroy(&filt_seq));
  PetscCall(VecDestroy(&proj_seq));
  PetscCall(VecDestroy(&rho_nat));
  PetscCall(VecDestroy(&filt_nat));
  PetscCall(VecDestroy(&proj_nat));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Wrote density pipeline VTK: %s\n", path));
  return 0;
}

PetscErrorCode write_density_report(const char *output_prefix, const CellDM &cell_dm,
                                    const DensityPipelineOptions &opts,
                                    const std::vector<DraftDirection> &dirs,
                                    DraftCombine combine,
                                    PetscReal avg_initial,
                                    PetscReal avg_filtered,
                                    PetscReal avg_projected) {
  PetscViewer viewer = nullptr;
  char path[PETSC_MAX_PATH_LEN];
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s_density_summary.txt",
                          output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, path, &viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "mode=density\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "elements_x=%lld\n",
                                   static_cast<long long>(cell_dm.ex)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "elements_y=%lld\n",
                                   static_cast<long long>(cell_dm.ey)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "elements_z=%lld\n",
                                   static_cast<long long>(cell_dm.ez)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "filter_radius=%.12e\n",
                                   static_cast<double>(opts.filter_radius)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_radius=%lld\n",
                                   static_cast<long long>(opts.draft_radius)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_pnorm=%.12e\n",
                                   static_cast<double>(opts.draft_pnorm)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_beta=%.12e\n",
                                   static_cast<double>(opts.draft_beta)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_eta=%.12e\n",
                                   static_cast<double>(opts.draft_eta)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_dirs="));
  for (std::size_t i = 0; i < dirs.size(); ++i) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s%s", i ? "," : "",
                                     direction_name(dirs[i])));
  }
  if (dirs.empty()) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "none"));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "draft_combine=%s\n",
                                   combine_name(combine)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "avg_initial=%.12e\n",
                                   static_cast<double>(avg_initial)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "avg_filtered=%.12e\n",
                                   static_cast<double>(avg_filtered)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "avg_projected=%.12e\n",
                                   static_cast<double>(avg_projected)));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Density report: %s, avg %.6g -> %.6g -> %.6g\n",
                        path, static_cast<double>(avg_initial),
                        static_cast<double>(avg_filtered),
                        static_cast<double>(avg_projected)));
  return 0;
}

} // namespace

PetscErrorCode run_density_pipeline(const Grid &grid,
                                    const DensityOptions &density_options,
                                    const DensityPipelineOptions &pipeline_options,
                                    const char *output_prefix,
                                    PetscBool write_vtk,
                                    const char *vtk_file,
                                    PetscBool write_binary,
                                    const char *binary_file) {
  CellDM cell_dm;
  Vec rho = nullptr;
  Vec filtered = nullptr;
  Vec projected = nullptr;
  PetscReal avg_initial = 0.0;
  PetscReal avg_filtered = 0.0;
  PetscReal avg_projected = 0.0;
  std::vector<DraftDirection> dirs;
  DraftCombine combine = DraftCombine::Maximum;
  const PetscInt stencil_width =
      PetscMax(static_cast<PetscInt>(PetscCeilReal(pipeline_options.filter_radius)),
               pipeline_options.draft_radius);

  PetscCheck(pipeline_options.draft_pnorm >= 1.0, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_OUTOFRANGE, "-draft_pnorm must be >= 1");
  PetscCheck(stencil_width >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "stencil width must be non-negative");

  PetscCall(parse_draft_dirs(pipeline_options.draft_dirs, &dirs));
  PetscCall(parse_draft_combine(pipeline_options.draft_combine, &combine));
  PetscCall(create_cell_dm(grid, stencil_width, &cell_dm));
  PetscCall(create_initial_density(cell_dm, grid, density_options, &rho));
  PetscCall(VecDuplicate(rho, &filtered));
  PetscCall(VecDuplicate(rho, &projected));

  PetscCall(apply_cone_filter(cell_dm, rho, filtered,
                              pipeline_options.filter_radius));
  PetscCall(apply_draft_projection(cell_dm, filtered, projected,
                                   pipeline_options, dirs));
  PetscCall(vec_average(rho, &avg_initial));
  PetscCall(vec_average(filtered, &avg_filtered));
  PetscCall(vec_average(projected, &avg_projected));

  if (write_vtk) {
    PetscCall(write_density_pipeline_vtk(cell_dm, vtk_file, rho, filtered, projected));
  }
  if (write_binary) {
    PetscViewer viewer = nullptr;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, binary_file,
                                    FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(projected, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Wrote projected density PETSc Vec: %s\n",
                          binary_file));
  }

  PetscCall(write_density_report(output_prefix, cell_dm, pipeline_options, dirs,
                                 combine,
                                 avg_initial, avg_filtered, avg_projected));
  PetscCall(VecDestroy(&rho));
  PetscCall(VecDestroy(&filtered));
  PetscCall(VecDestroy(&projected));
  PetscCall(DMDestroy(&cell_dm.da));
  return 0;
}

} // namespace control_arm
