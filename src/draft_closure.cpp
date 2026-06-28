#include "control_arm/draft_closure.hpp"

#include <petscmath.h>

#include <algorithm>
#include <string>

namespace control_arm {
namespace {

char lowercase_ascii(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool mask_is_active(PetscReal value) { return value > 0.5; }

PetscBool axis_from_char(char c, PetscInt *axis) {
  c = lowercase_ascii(c);
  if (c == 'x') {
    *axis = 0;
    return PETSC_TRUE;
  }
  if (c == 'y') {
    *axis = 1;
    return PETSC_TRUE;
  }
  if (c == 'z') {
    *axis = 2;
    return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

std::string compact_lower_token(const char *begin, const char *end) {
  std::string token;
  for (const char *p = begin; p < end; ++p) {
    const char c = lowercase_ascii(*p);
    if (c == '_' || c == '-') continue;
    token.push_back(c);
  }
  return token;
}

PetscErrorCode parse_draft_axis_token(const char *begin, const char *end,
                                      std::vector<DraftDirection> *dirs) {
  while (begin < end && (*begin == ' ' || *begin == '\t')) ++begin;
  while (end > begin && (*(end - 1) == ' ' || *(end - 1) == '\t')) --end;
  if (begin == end) return 0;

  std::string token;
  for (const char *p = begin; p < end; ++p) {
    token.push_back(lowercase_ascii(*p));
  }
  if (token == "none") return 0;

  PetscInt axis = -1;
  PetscInt sign = 0;
  PetscBool ok = PETSC_FALSE;
  if (token.size() == 1) {
    ok = axis_from_char(token[0], &axis);
  } else if (token.size() == 2 && (token[0] == '+' || token[0] == '-')) {
    ok = axis_from_char(token[1], &axis);
    sign = (token[0] == '+') ? 1 : -1;
  } else if (token.size() == 2 &&
             (token[0] == 'p' || token[0] == 'm' || token[0] == 'n')) {
    ok = axis_from_char(token[1], &axis);
    sign = (token[0] == 'p') ? 1 : -1;
  } else {
    const std::string compact = compact_lower_token(begin, end);
    const std::size_t len = compact.size();
    if (len >= 4 && compact.compare(0, 3, "pos") == 0) {
      ok = axis_from_char(compact[len - 1], &axis);
      sign = 1;
    } else if (len >= 5 && compact.compare(0, 4, "plus") == 0) {
      ok = axis_from_char(compact[len - 1], &axis);
      sign = 1;
    } else if (len >= 4 && compact.compare(0, 3, "neg") == 0) {
      ok = axis_from_char(compact[len - 1], &axis);
      sign = -1;
    } else if (len >= 6 && compact.compare(0, 5, "minus") == 0) {
      ok = axis_from_char(compact[len - 1], &axis);
      sign = -1;
    }
  }

  PetscCheck(ok && axis >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "%s", draft_axes_error_message());
  DraftDirection parsed;
  parsed.axis = axis;
  parsed.sign = sign;
  dirs->push_back(parsed);
  return 0;
}

PetscInt draft_axis_length(PetscInt axis, PetscInt nx, PetscInt ny,
                           PetscInt nz) {
  if (axis == 0) return nx;
  if (axis == 1) return ny;
  return nz;
}

PetscInt draft_column_count(PetscInt axis, PetscInt nx, PetscInt ny,
                            PetscInt nz) {
  if (axis == 0) return ny * nz;
  if (axis == 1) return nx * nz;
  return nx * ny;
}

PetscInt draft_axis_coordinate(PetscInt axis, PetscInt i, PetscInt j,
                               PetscInt k) {
  if (axis == 0) return i;
  if (axis == 1) return j;
  return k;
}

PetscInt draft_column_id(PetscInt axis, PetscInt i, PetscInt j, PetscInt k,
                         PetscInt nx, PetscInt ny, PetscInt nz) {
  (void)nz;
  if (axis == 0) return k * ny + j;
  if (axis == 1) return k * nx + i;
  return j * nx + i;
}

} // namespace

const char *draft_axes_error_message() {
  return "-opt_draft_axes must use +x,-x,+y,-y,+z,-z, "
         "px,mx,py,my,pz,mz, x, y, z, or none";
}

PetscErrorCode parse_draft_axes(const char *text,
                                std::vector<DraftDirection> *dirs) {
  dirs->clear();
  PetscCheck(text != nullptr && text[0] != '\0', PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONG, "%s", draft_axes_error_message());

  const char *token_begin = text;
  for (const char *p = text;; ++p) {
    const char c = *p;
    if (c == '\0' || c == ',' || c == ';' || c == '/' || c == ' ' ||
        c == '\t') {
      PetscCall(parse_draft_axis_token(token_begin, p, dirs));
      if (c == '\0') break;
      token_begin = p + 1;
    }
  }
  return 0;
}

PetscErrorCode collect_effective_draft_axes(
    const OptimizerOptions &options, std::vector<DraftDirection> *effective) {
  std::vector<DraftDirection> dirs;
  bool processed_axis[3] = {false, false, false};

  effective->clear();
  if (!options.z_draft_closure) return 0;
  PetscCall(parse_draft_axes(options.draft_axes, &dirs));
  if (dirs.empty()) return 0;

  for (const DraftDirection &dir : dirs) {
    const PetscInt axis = dir.axis;
    if (processed_axis[axis]) continue;

    bool has_plus = false;
    bool has_minus = false;
    bool has_split = false;
    for (const DraftDirection &same_axis : dirs) {
      if (same_axis.axis != axis) continue;
      has_plus = has_plus || same_axis.sign > 0;
      has_minus = has_minus || same_axis.sign < 0;
      has_split = has_split || same_axis.sign == 0;
    }

    PetscInt effective_sign = 0;
    if (!has_split && !(has_plus && has_minus)) {
      effective_sign = has_plus ? 1 : -1;
    }

    DraftDirection effective_dir;
    effective_dir.axis = axis;
    effective_dir.sign = effective_sign;
    effective->push_back(effective_dir);
    processed_axis[axis] = true;
  }
  return 0;
}

PetscErrorCode apply_axis_draft_closure(DM da, Vec mask, PetscReal eta,
                                        PetscInt axis, PetscInt sign, Vec rho) {
  PetscScalar ***r = nullptr;
  PetscScalar ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt nx = 0, ny = 0, nz = 0;
  PetscMPIInt reduce_count = 0;

  PetscCall(DMDAGetInfo(da, nullptr, &nx, &ny, &nz, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  const PetscInt axis_length = draft_axis_length(axis, nx, ny, nz);
  const PetscInt column_count = draft_column_count(axis, nx, ny, nz);
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

  PetscCall(DMDAVecGetArrayRead(da, rho, &r));
  PetscCall(DMDAVecGetArrayRead(da, mask, &m));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        const PetscReal rho_value = PetscRealPart(r[k][j][i]);
        if (!mask_is_active(mask_value) || rho_value < threshold) continue;
        const PetscInt id = draft_column_id(axis, i, j, k, nx, ny, nz);
        const PetscInt coord = draft_axis_coordinate(axis, i, j, k);
        local_min[static_cast<std::size_t>(id)] =
            PetscMin(local_min[static_cast<std::size_t>(id)], coord);
        local_max[static_cast<std::size_t>(id)] =
            PetscMax(local_max[static_cast<std::size_t>(id)], coord);
        local_peak[static_cast<std::size_t>(id)] =
            PetscMax(local_peak[static_cast<std::size_t>(id)], rho_value);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(da, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(da, mask, &m));

  PetscCallMPI(MPI_Allreduce(local_min.data(), global_min.data(), reduce_count,
                             MPIU_INT, MPI_MIN, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_max.data(), global_max.data(), reduce_count,
                             MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(local_peak.data(), global_peak.data(),
                             reduce_count, MPIU_REAL, MPI_MAX,
                             PETSC_COMM_WORLD));

  PetscCall(DMDAVecGetArray(da, rho, &r));
  PetscCall(DMDAVecGetArrayRead(da, mask, &m));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (!mask_is_active(mask_value)) continue;
        const PetscInt id = draft_column_id(axis, i, j, k, nx, ny, nz);
        const PetscInt coord = draft_axis_coordinate(axis, i, j, k);
        const PetscInt axis_min = global_min[static_cast<std::size_t>(id)];
        const PetscInt axis_max = global_max[static_cast<std::size_t>(id)];
        const bool fill_split = (axis_min <= coord && coord <= axis_max);
        const bool fill_plus = (axis_min < axis_length && coord >= axis_min);
        const bool fill_minus = (axis_max >= 0 && coord <= axis_max);
        const bool should_fill =
            sign > 0 ? fill_plus : (sign < 0 ? fill_minus : fill_split);
        if (should_fill) {
          r[k][j][i] = PetscMax(PetscRealPart(r[k][j][i]),
                                global_peak[static_cast<std::size_t>(id)]);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(da, mask, &m));
  return 0;
}

PetscErrorCode apply_draft_closure(DM da, Vec mask,
                                   const OptimizerOptions &options, Vec rho) {
  std::vector<DraftDirection> dirs;
  PetscCall(collect_effective_draft_axes(options, &dirs));
  for (const DraftDirection &dir : dirs) {
    PetscCall(apply_axis_draft_closure(da, mask, options.z_draft_eta, dir.axis,
                                       dir.sign, rho));
  }
  return 0;
}

} // namespace control_arm
