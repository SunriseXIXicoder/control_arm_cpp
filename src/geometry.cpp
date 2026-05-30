#include "control_arm/geometry.hpp"

#include <petscmath.h>

namespace control_arm {
namespace {

PetscReal clamp_real(PetscReal value, PetscReal lo, PetscReal hi) {
  return PetscMax(lo, PetscMin(hi, value));
}

PetscReal segment_distance_squared(PetscReal px, PetscReal py, PetscReal pz,
                                   PetscReal ax, PetscReal ay, PetscReal az,
                                   PetscReal bx, PetscReal by, PetscReal bz) {
  const PetscReal vx = bx - ax;
  const PetscReal vy = by - ay;
  const PetscReal vz = bz - az;
  const PetscReal wx = px - ax;
  const PetscReal wy = py - ay;
  const PetscReal wz = pz - az;
  const PetscReal den = vx * vx + vy * vy + vz * vz;
  PetscReal t = den > 0.0 ? (wx * vx + wy * vy + wz * vz) / den : 0.0;
  t = clamp_real(t, 0.0, 1.0);

  const PetscReal dx = px - (ax + t * vx);
  const PetscReal dy = py - (ay + t * vy);
  const PetscReal dz = pz - (az + t * vz);
  return dx * dx + dy * dy + dz * dz;
}

bool in_triangle_2d(PetscReal px, PetscReal py,
                    PetscReal ax, PetscReal ay,
                    PetscReal bx, PetscReal by,
                    PetscReal cx, PetscReal cy) {
  const PetscReal d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  const PetscReal d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  const PetscReal d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  const bool has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
  const bool has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
  return !(has_neg && has_pos);
}

} // namespace

PetscReal domain_length(const Grid &grid) {
  return static_cast<PetscReal>(grid.nx - 1) /
         static_cast<PetscReal>(grid.nz - 1) * domain_height(grid);
}

PetscReal domain_width(const Grid &grid) {
  return static_cast<PetscReal>(grid.ny - 1) /
         static_cast<PetscReal>(grid.nz - 1) * domain_height(grid);
}

PetscReal domain_height(const Grid &grid) {
  return grid.physical_height;
}

PetscReal density_at_normalized(PetscReal x, PetscReal y, PetscReal z,
                                const Grid &grid,
                                const DensityOptions &options) {
  if (!options.use_control_arm_mask) {
    return 1.0;
  }

  const PetscReal DL = domain_length(grid);
  const PetscReal DW = domain_width(grid);
  const PetscReal DH = domain_height(grid);
  const PetscReal X = x * DL;
  const PetscReal Y = y * DW;
  const PetscReal Z = z * DH;
  const PetscReal min_ld = PetscMin(DL, DH);
  const PetscReal min_lw = PetscMin(DL, DW);

  const PetscReal A[3] = {0.18 * DL, DW, 0.50 * DH};
  const PetscReal B[3] = {0.18 * DL, 0.0, 0.50 * DH};
  const PetscReal C[3] = {0.78 * DL, 0.50 * DW, 0.50 * DH};
  const PetscReal spring_center[3] = {0.50 * DL, 0.50 * DW, 0.50 * DH};

  const PetscReal rA_hole = 0.25 * min_ld;
  const PetscReal rB_hole = 0.25 * min_ld;
  const PetscReal rC_hole = 0.10 * min_lw;
  const PetscReal rA_keep = 0.30 * min_ld;
  const PetscReal rB_keep = 0.30 * min_ld;
  const PetscReal rC_keep = 0.13 * min_lw;
  const PetscReal rA_pad = 0.35 * min_ld;
  const PetscReal rB_pad = 0.35 * min_ld;
  const PetscReal rC_pad = 0.15 * min_lw;
  const PetscReal rAB_void = 0.070 * PetscMin(DL, DW);

  const bool middle =
      (Z >= 0.25 * DH && Z <= 0.75 * DH);
  const PetscReal ab_triangle_retract =
      PetscMax(0.0, options.ab_triangle_retract) * min_ld;
  const bool triangle =
      middle && X >= A[0] + ab_triangle_retract &&
      in_triangle_2d(X, Y, A[0], A[1], B[0], B[1], C[0], C[1]);

  const bool axialA = (A[1] - Y <= rA_pad);
  const bool axialB = (Y - B[1] <= rB_pad);

  const bool holeA =
      ((X - A[0]) * (X - A[0]) + (Z - A[2]) * (Z - A[2]) <=
       rA_hole * rA_hole) &&
      axialA;
  const bool holeB =
      ((X - B[0]) * (X - B[0]) + (Z - B[2]) * (Z - B[2]) <=
       rB_hole * rB_hole) &&
      axialB;
  const bool keepA =
      ((X - A[0]) * (X - A[0]) + (Z - A[2]) * (Z - A[2]) <=
       rA_keep * rA_keep) &&
      axialA;
  const bool keepB =
      ((X - B[0]) * (X - B[0]) + (Z - B[2]) * (Z - B[2]) <=
       rB_keep * rB_keep) &&
      axialB;

  const bool holeC =
      ((X - C[0]) * (X - C[0]) + (Y - C[1]) * (Y - C[1]) <=
       rC_hole * rC_hole);
  const bool keepC =
      ((X - C[0]) * (X - C[0]) + (Y - C[1]) * (Y - C[1]) <=
       rC_keep * rC_keep);

  const bool padA =
      ((X - A[0]) * (X - A[0]) + (Z - A[2]) * (Z - A[2]) <=
       rA_pad * rA_pad) &&
      axialA;
  const bool padB =
      ((X - B[0]) * (X - B[0]) + (Z - B[2]) * (Z - B[2]) <=
       rB_pad * rB_pad) &&
      axialB;
  const bool padC =
      ((X - C[0]) * (X - C[0]) + (Y - C[1]) * (Y - C[1]) <=
       rC_pad * rC_pad) &&
      middle;
  const bool spring_mount =
      middle &&
      (PetscAbsReal(X - spring_center[0]) <= 0.105 * DL / 2.0) &&
      (PetscAbsReal(Y - spring_center[1]) <= 0.105 * DW / 2.0);

  const bool outer_mask = triangle || padA || padB || padC;
  bool ab_void = false;

  const PetscReal ABx = B[0] - A[0];
  const PetscReal ABy = B[1] - A[1];
  const PetscReal ABz = B[2] - A[2];
  const PetscReal ABlen = PetscSqrtReal(ABx * ABx + ABy * ABy + ABz * ABz);
  const PetscReal hx = DL / static_cast<PetscReal>(grid.nx);
  const PetscReal hy = DW / static_cast<PetscReal>(grid.ny);
  const PetscReal hz = DH / static_cast<PetscReal>(grid.nz);
  const PetscReal neck_buffer = 2.0 * PetscMin(PetscMin(hx, hy), hz);
  const PetscReal clearA = 1.05 * rA_pad + rAB_void + neck_buffer;
  const PetscReal clearB = 1.05 * rB_pad + rAB_void + neck_buffer;
  if (ABlen > clearA + clearB) {
    const PetscReal ux = ABx / ABlen;
    const PetscReal uy = ABy / ABlen;
    const PetscReal uz = ABz / ABlen;
    const PetscReal P1[3] = {A[0] + ux * clearA, A[1] + uy * clearA,
                             A[2] + uz * clearA};
    const PetscReal P2[3] = {B[0] - ux * clearB, B[1] - uy * clearB,
                             B[2] - uz * clearB};
    ab_void =
        (segment_distance_squared(X, Y, Z, P1[0], P1[1], P1[2], P2[0],
                                  P2[1], P2[2]) <= rAB_void * rAB_void) &&
        !(keepA || keepB || padA || padB);
  }

  if ((!outer_mask) || holeA || holeB || holeC || ab_void) {
    return options.void_density;
  }
  if (keepA || keepB || keepC || spring_mount) {
    return 1.0;
  }
  return 1.0;
}

PetscReal node_density(PetscInt i, PetscInt j, PetscInt k,
                       const Grid &grid,
                       const DensityOptions &options) {
  const PetscReal x =
      grid.nx > 1 ? static_cast<PetscReal>(i) / static_cast<PetscReal>(grid.nx - 1) : 0.0;
  const PetscReal y =
      grid.ny > 1 ? static_cast<PetscReal>(j) / static_cast<PetscReal>(grid.ny - 1) : 0.0;
  const PetscReal z =
      grid.nz > 1 ? static_cast<PetscReal>(k) / static_cast<PetscReal>(grid.nz - 1) : 0.5;
  return density_at_normalized(x, y, z, grid, options);
}

PetscReal cell_density(PetscInt i, PetscInt j, PetscInt k,
                       const Grid &grid,
                       const DensityOptions &options) {
  const PetscReal x =
      (static_cast<PetscReal>(i) + 0.5) / static_cast<PetscReal>(grid.nx - 1);
  const PetscReal y =
      (static_cast<PetscReal>(j) + 0.5) / static_cast<PetscReal>(grid.ny - 1);
  const PetscReal z =
      (static_cast<PetscReal>(k) + 0.5) / static_cast<PetscReal>(grid.nz - 1);
  return density_at_normalized(x, y, z, grid, options);
}

PetscReal edge_stiffness(PetscReal rho0, PetscReal rho1,
                         const DensityOptions &options) {
  const PetscReal rho = clamp_real(0.5 * (rho0 + rho1), 0.0, 1.0);
  return options.young_modulus *
         (options.emin + (1.0 - options.emin) *
                             PetscPowReal(rho, options.penal));
}

PetscInt node_count(const Grid &grid) {
  return grid.nx * grid.ny * grid.nz;
}

PetscInt dof_count(const Grid &grid) {
  return 3 * node_count(grid);
}

PetscInt cell_count(const Grid &grid) {
  return (grid.nx - 1) * (grid.ny - 1) * (grid.nz - 1);
}

PetscInt global_dof(PetscInt i, PetscInt j, PetscInt k, PetscInt c,
                    const Grid &grid) {
  return 3 * (i + grid.nx * (j + grid.ny * k)) + c;
}

PetscErrorCode compute_mask_volume_fraction(MPI_Comm comm, const Grid &grid,
                                            const DensityOptions &options,
                                            PetscReal *volume_fraction,
                                            PetscInt *solid_cells,
                                            PetscInt *total_cells) {
  PetscMPIInt rank = 0;
  PetscMPIInt size = 1;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  const PetscInt total = cell_count(grid);
  const PetscInt start = (total * rank) / size;
  const PetscInt end = (total * (rank + 1)) / size;
  PetscInt local_solid = 0;

  for (PetscInt cid = start; cid < end; ++cid) {
    const PetscInt plane = (grid.nx - 1) * (grid.ny - 1);
    const PetscInt k = cid / plane;
    const PetscInt rem = cid - k * plane;
    const PetscInt j = rem / (grid.nx - 1);
    const PetscInt i = rem - j * (grid.nx - 1);
    if (cell_density(i, j, k, grid, options) >= options.mask_threshold) {
      ++local_solid;
    }
  }

  PetscInt global_solid = 0;
  PetscCallMPI(MPI_Allreduce(&local_solid, &global_solid, 1, MPIU_INT, MPI_SUM, comm));
  if (volume_fraction) {
    *volume_fraction = total > 0 ? static_cast<PetscReal>(global_solid) /
                                       static_cast<PetscReal>(total)
                                 : 0.0;
  }
  if (solid_cells) {
    *solid_cells = global_solid;
  }
  if (total_cells) {
    *total_cells = total;
  }
  return 0;
}

} // namespace control_arm
