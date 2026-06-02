#include <petscksp.h>
#include <math.h>
#include <stdio.h>

static char help[] =
"Distributed PETSc control-arm-scale stiffness solve.\n"
"This program assembles a 3-D, 3-dof-per-node SPD stiffness benchmark with\n"
"Dirichlet constraints on the left face and a right-face load.  It is intended\n"
"as the PETSc/MPI kernel for 100M+ DOF runs before fully porting the topology\n"
"optimization loop.\n\n"
"Main options:\n"
"  -nx -ny -nz                         grid dimensions\n"
"  -operator low_order|h8_matrix_free  operator implementation\n"
"  -density_file <path>                reserved PETSc/HDF5 density input\n"
"  -output_prefix <path>               summary and memory report prefix\n"
"  -write_solution true|false          write PETSc binary solution vector\n"
"  -write_vtk true|false               write small legacy VTK output on rank 0\n"
"  -vtk_file <path>                    structured-grid VTK output path\n"
"  -write_mask_vtk true|false          write thresholded solid-mask VTK cells\n"
"  -mask_vtk_file <path>               solid-mask VTK output path\n\n";

static PetscInt GlobalDof(PetscInt i, PetscInt j, PetscInt k, PetscInt c,
                          PetscInt nx, PetscInt ny)
{
  return 3 * (i + nx * (j + ny * k)) + c;
}

static PetscReal ClampReal(PetscReal v, PetscReal lo, PetscReal hi)
{
  return PetscMax(lo, PetscMin(hi, v));
}

static PetscReal SegmentDistance3D2(PetscReal px, PetscReal py, PetscReal pz,
                                    PetscReal ax, PetscReal ay, PetscReal az,
                                    PetscReal bx, PetscReal by, PetscReal bz)
{
  PetscReal vx = bx - ax;
  PetscReal vy = by - ay;
  PetscReal vz = bz - az;
  PetscReal wx = px - ax;
  PetscReal wy = py - ay;
  PetscReal wz = pz - az;
  PetscReal den = vx * vx + vy * vy + vz * vz;
  PetscReal t = den > 0.0 ? (wx * vx + wy * vy + wz * vz) / den : 0.0;
  PetscReal dx, dy, dz;

  t = ClampReal(t, 0.0, 1.0);
  dx = px - (ax + t * vx);
  dy = py - (ay + t * vy);
  dz = pz - (az + t * vz);
  return dx * dx + dy * dy + dz * dz;
}

static PetscBool InTriangle2D(PetscReal px, PetscReal py,
                              PetscReal ax, PetscReal ay,
                              PetscReal bx, PetscReal by,
                              PetscReal cx, PetscReal cy)
{
  PetscReal d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  PetscReal d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  PetscReal d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  PetscBool hasNeg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
  PetscBool hasPos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
  return !(hasNeg && hasPos) ? PETSC_TRUE : PETSC_FALSE;
}

static PetscReal ControlArmDensityXYZ(PetscReal x, PetscReal y, PetscReal z,
                                      PetscInt nx, PetscInt ny, PetscInt nz,
                                      PetscReal voidDensity)
{
  PetscReal DL = ((PetscReal)nx) / ((PetscReal)nz);
  PetscReal DW = ((PetscReal)ny) / ((PetscReal)nz);
  PetscReal DH = 1.0;
  PetscReal X = x * DL;
  PetscReal Y = y * DW;
  PetscReal Z = z * DH;
  PetscReal minLD = PetscMin(DL, DH);
  PetscReal minLW = PetscMin(DL, DW);
  PetscReal A[3] = {0.18 * DL, DW, 0.50 * DH};
  PetscReal B[3] = {0.18 * DL, 0.0, 0.50 * DH};
  PetscReal C[3] = {0.78 * DL, 0.50 * DW, 0.50 * DH};
  PetscReal rA_hole = 0.25 * minLD;
  PetscReal rB_hole = 0.25 * minLD;
  PetscReal rC_hole = 0.10 * minLW;
  PetscReal rA_keep = 0.30 * minLD;
  PetscReal rB_keep = 0.30 * minLD;
  PetscReal rC_keep = 0.13 * minLW;
  PetscReal rA_pad = 0.35 * minLD;
  PetscReal rB_pad = 0.35 * minLD;
  PetscReal rC_pad = 0.15 * minLW;
  PetscReal rAB_void = 0.070 * PetscMin(DL, DW);
  PetscReal smC[3] = {0.50 * DL, 0.50 * DW, 0.50 * DH};
  PetscBool middle = (Z >= 0.25 * DH && Z <= 0.75 * DH) ? PETSC_TRUE : PETSC_FALSE;
  PetscBool triangle = middle && InTriangle2D(X, Y, A[0], A[1], B[0], B[1], C[0], C[1]);
  PetscBool axialA = (A[1] - Y <= rA_pad) ? PETSC_TRUE : PETSC_FALSE;
  PetscBool axialB = (Y - B[1] <= rB_pad) ? PETSC_TRUE : PETSC_FALSE;
  PetscBool holeA = ((X - A[0]) * (X - A[0]) + (Z - A[2]) * (Z - A[2]) <= rA_hole * rA_hole) && axialA;
  PetscBool holeB = ((X - B[0]) * (X - B[0]) + (Z - B[2]) * (Z - B[2]) <= rB_hole * rB_hole) && axialB;
  PetscBool keepA = ((X - A[0]) * (X - A[0]) + (Z - A[2]) * (Z - A[2]) <= rA_keep * rA_keep) && axialA;
  PetscBool keepB = ((X - B[0]) * (X - B[0]) + (Z - B[2]) * (Z - B[2]) <= rB_keep * rB_keep) && axialB;
  PetscBool holeC = ((X - C[0]) * (X - C[0]) + (Y - C[1]) * (Y - C[1]) <= rC_hole * rC_hole);
  PetscBool keepC = ((X - C[0]) * (X - C[0]) + (Y - C[1]) * (Y - C[1]) <= rC_keep * rC_keep);
  PetscBool padA = ((X - A[0]) * (X - A[0]) + (Z - A[2]) * (Z - A[2]) <= rA_pad * rA_pad) && axialA;
  PetscBool padB = ((X - B[0]) * (X - B[0]) + (Z - B[2]) * (Z - B[2]) <= rB_pad * rB_pad) && axialB;
  PetscBool padC = ((X - C[0]) * (X - C[0]) + (Y - C[1]) * (Y - C[1]) <= rC_pad * rC_pad) && middle;
  PetscBool springMount = middle &&
    (PetscAbsReal(X - smC[0]) <= 0.105 * DL / 2.0) &&
    (PetscAbsReal(Y - smC[1]) <= 0.105 * DW / 2.0);
  PetscBool outerMask = triangle || padA || padB || padC;
  PetscBool abVoid = PETSC_FALSE;

  {
    PetscReal ABx = B[0] - A[0];
    PetscReal ABy = B[1] - A[1];
    PetscReal ABz = B[2] - A[2];
    PetscReal ABlen = sqrt(ABx * ABx + ABy * ABy + ABz * ABz);
    PetscReal neckBuffer = 2.0 * PetscMin(PetscMin(DL / (PetscReal)nx, DW / (PetscReal)ny), DH / (PetscReal)nz);
    PetscReal clearA = 1.05 * rA_pad + rAB_void + neckBuffer;
    PetscReal clearB = 1.05 * rB_pad + rAB_void + neckBuffer;
    if (ABlen > clearA + clearB) {
      PetscReal ux = ABx / ABlen;
      PetscReal uy = ABy / ABlen;
      PetscReal uz = ABz / ABlen;
      PetscReal P1[3] = {A[0] + ux * clearA, A[1] + uy * clearA, A[2] + uz * clearA};
      PetscReal P2[3] = {B[0] - ux * clearB, B[1] - uy * clearB, B[2] - uz * clearB};
      abVoid = (SegmentDistance3D2(X, Y, Z, P1[0], P1[1], P1[2], P2[0], P2[1], P2[2]) <= rAB_void * rAB_void) &&
               !(keepA || keepB || padA || padB);
    }
  }

  if ((!outerMask) || holeA || holeB || holeC || abVoid) return voidDensity;
  if (keepA || keepB || keepC || springMount) return 1.0;
  return 1.0;
}

static PetscReal ControlArmMaskDensity(PetscInt i, PetscInt j, PetscInt k,
                                       PetscInt nx, PetscInt ny, PetscInt nz,
                                       PetscReal voidDensity)
{
  PetscReal x = nx > 1 ? ((PetscReal)i) / ((PetscReal)(nx - 1)) : 0.0;
  PetscReal y = ny > 1 ? ((PetscReal)j) / ((PetscReal)(ny - 1)) : 0.0;
  PetscReal z = nz > 1 ? ((PetscReal)k) / ((PetscReal)(nz - 1)) : 0.5;
  return ControlArmDensityXYZ(x, y, z, nx, ny, nz, voidDensity);
}

static PetscReal NodeDensity(PetscInt i, PetscInt j, PetscInt k,
                             PetscInt nx, PetscInt ny, PetscInt nz,
                             PetscBool useControlArmMask,
                             PetscReal voidDensity)
{
  if (!useControlArmMask) return 1.0;
  return ControlArmMaskDensity(i, j, k, nx, ny, nz, voidDensity);
}

static PetscReal EdgeStiffness(PetscReal rho0, PetscReal rho1,
                               PetscReal penal, PetscReal emin)
{
  PetscReal rho = ClampReal(0.5 * (rho0 + rho1), 0.0, 1.0);
  return emin + (1.0 - emin) * PetscPowReal(rho, penal);
}

static void CountPreallocColumn(PetscInt col, PetscInt cstart, PetscInt cend,
                                PetscInt *dnz, PetscInt *onz)
{
  if (col >= cstart && col < cend) {
    (*dnz)++;
  } else {
    (*onz)++;
  }
}

static PetscErrorCode WriteRunReport(const char *outputPrefix, PetscInt nx, PetscInt ny,
                                     PetscInt nz, PetscInt ndof, PetscMPIInt ranks,
                                     const char *operatorType, PetscLogDouble assemblyTime,
                                     PetscLogDouble solveTime, PetscInt its,
                                     PetscReal rnorm, PetscReal compliance)
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  char reportFile[PETSC_MAX_PATH_LEN];
  PetscLogDouble memCurrent = 0.0, memMax = 0.0, memCurrentMax = 0.0, memMaxMax = 0.0;

  ierr = PetscMemoryGetCurrentUsage(&memCurrent);CHKERRQ(ierr);
  ierr = PetscMemoryGetMaximumUsage(&memMax);CHKERRQ(ierr);
  ierr = MPI_Reduce(&memCurrent, &memCurrentMax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Reduce(&memMax, &memMaxMax, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscSNPrintf(reportFile, sizeof(reportFile), "%s_summary.txt", outputPrefix);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, reportFile, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "operator=%s\n", operatorType);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "nx=%lld\nny=%lld\nnz=%lld\ndof=%lld\nranks=%d\n",
                                (long long)nx, (long long)ny, (long long)nz,
                                (long long)ndof, (int)ranks);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "assembly_time_sec=%.12e\n", (double)assemblyTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "solve_time_sec=%.12e\n", (double)solveTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "ksp_iterations=%lld\n", (long long)its);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "ksp_residual=%.12e\n", (double)rnorm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "compliance=%.12e\n", (double)compliance);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "max_current_memory_bytes=%.0f\n", (double)memCurrentMax);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "max_peak_memory_bytes=%.0f\n", (double)memMaxMax);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Report: %s, max current memory %.3f GiB/rank, max peak memory %.3f GiB/rank\n",
                     reportFile,
                     (double)(memCurrentMax / 1073741824.0),
                     (double)(memMaxMax / 1073741824.0));CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode WriteLegacyVTK(Vec u, const char *vtkFile, PetscInt nx, PetscInt ny,
                                     PetscInt nz, PetscBool useControlArmMask,
                                     PetscReal voidDensity)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  VecScatter scatter;
  Vec uSeq;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = VecScatterCreateToZero(u, &scatter, &uSeq);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter, u, uSeq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter, u, uSeq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

  if (rank == 0) {
    const PetscScalar *ua;
    FILE *fp;
    PetscInt i, j, k;
    PetscInt nnode = nx * ny * nz;
    PetscReal hx = 1.0 / (PetscReal)(nx - 1);
    PetscReal hy = 1.0 / (PetscReal)(ny - 1);
    PetscReal hz = 1.0 / (PetscReal)(nz - 1);

    fp = fopen(vtkFile, "w");
    if (!fp) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open VTK file");
    }

    ierr = VecGetArrayRead(uSeq, &ua);CHKERRQ(ierr);
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "PETSc low_order control-arm-scale result\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    fprintf(fp, "DIMENSIONS %lld %lld %lld\n", (long long)nx, (long long)ny, (long long)nz);
    fprintf(fp, "ORIGIN 0 0 0\n");
    fprintf(fp, "SPACING %.17g %.17g %.17g\n", (double)hx, (double)hy, (double)hz);
    fprintf(fp, "POINT_DATA %lld\n", (long long)nnode);

    fprintf(fp, "VECTORS displacement double\n");
    for (k = 0; k < nz; k++) {
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          PetscInt node = i + nx * (j + ny * k);
          PetscReal ux = PetscRealPart(ua[3 * node + 0]);
          PetscReal uy = PetscRealPart(ua[3 * node + 1]);
          PetscReal uz = PetscRealPart(ua[3 * node + 2]);
          fprintf(fp, "%.17e %.17e %.17e\n", (double)ux, (double)uy, (double)uz);
        }
      }
    }

    fprintf(fp, "SCALARS displacement_magnitude double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (k = 0; k < nz; k++) {
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          PetscInt node = i + nx * (j + ny * k);
          PetscReal ux = PetscRealPart(ua[3 * node + 0]);
          PetscReal uy = PetscRealPart(ua[3 * node + 1]);
          PetscReal uz = PetscRealPart(ua[3 * node + 2]);
          PetscReal umag = sqrt(ux * ux + uy * uy + uz * uz);
          fprintf(fp, "%.17e\n", (double)umag);
        }
      }
    }

    fprintf(fp, "SCALARS density double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (k = 0; k < nz; k++) {
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          PetscReal rho = NodeDensity(i, j, k, nx, ny, nz, useControlArmMask, voidDensity);
          fprintf(fp, "%.17e\n", (double)rho);
        }
      }
    }

    ierr = VecRestoreArrayRead(uSeq, &ua);CHKERRQ(ierr);
    fclose(fp);
  }

  ierr = VecScatterDestroy(&scatter);CHKERRQ(ierr);
  ierr = VecDestroy(&uSeq);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Wrote legacy VTK: %s\n", vtkFile);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode WriteSolidMaskVTK(const char *vtkFile, PetscInt nx, PetscInt ny,
                                        PetscInt nz, PetscBool useControlArmMask,
                                        PetscReal voidDensity, PetscReal threshold,
                                        PetscInt maxCells)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  if (rank == 0) {
    FILE *fp;
    PetscInt i, j, k, count = 0, cellId = 0;
    PetscReal DL = ((PetscReal)nx) / ((PetscReal)nz);
    PetscReal DW = ((PetscReal)ny) / ((PetscReal)nz);
    PetscReal DH = 1.0;

    for (k = 0; k < nz - 1; k++) {
      for (j = 0; j < ny - 1; j++) {
        for (i = 0; i < nx - 1; i++) {
          PetscReal xc = ((PetscReal)i + 0.5) / ((PetscReal)(nx - 1));
          PetscReal yc = ((PetscReal)j + 0.5) / ((PetscReal)(ny - 1));
          PetscReal zc = ((PetscReal)k + 0.5) / ((PetscReal)(nz - 1));
          PetscReal rho = useControlArmMask ?
            ControlArmDensityXYZ(xc, yc, zc, nx, ny, nz, voidDensity) : 1.0;
          if (rho >= threshold) count++;
        }
      }
    }

    if (count > maxCells) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
              "Solid-mask VTK exceeds -mask_vtk_max_cells; raise the limit only for diagnostic runs");
    }

    fp = fopen(vtkFile, "w");
    if (!fp) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open solid-mask VTK file");
    }

    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "PETSc thresholded solid control-arm mask\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %lld double\n", (long long)(8 * count));

    for (k = 0; k < nz - 1; k++) {
      for (j = 0; j < ny - 1; j++) {
        for (i = 0; i < nx - 1; i++) {
          PetscReal xc = ((PetscReal)i + 0.5) / ((PetscReal)(nx - 1));
          PetscReal yc = ((PetscReal)j + 0.5) / ((PetscReal)(ny - 1));
          PetscReal zc = ((PetscReal)k + 0.5) / ((PetscReal)(nz - 1));
          PetscReal rho = useControlArmMask ?
            ControlArmDensityXYZ(xc, yc, zc, nx, ny, nz, voidDensity) : 1.0;
          if (rho >= threshold) {
            PetscReal x0 = DL * ((PetscReal)i) / ((PetscReal)(nx - 1));
            PetscReal x1 = DL * ((PetscReal)(i + 1)) / ((PetscReal)(nx - 1));
            PetscReal y0 = DW * ((PetscReal)j) / ((PetscReal)(ny - 1));
            PetscReal y1 = DW * ((PetscReal)(j + 1)) / ((PetscReal)(ny - 1));
            PetscReal z0 = DH * ((PetscReal)k) / ((PetscReal)(nz - 1));
            PetscReal z1 = DH * ((PetscReal)(k + 1)) / ((PetscReal)(nz - 1));
            fprintf(fp, "%.17e %.17e %.17e\n", (double)x0, (double)y0, (double)z0);
            fprintf(fp, "%.17e %.17e %.17e\n", (double)x1, (double)y0, (double)z0);
            fprintf(fp, "%.17e %.17e %.17e\n", (double)x1, (double)y1, (double)z0);
            fprintf(fp, "%.17e %.17e %.17e\n", (double)x0, (double)y1, (double)z0);
            fprintf(fp, "%.17e %.17e %.17e\n", (double)x0, (double)y0, (double)z1);
            fprintf(fp, "%.17e %.17e %.17e\n", (double)x1, (double)y0, (double)z1);
            fprintf(fp, "%.17e %.17e %.17e\n", (double)x1, (double)y1, (double)z1);
            fprintf(fp, "%.17e %.17e %.17e\n", (double)x0, (double)y1, (double)z1);
          }
        }
      }
    }

    fprintf(fp, "CELLS %lld %lld\n", (long long)count, (long long)(9 * count));
    for (cellId = 0; cellId < count; cellId++) {
      PetscInt base = 8 * cellId;
      fprintf(fp, "8 %lld %lld %lld %lld %lld %lld %lld %lld\n",
              (long long)(base + 0), (long long)(base + 1),
              (long long)(base + 2), (long long)(base + 3),
              (long long)(base + 4), (long long)(base + 5),
              (long long)(base + 6), (long long)(base + 7));
    }

    fprintf(fp, "CELL_TYPES %lld\n", (long long)count);
    for (cellId = 0; cellId < count; cellId++) {
      fprintf(fp, "12\n");
    }

    fprintf(fp, "CELL_DATA %lld\n", (long long)count);
    fprintf(fp, "SCALARS density double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (cellId = 0; cellId < count; cellId++) {
      fprintf(fp, "1.0\n");
    }

    fclose(fp);
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Wrote solid-mask VTK: %s\n", vtkFile);CHKERRQ(ierr);
  return 0;
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt size;
  PetscInt nx = 700, ny = 400, nz = 120;
  PetscInt nxy, nnode, ndof;
  PetscInt mLocal = PETSC_DECIDE;
  PetscInt rstart, rend, row;
  PetscInt *d_nnz = NULL, *o_nnz = NULL;
  PetscReal hx, hy, hz, kx, ky, kz;
  PetscReal load = 1.0;
  PetscReal penal = 3.0;
  PetscReal emin = 1.0e-6;
  PetscReal voidDensity = 0.02;
  PetscBool useControlArmMask = PETSC_FALSE;
  PetscBool writeSolution = PETSC_FALSE;
  PetscBool writeVtk = PETSC_FALSE;
  PetscBool writeMaskVtk = PETSC_FALSE;
  PetscBool hasVtkFile = PETSC_FALSE;
  PetscBool hasMaskVtkFile = PETSC_FALSE;
  PetscBool hasDensityFile = PETSC_FALSE;
  PetscBool isLowOrder = PETSC_FALSE;
  PetscBool isH8MatrixFree = PETSC_FALSE;
  char operatorType[64] = "low_order";
  char densityFile[PETSC_MAX_PATH_LEN] = "";
  char outputPrefix[PETSC_MAX_PATH_LEN] = "result/control_arm_petsc";
  char solutionFile[PETSC_MAX_PATH_LEN] = "result/solution.petscbin";
  char vtkFile[PETSC_MAX_PATH_LEN] = "";
  char maskVtkFile[PETSC_MAX_PATH_LEN] = "";
  PetscInt vtkMaxPoints = 5000000;
  PetscInt maskVtkMaxCells = 300000;
  PetscReal maskThreshold = 0.5;
  Mat A;
  Vec b, u;
  KSP ksp;
  PC pc;
  PetscLogDouble t0, t1, t2;
  PetscInt its;
  PetscReal rnorm;
  PetscScalar compliance;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-ny", &ny, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-nz", &nz, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-load", &load, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-penal", &penal, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-emin", &emin, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-void_density", &voidDensity, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-control_arm_mask", &useControlArmMask, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-operator", operatorType, sizeof(operatorType), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-density_file", densityFile, sizeof(densityFile), &hasDensityFile);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-output_prefix", outputPrefix, sizeof(outputPrefix), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-write_solution", &writeSolution, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-solution_file", solutionFile, sizeof(solutionFile), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-write_vtk", &writeVtk, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-vtk_file", vtkFile, sizeof(vtkFile), &hasVtkFile);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-vtk_max_points", &vtkMaxPoints, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-write_mask_vtk", &writeMaskVtk, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-mask_vtk_file", maskVtkFile, sizeof(maskVtkFile), &hasMaskVtkFile);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-mask_threshold", &maskThreshold, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-mask_vtk_max_cells", &maskVtkMaxCells, NULL);CHKERRQ(ierr);

  ierr = PetscStrcmp(operatorType, "low_order", &isLowOrder);CHKERRQ(ierr);
  ierr = PetscStrcmp(operatorType, "h8_matrix_free", &isH8MatrixFree);CHKERRQ(ierr);
  if (!isLowOrder && !isH8MatrixFree) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
            "-operator must be low_order or h8_matrix_free");
  }
  if (isH8MatrixFree) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP,
            "-operator h8_matrix_free is reserved for the next migration stage; current executable provides low_order");
  }
  if (hasDensityFile) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP,
            "-density_file is reserved for distributed rho/rho_phys input; current low_order run uses uniform or analytic mask density");
  }

  if (nx < 2 || ny < 2 || nz < 2) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "nx, ny, and nz must all be at least 2");
  }
  if (3.0 * (double)nx * (double)ny * (double)nz > (double)PETSC_MAX_INT) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
            "Requested problem exceeds this PETSc build's PetscInt range");
  }

  nxy = nx * ny;
  nnode = nxy * nz;
  ndof = 3 * nnode;
  if (writeVtk) {
    if (!hasVtkFile) {
      ierr = PetscSNPrintf(vtkFile, sizeof(vtkFile), "%s.vtk", outputPrefix);CHKERRQ(ierr);
    }
    if (nnode > vtkMaxPoints) {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
              "VTK output is intentionally limited by -vtk_max_points; increase it only for small diagnostic runs");
    }
  }
  if (writeMaskVtk && !hasMaskVtkFile) {
    ierr = PetscSNPrintf(maskVtkFile, sizeof(maskVtkFile), "%s_solid.vtk", outputPrefix);CHKERRQ(ierr);
  }
  hx = 1.0 / (PetscReal)(nx - 1);
  hy = 1.0 / (PetscReal)(ny - 1);
  hz = 1.0 / (PetscReal)(nz - 1);
  kx = 1.0 / (hx * hx);
  ky = 1.0 / (hy * hy);
  kz = 1.0 / (hz * hz);

  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "PETSc control-arm-scale solve: operator=%s nx=%lld ny=%lld nz=%lld, dof=%lld, ranks=%d\n",
                     operatorType, (long long)nx, (long long)ny, (long long)nz,
                     (long long)ndof, (int)size);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Material: penal=%g emin=%g control_arm_mask=%d void_density=%g\n",
                     (double)penal, (double)emin, (int)useControlArmMask, (double)voidDensity);CHKERRQ(ierr);

  ierr = PetscTime(&t0);CHKERRQ(ierr);
  ierr = PetscSplitOwnership(PETSC_COMM_WORLD, &mLocal, &ndof);CHKERRQ(ierr);
  {
    PetscInt scanEnd;
    ierr = MPI_Scan(&mLocal, &scanEnd, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
    rstart = scanEnd - mLocal;
    rend = scanEnd;
  }

  ierr = PetscMalloc2(mLocal, &d_nnz, mLocal, &o_nnz);CHKERRQ(ierr);
  for (row = rstart; row < rend; row++) {
    PetscInt local = row - rstart;
    PetscInt c = row % 3;
    PetscInt node = row / 3;
    PetscInt k = node / nxy;
    PetscInt rem = node - k * nxy;
    PetscInt j = rem / nx;
    PetscInt i = rem - j * nx;
    PetscInt q;
    PetscInt di[6] = {-1, 1, 0, 0, 0, 0};
    PetscInt dj[6] = {0, 0, -1, 1, 0, 0};
    PetscInt dk[6] = {0, 0, 0, 0, -1, 1};

    d_nnz[local] = 0;
    o_nnz[local] = 0;
    CountPreallocColumn(row, rstart, rend, &d_nnz[local], &o_nnz[local]);
    if (i == 0) continue;

    for (q = 0; q < 6; q++) {
      PetscInt ii = i + di[q];
      PetscInt jj = j + dj[q];
      PetscInt kk = k + dk[q];
      if (ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz && ii != 0) {
        PetscInt col = GlobalDof(ii, jj, kk, c, nx, ny);
        CountPreallocColumn(col, rstart, rend, &d_nnz[local], &o_nnz[local]);
      }
    }
  }

  ierr = MatCreateAIJ(PETSC_COMM_WORLD, mLocal, mLocal, ndof, ndof,
                      0, d_nnz, 0, o_nnz, &A);CHKERRQ(ierr);
  ierr = PetscFree2(d_nnz, o_nnz);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);

  for (row = rstart; row < rend; row++) {
    PetscInt c = row % 3;
    PetscInt node = row / 3;
    PetscInt k = node / nxy;
    PetscInt rem = node - k * nxy;
    PetscInt j = rem / nx;
    PetscInt i = rem - j * nx;
    PetscBool fixed = (i == 0) ? PETSC_TRUE : PETSC_FALSE;
    PetscInt cols[7];
    PetscScalar vals[7];
    PetscInt ncols = 0;

    if (fixed) {
      cols[0] = row;
      vals[0] = 1.0;
      ierr = MatSetValues(A, 1, &row, 1, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscReal rho0 = NodeDensity(i, j, k, nx, ny, nz, useControlArmMask, voidDensity);
      PetscReal diag = 0.0;
      PetscInt di[6] = {-1, 1, 0, 0, 0, 0};
      PetscInt dj[6] = {0, 0, -1, 1, 0, 0};
      PetscInt dk[6] = {0, 0, 0, 0, -1, 1};
      PetscReal kw[6] = {kx, kx, ky, ky, kz, kz};
      PetscInt q;

      for (q = 0; q < 6; q++) {
        PetscInt ii = i + di[q];
        PetscInt jj = j + dj[q];
        PetscInt kk = k + dk[q];
        if (ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz) {
          PetscReal rho1 = NodeDensity(ii, jj, kk, nx, ny, nz, useControlArmMask, voidDensity);
          PetscReal kij = kw[q] * EdgeStiffness(rho0, rho1, penal, emin);
          diag += kij;
          if (ii != 0) {
            cols[ncols] = GlobalDof(ii, jj, kk, c, nx, ny);
            vals[ncols] = -kij;
            ncols++;
          }
        }
      }

      cols[ncols] = row;
      vals[ncols] = diag;
      ncols++;
      ierr = MatSetValues(A, 1, &row, ncols, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscTime(&t1);CHKERRQ(ierr);

  ierr = MatCreateVecs(A, &u, &b);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = VecSet(b, 0.0);CHKERRQ(ierr);

  for (row = rstart; row < rend; row++) {
    PetscInt c = row % 3;
    PetscInt node = row / 3;
    PetscInt k = node / nxy;
    PetscInt rem = node - k * nxy;
    PetscInt j = rem / nx;
    PetscInt i = rem - j * nx;

    if (i == nx - 1 && c == 2) {
      PetscScalar val = -load / (PetscReal)(ny * nz);
      ierr = VecSetValue(b, row, val, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPCG);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCGAMG);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp, 1.0e-6, PETSC_DEFAULT, PETSC_DEFAULT, 500);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Assembly time: %.6e s\n", (double)(t1 - t0));CHKERRQ(ierr);
  ierr = PetscTime(&t1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, u);CHKERRQ(ierr);
  ierr = PetscTime(&t2);CHKERRQ(ierr);

  ierr = KSPGetIterationNumber(ksp, &its);CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(ksp, &rnorm);CHKERRQ(ierr);
  ierr = VecDot(b, u, &compliance);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Solve time: %.6e s, iterations=%lld, residual=%g, compliance=%g\n",
                     (double)(t2 - t1), (long long)its,
                     (double)rnorm, (double)PetscRealPart(compliance));CHKERRQ(ierr);
  ierr = WriteRunReport(outputPrefix, nx, ny, nz, ndof, size, operatorType,
                        t1 - t0, t2 - t1, its, rnorm,
                        PetscRealPart(compliance));CHKERRQ(ierr);

  if (writeSolution) {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, solutionFile, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
    ierr = VecView(u, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Wrote PETSc binary solution: %s\n", solutionFile);CHKERRQ(ierr);
  }

  if (writeVtk) {
    ierr = WriteLegacyVTK(u, vtkFile, nx, ny, nz, useControlArmMask, voidDensity);CHKERRQ(ierr);
  }

  if (writeMaskVtk) {
    ierr = WriteSolidMaskVTK(maskVtkFile, nx, ny, nz, useControlArmMask,
                             voidDensity, maskThreshold, maskVtkMaxCells);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
