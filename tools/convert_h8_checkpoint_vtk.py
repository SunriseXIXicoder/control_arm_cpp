#!/usr/bin/env python3
"""Convert H8 PETSc binary density checkpoints to legacy VTK.

The optimizer checkpoints are PETSc DMDA Vec binary files.  PETSc writes these
vectors in DMDA natural ordering, so the file order is x-fastest, then y, then z
over the element grid.  This is independent of the MPI process grid that wrote
the checkpoint.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


PETSC_VEC_CLASSID = 1211214


def read_petsc_vec(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError(f"{path} is too small to be a PETSc Vec binary file")

    class32 = int.from_bytes(data[0:4], "big")
    if class32 == PETSC_VEC_CLASSID:
        n = int.from_bytes(data[4:8], "big")
        offset = 8
    else:
        class64 = int.from_bytes(data[0:8], "big")
        if class64 != PETSC_VEC_CLASSID:
            raise ValueError(f"{path} does not start with a PETSc Vec header")
        n = int.from_bytes(data[8:16], "big")
        offset = 16

    expected = offset + 8 * n
    if len(data) < expected:
        raise ValueError(
            f"{path} is truncated: has {len(data)} bytes, expected {expected}"
        )
    return list(struct.unpack(f">{n}d", data[offset:expected]))


def write_vtk(
    out: Path,
    rho: list[float],
    mask: list[float] | None,
    nx: int,
    ny: int,
    nz: int,
    height: float,
    stride: int,
) -> None:
    ex, ey, ez = nx - 1, ny - 1, nz - 1
    ncx = max(1, math.ceil(ex / stride))
    ncy = max(1, math.ceil(ey / stride))
    ncz = max(1, math.ceil(ez / stride))
    cell_count = ncx * ncy * ncz
    length = height * (nx - 1) / max(1, nz - 1)
    width = height * (ny - 1) / max(1, nz - 1)

    values: list[tuple[float, float]] = []
    for cz in range(ncz):
        k = min(ez - 1, cz * stride + stride // 2)
        for cy in range(ncy):
            j = min(ey - 1, cy * stride + stride // 2)
            for cx in range(ncx):
                i = min(ex - 1, cx * stride + stride // 2)
                idx = (k * ey + j) * ex + i
                m = 1.0 if mask is None else mask[idx]
                values.append((rho[idx], m))

    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("H8 PETSc checkpoint density downsample with source-order remap\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {ncx + 1} {ncy + 1} {ncz + 1}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {length / ncx:.17g} {width / ncy:.17g} {height / ncz:.17g}\n")
        f.write(f"CELL_DATA {cell_count}\n")

        f.write("SCALARS rho double 1\nLOOKUP_TABLE default\n")
        for r, _ in values:
            f.write(f"{max(0.0, r):.12e}\n")

        f.write("SCALARS rho_masked double 1\nLOOKUP_TABLE default\n")
        for r, m in values:
            f.write(f"{r if m > 0.5 else 0.0:.12e}\n")

        f.write("SCALARS design_mask double 1\nLOOKUP_TABLE default\n")
        for _, m in values:
            f.write(f"{max(0.0, m):.12e}\n")

        f.write("SCALARS fixed_x0_face double 1\nLOOKUP_TABLE default\n")
        for _cz in range(ncz):
            for _cy in range(ncy):
                for cx in range(ncx):
                    f.write("1.000000000000e+00\n" if cx == 0 else "0.000000000000e+00\n")

        f.write("SCALARS load_xmax_face double 1\nLOOKUP_TABLE default\n")
        for _cz in range(ncz):
            for _cy in range(ncy):
                for cx in range(ncx):
                    f.write(
                        "1.000000000000e+00\n"
                        if cx == ncx - 1
                        else "0.000000000000e+00\n"
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", required=True, type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nx", required=True, type=int)
    parser.add_argument("--ny", required=True, type=int)
    parser.add_argument("--nz", required=True, type=int)
    parser.add_argument("--height", default=0.08, type=float)
    parser.add_argument("--stride", default=2, type=int)
    parser.add_argument("--source-px", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--source-py", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--source-pz", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    rho = read_petsc_vec(args.density)
    mask = read_petsc_vec(args.mask) if args.mask else None
    write_vtk(args.out, rho, mask, args.nx, args.ny, args.nz, args.height, args.stride)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
