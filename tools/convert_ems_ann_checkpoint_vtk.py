#!/usr/bin/env python3
"""Convert EMsFEM ANN PETSc binary density checkpoints to legacy VTK.

The ANN optimizer writes fine-density PETSc Vec checkpoints in DMDA natural
ordering: x-fastest, then y, then z over the fine cell grid.  This converter
does not need PETSc.  It samples rows directly from the binary file, so it can
handle the 10M-DOF sub_n=5 checkpoint without loading all fine cells at once.
"""

from __future__ import annotations

import argparse
from array import array
import math
import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO


PETSC_VEC_CLASSID = 1211214


def petsc_vec_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(16)
    if len(header) < 16:
        raise ValueError(f"{path} is too small to be a PETSc Vec binary file")

    class32 = int.from_bytes(header[0:4], "big")
    if class32 == PETSC_VEC_CLASSID:
        return int.from_bytes(header[4:8], "big"), 8

    class64 = int.from_bytes(header[0:8], "big")
    if class64 == PETSC_VEC_CLASSID:
        return int.from_bytes(header[8:16], "big"), 16

    raise ValueError(f"{path} does not start with a PETSc Vec header")


def check_vec(path: Path, expected_entries: int) -> tuple[int, int]:
    entries, offset = petsc_vec_header(path)
    if entries != expected_entries:
        raise ValueError(
            f"{path} has {entries} entries, expected {expected_entries}"
        )
    expected_bytes = offset + entries * 8
    actual_bytes = path.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"{path} is truncated: has {actual_bytes} bytes, "
            f"expected at least {expected_bytes}"
        )
    return entries, offset


def read_row(f: BinaryIO, data_offset: int, fnx: int, fny: int, j: int, k: int) -> tuple[float, ...]:
    idx = (k * fny + j) * fnx
    f.seek(data_offset + idx * 8)
    raw = f.read(fnx * 8)
    if len(raw) != fnx * 8:
        raise EOFError("unexpected end of PETSc Vec while reading a fine-density row")
    return struct.unpack(f">{fnx}d", raw)


def read_petsc_vec_array(path: Path, expected_entries: int) -> array:
    _, offset = check_vec(path, expected_entries)
    data = array("d")
    with path.open("rb") as f:
        f.seek(offset)
        data.fromfile(f, expected_entries)
    if sys.byteorder == "little":
        data.byteswap()
    if len(data) != expected_entries:
        raise EOFError(
            f"unexpected end of PETSc Vec while reading {path}: "
            f"got {len(data)} entries, expected {expected_entries}"
        )
    return data


def sampled_indices(count: int, stride: int) -> list[int]:
    bins = max(1, math.ceil(count / stride))
    return [min(count - 1, c * stride + stride // 2) for c in range(bins)]


def coarse_position(point_index: int, sampled_points: int, coarse_nodes: int) -> tuple[int, float]:
    if coarse_nodes <= 1:
        return 0, 0.0
    if sampled_points <= 1:
        return 0, 0.0
    pos = (coarse_nodes - 1) * point_index / (sampled_points - 1)
    base = int(math.floor(pos))
    if base >= coarse_nodes - 1:
        return coarse_nodes - 2, 1.0
    return base, pos - base


def write_displacement_vectors(
    out,
    displacement_path: Path,
    nx: int,
    ny: int,
    nz: int,
    ncx: int,
    ncy: int,
    ncz: int,
) -> None:
    u = read_petsc_vec_array(displacement_path, nx * ny * nz * 3)

    def value(i: int, j: int, k: int, c: int) -> float:
        return u[3 * (i + nx * (j + ny * k)) + c]

    out.write(f"POINT_DATA {(ncx + 1) * (ncy + 1) * (ncz + 1)}\n")
    out.write("VECTORS displacement double\n")
    for pk in range(ncz + 1):
        ck, tz = coarse_position(pk, ncz + 1, nz)
        wz = (1.0 - tz, tz)
        for pj in range(ncy + 1):
            cj, ty = coarse_position(pj, ncy + 1, ny)
            wy = (1.0 - ty, ty)
            for pi in range(ncx + 1):
                ci, tx = coarse_position(pi, ncx + 1, nx)
                wx = (1.0 - tx, tx)
                disp = [0.0, 0.0, 0.0]
                for dz in (0, 1):
                    for dy in (0, 1):
                        for dx in (0, 1):
                            w = wx[dx] * wy[dy] * wz[dz]
                            for c in (0, 1, 2):
                                disp[c] += w * value(ci + dx, cj + dy, ck + dz, c)
                out.write(f"{disp[0]:.12e} {disp[1]:.12e} {disp[2]:.12e}\n")


def write_scalar(
    out,
    name: str,
    rho_path: Path,
    rho_offset: int,
    mask_path: Path | None,
    mask_offset: int | None,
    fnx: int,
    fny: int,
    xs: list[int],
    ys: list[int],
    zs: list[int],
    mode: str,
) -> None:
    out.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
    with rho_path.open("rb") as rho_file:
        mask_file = mask_path.open("rb") if mask_path is not None else None
        try:
            for k in zs:
                for j in ys:
                    rho_row = read_row(rho_file, rho_offset, fnx, fny, j, k)
                    mask_row = (
                        read_row(mask_file, mask_offset, fnx, fny, j, k)
                        if mask_file is not None and mask_offset is not None
                        else None
                    )
                    for i in xs:
                        rho = max(0.0, rho_row[i])
                        mask = 1.0 if mask_row is None else max(0.0, mask_row[i])
                        if mode == "rho":
                            value = rho
                        elif mode == "rho_plot":
                            value = rho if mask > 0.5 else 0.0
                        elif mode == "mask":
                            value = mask
                        else:
                            raise ValueError(f"unknown scalar mode {mode}")
                        out.write(f"{value:.12e}\n")
        finally:
            if mask_file is not None:
                mask_file.close()


def write_vtk(
    rho_path: Path,
    mask_path: Path | None,
    displacement_path: Path | None,
    out_path: Path,
    nx: int,
    ny: int,
    nz: int,
    sub_n: int,
    height: float,
    stride: int,
    max_cells: int,
) -> None:
    fnx = (nx - 1) * sub_n
    fny = (ny - 1) * sub_n
    fnz = (nz - 1) * sub_n
    fine_cells = fnx * fny * fnz
    _, rho_offset = check_vec(rho_path, fine_cells)
    mask_offset = None
    if mask_path is not None:
        _, mask_offset = check_vec(mask_path, fine_cells)

    xs = sampled_indices(fnx, stride)
    ys = sampled_indices(fny, stride)
    zs = sampled_indices(fnz, stride)
    ncx, ncy, ncz = len(xs), len(ys), len(zs)
    sampled_cells = ncx * ncy * ncz
    if max_cells > 0 and sampled_cells > max_cells:
        raise ValueError(
            f"downsampled VTK would contain {sampled_cells} cells, above "
            f"--max-cells={max_cells}; increase --stride or set --max-cells 0"
        )

    length = height * (nx - 1) / max(1, nz - 1)
    width = height * (ny - 1) / max(1, nz - 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# vtk DataFile Version 3.0\n")
        out.write("EMsFEM ANN fine-density checkpoint downsample\n")
        out.write("ASCII\n")
        out.write("DATASET STRUCTURED_POINTS\n")
        out.write(f"DIMENSIONS {ncx + 1} {ncy + 1} {ncz + 1}\n")
        out.write("ORIGIN 0 0 0\n")
        out.write(f"SPACING {length / ncx:.17g} {width / ncy:.17g} {height / ncz:.17g}\n")

        if displacement_path is not None:
            write_displacement_vectors(out, displacement_path, nx, ny, nz, ncx, ncy, ncz)

        out.write(f"CELL_DATA {sampled_cells}\n")

        write_scalar(
            out, "rho", rho_path, rho_offset, mask_path, mask_offset,
            fnx, fny, xs, ys, zs, "rho"
        )
        write_scalar(
            out, "rho_plot", rho_path, rho_offset, mask_path, mask_offset,
            fnx, fny, xs, ys, zs, "rho_plot"
        )
        write_scalar(
            out, "design_mask", rho_path, rho_offset, mask_path, mask_offset,
            fnx, fny, xs, ys, zs, "mask"
        )

    print(
        f"Wrote {out_path} ({sampled_cells} sampled cells from {fine_cells} fine cells, "
        f"stride={stride})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an EMsFEM ANN rho_phys PETSc checkpoint to downsampled VTK."
    )
    parser.add_argument("--density", required=True, type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument(
        "--displacement",
        type=Path,
        help="optional coarse-node EMsFEM ANN *_final_u*.petscbin checkpoint",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nx", required=True, type=int)
    parser.add_argument("--ny", required=True, type=int)
    parser.add_argument("--nz", required=True, type=int)
    parser.add_argument("--sub-n", default=5, type=int)
    parser.add_argument("--height", default=0.08, type=float)
    parser.add_argument(
        "--stride",
        default=5,
        type=int,
        help="sample every N fine cells; stride=5 gives one sample per coarse cell for sub_n=5",
    )
    parser.add_argument(
        "--max-cells",
        default=5_000_000,
        type=int,
        help="guardrail for sampled VTK cells; use 0 to disable",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.sub_n < 1:
        raise ValueError("--sub-n must be at least 1")
    if args.height <= 0.0:
        raise ValueError("--height must be positive")

    write_vtk(
        args.density,
        args.mask,
        args.displacement,
        args.out,
        args.nx,
        args.ny,
        args.nz,
        args.sub_n,
        args.height,
        args.stride,
        args.max_cells,
    )


if __name__ == "__main__":
    main()
