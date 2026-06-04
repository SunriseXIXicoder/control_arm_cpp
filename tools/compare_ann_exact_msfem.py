#!/usr/bin/env python3
"""Compare ANN EMsFEM against an exact local fine-grid MsFEM solve.

For one coarse element with sub_n^3 fine H8 cells:

  1. Assemble the full fine-grid stiffness K_f.
  2. Prescribe all boundary fine-node values from the coarse trilinear
     boundary interpolation.
  3. Solve K_II Phi_I = -K_IB Phi_B for the internal fine-node basis values.
  4. Condense the exact multiscale element matrix Phi^T K_f Phi.
  5. Compare ANN and plain trilinear stiffness against that exact local matrix.

This is the local multiscale finite-element object that the ANN shape model is
supposed to approximate.  It is intentionally a small single-element diagnostic,
not a full control-arm solve.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from diagnose_ann_uniform_stiffness import (
    LOCAL_NODE_OFFSETS,
    accumulate_stiffness,
    fill_trilinear_shape,
    h8_element_stiffness,
    predict_shape,
    strain_mode_energy_ratios,
)


def fine_node_id(ix: int, iy: int, iz: int, sub_n: int) -> int:
    nn = sub_n + 1
    return (iz * nn + iy) * nn + ix


def assemble_fine_k(sub_n: int, material: np.ndarray, poisson: float) -> coo_matrix:
    nn = sub_n + 1
    ndof = 3 * nn * nn * nn
    fine_ke = h8_element_stiffness(1.0 / sub_n, 1.0 / sub_n, 1.0 / sub_n, 1.0, poisson)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    fine_id = 0
    for fz in range(sub_n):
        for fy in range(sub_n):
            for fx in range(sub_n):
                edofs: list[int] = []
                for ox, oy, oz in LOCAL_NODE_OFFSETS:
                    nid = fine_node_id(fx + int(ox), fy + int(oy), fz + int(oz), sub_n)
                    edofs.extend([3 * nid + 0, 3 * nid + 1, 3 * nid + 2])
                scale = float(material[fine_id])
                fine_id += 1
                for a, ra in enumerate(edofs):
                    for b, cb in enumerate(edofs):
                        rows.append(ra)
                        cols.append(cb)
                        vals.append(scale * fine_ke[a, b])
    return coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()


def boundary_and_internal_dofs(sub_n: int) -> tuple[np.ndarray, np.ndarray]:
    boundary: list[int] = []
    internal: list[int] = []
    for iz in range(sub_n + 1):
        for iy in range(sub_n + 1):
            for ix in range(sub_n + 1):
                nid = fine_node_id(ix, iy, iz, sub_n)
                target = boundary if (
                    ix == 0 or ix == sub_n or
                    iy == 0 or iy == sub_n or
                    iz == 0 or iz == sub_n
                ) else internal
                target.extend([3 * nid + 0, 3 * nid + 1, 3 * nid + 2])
    return np.array(boundary, dtype=int), np.array(internal, dtype=int)


def exact_msfem_shape_and_ke(sub_n: int, material: np.ndarray, poisson: float) -> tuple[np.ndarray, np.ndarray]:
    kfine = assemble_fine_k(sub_n, material, poisson)
    tri_shape = fill_trilinear_shape(sub_n)
    boundary, internal = boundary_and_internal_dofs(sub_n)

    shape = np.zeros_like(tri_shape)
    shape[boundary, :] = tri_shape[boundary, :]

    kii = kfine[internal][:, internal].tocsc()
    kib = kfine[internal][:, boundary]
    rhs = -(kib @ shape[boundary, :])
    lu = splu(kii)
    shape[internal, :] = lu.solve(rhs)

    ke = shape.T @ (kfine @ shape)
    return shape, np.asarray(ke)


def report_matrix_compare(label: str, candidate: np.ndarray, exact: np.ndarray) -> None:
    diff = candidate - exact
    print(f"{label}_trace_ratio           = {np.trace(candidate) / np.trace(exact):.12e}")
    print(f"{label}_frobenius_rel_error   = {np.linalg.norm(diff) / np.linalg.norm(exact):.12e}")
    print(f"{label}_max_abs_rel_to_exact = {np.max(np.abs(diff)) / np.max(np.abs(exact)):.12e}")
    for name, ratio in strain_mode_energy_ratios(candidate, exact).items():
        print(f"{label}_energy_ratio_{name:>11s} = {ratio:.12e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", required=True, type=Path)
    parser.add_argument("--sub-n", required=True, type=int)
    parser.add_argument("--rho", type=float, default=0.30)
    parser.add_argument("--emin", type=float, default=1.0e-6)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--poisson", type=float, default=0.30)
    args = parser.parse_args()

    material_value = args.emin + (1.0 - args.emin) * args.rho**args.penal
    material = np.full(args.sub_n * args.sub_n * args.sub_n, material_value, dtype=np.float64)

    print(
        "Exact local MsFEM comparison: "
        f"sub_n={args.sub_n}, rho={args.rho}, material={material_value:.12e}"
    )
    print("Exact solve: boundary fine nodes use trilinear coarse values; internal nodes are statically condensed.")
    print()

    print("Building exact local MsFEM matrix...")
    _, exact_ke = exact_msfem_shape_and_ke(args.sub_n, material, args.poisson)

    print("Building trilinear and ANN matrices...")
    tri_shape = fill_trilinear_shape(args.sub_n)
    tri_ke = accumulate_stiffness(tri_shape, args.sub_n, material_value, args.poisson)
    ann_shape = predict_shape(args.ann_dir, args.sub_n, material_value)
    ann_ke = accumulate_stiffness(ann_shape, args.sub_n, material_value, args.poisson)

    print()
    report_matrix_compare("trilinear_vs_exact", tri_ke, exact_ke)
    print()
    report_matrix_compare("ann_vs_exact", ann_ke, exact_ke)


if __name__ == "__main__":
    main()
