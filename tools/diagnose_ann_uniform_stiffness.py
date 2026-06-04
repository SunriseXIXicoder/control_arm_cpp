#!/usr/bin/env python3
"""Compare ANN EMsFEM element stiffness for uniform material.

For a uniform material field, changing the density only scales the stiffness;
the shape-function mapping should not jump merely because rho is below 1.  The
production code bypasses ANN at material=1, but uses ANN at intermediate
uniform material values, so this check quantifies the discontinuity:

    ANN_ke(uniform material m)  vs.  m * solid_trilinear_H8_ke

The JSON parsing intentionally mirrors src/ann_model.cpp: all numbers are
scanned in file order, and dense weights are consumed as W[in, out] followed by
bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verify_h8_cantilever_coarsening import h8_element_stiffness


LOCAL_NODE_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=int,
)


def scan_weight_numbers(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    clean = text.translate(str.maketrans({"[": " ", "]": " "}))
    values = np.fromstring(clean, sep=",", dtype=np.float64)
    if values.size == 0:
        raise RuntimeError(f"no numeric weights parsed from {path}")
    return values


def activate(values: np.ndarray, kind: str | None) -> np.ndarray:
    if kind is None or kind == "null" or kind == "":
        return values
    if kind == "sigmoid":
        return 1.0 / (np.exp(-values) + 1.0)
    if kind == "relu":
        return np.maximum(values, 0.0)
    if kind == "tanh":
        return np.tanh(values)
    if kind == "elu":
        return np.where(values >= 0.0, values, np.exp(values) - 1.0)
    raise RuntimeError(f"unknown activation {kind!r}")


def function_path(ann_dir: Path, sub_n: int) -> Path:
    direct = ann_dir / "model_function.json"
    if direct.exists():
        return direct
    if sub_n == 5:
        return ann_dir / "model_function_3DEMs_FEMBLC_20220821_input_5_layer1.json"
    if sub_n == 10:
        return ann_dir / "model_function_3DEMs_FEMBLC_20220824_input_10_layer1.json"
    return direct


def weight_path(ann_dir: Path, sub_n: int, layer: int) -> Path:
    direct = ann_dir / f"model_weight_layer{layer}.json"
    if direct.exists():
        return direct
    if sub_n == 5:
        return ann_dir / f"model_weight_3DEMs_FEMBLC_20220821_input_5_layer{layer}.json"
    if sub_n == 10:
        return ann_dir / f"model_weight_3DEMs_FEMBLC_20220824_input_10_layer{layer}.json"
    return direct


def load_function(ann_dir: Path, sub_n: int) -> tuple[list[str | None], list[int]]:
    with function_path(ann_dir, sub_n).open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    activations = data[0]
    dims = [int(v) for v in data[1]]
    return activations, dims


def forward_network(weight_file: Path, activations: list[str | None], dims: list[int], material: np.ndarray) -> np.ndarray:
    numbers = scan_weight_numbers(weight_file)
    current = material
    offset = 0
    for layer_id, out_dim in enumerate(dims):
        in_dim = current.size
        wcount = in_dim * out_dim
        bcount = out_dim
        weight = numbers[offset : offset + wcount].reshape((in_dim, out_dim))
        offset += wcount
        bias = numbers[offset : offset + bcount]
        offset += bcount
        current = activate(current @ weight + bias, activations[layer_id] if layer_id < len(activations) else None)
    if offset != numbers.size:
        raise RuntimeError(f"unused weight tail in {weight_file}: used {offset}, parsed {numbers.size}")
    return current


def fine_node_id(ix: int, iy: int, iz: int, sub_n: int) -> int:
    nn = sub_n + 1
    return (iz * nn + iy) * nn + ix


def fill_trilinear_shape(sub_n: int) -> np.ndarray:
    nn = sub_n + 1
    shape = np.zeros((3 * nn * nn * nn, 24), dtype=np.float64)
    for iz in range(sub_n + 1):
        z = iz / sub_n
        for iy in range(sub_n + 1):
            y = iy / sub_n
            for ix in range(sub_n + 1):
                x = ix / sub_n
                n = np.array(
                    [
                        (1.0 - x) * (1.0 - y) * (1.0 - z),
                        x * (1.0 - y) * (1.0 - z),
                        x * y * (1.0 - z),
                        (1.0 - x) * y * (1.0 - z),
                        (1.0 - x) * (1.0 - y) * z,
                        x * (1.0 - y) * z,
                        x * y * z,
                        (1.0 - x) * y * z,
                    ]
                )
                node = fine_node_id(ix, iy, iz, sub_n)
                for c in range(3):
                    row = 3 * node + c
                    for a in range(8):
                        shape[row, 3 * a + c] = n[a]
    return shape


def predict_shape(ann_dir: Path, sub_n: int, material_value: float) -> np.ndarray:
    activations, dims = load_function(ann_dir, sub_n)
    expected_output = (sub_n - 1) * (sub_n - 1) * 3 * 21
    if dims[-1] != expected_output:
        raise RuntimeError(f"last layer output {dims[-1]} does not match expected {expected_output}")

    material = np.full(sub_n * sub_n * sub_n, material_value, dtype=np.float64)
    outputs = []
    for layer in range(1, sub_n):
        out = forward_network(weight_path(ann_dir, sub_n, layer), activations, dims, material)
        outputs.append(out)
        print(f"  loaded ANN slice {layer}/{sub_n - 1}: output_norm={np.linalg.norm(out):.6e}")
    concatenated = np.concatenate(outputs)

    inside_dofs = 3 * (sub_n - 1) * (sub_n - 1) * (sub_n - 1)
    inside = np.zeros((inside_dofs, 24), dtype=np.float64)
    values21 = concatenated.reshape((inside_dofs, 21))
    inside[:, :21] = values21
    for row in range(inside_dofs):
        component = row % 3
        inside[row, 21] = (1.0 if component == 0 else 0.0) - np.sum(values21[row, 0::3])
        inside[row, 22] = (1.0 if component == 1 else 0.0) - np.sum(values21[row, 1::3])
        inside[row, 23] = (1.0 if component == 2 else 0.0) - np.sum(values21[row, 2::3])

    if sub_n == 20:
        # The 20^3 generator stores coarse-basis columns in D3_CSQ_2/MATLAB
        # order [100,000,010,110,101,001,011,111].  Convert to the C++ H8
        # order [000,100,110,010,001,101,111,011].
        matlab_to_cpp = [1, 0, 3, 2, 5, 4, 7, 6]
        permuted = np.zeros_like(inside)
        for matlab_node, cpp_node in enumerate(matlab_to_cpp):
            permuted[:, 3 * cpp_node : 3 * cpp_node + 3] = inside[
                :, 3 * matlab_node : 3 * matlab_node + 3
            ]
        inside = permuted

    shape = fill_trilinear_shape(sub_n)
    row_in = 0
    for iz in range(1, sub_n):
        for iy in range(1, sub_n):
            for ix in range(1, sub_n):
                node = fine_node_id(ix, iy, iz, sub_n)
                for c in range(3):
                    shape[3 * node + c, :] = inside[row_in, :]
                    row_in += 1
    return shape


def accumulate_stiffness(shape: np.ndarray, sub_n: int, material_value: float, poisson: float) -> np.ndarray:
    fine_ke = h8_element_stiffness(1.0 / sub_n, 1.0 / sub_n, 1.0 / sub_n, 1.0, poisson)
    ke = np.zeros((24, 24), dtype=np.float64)
    for fz in range(sub_n):
        for fy in range(sub_n):
            for fx in range(sub_n):
                bmap = np.zeros((24, 24), dtype=np.float64)
                for local_node, (ox, oy, oz) in enumerate(LOCAL_NODE_OFFSETS):
                    fine_node = fine_node_id(fx + int(ox), fy + int(oy), fz + int(oz), sub_n)
                    for c in range(3):
                        bmap[3 * local_node + c, :] = shape[3 * fine_node + c, :]
                ke += material_value * (bmap.T @ fine_ke @ bmap)
    return ke


def strain_mode_energy_ratios(ann_ke: np.ndarray, ref_ke: np.ndarray) -> dict[str, float]:
    # Corner displacement patterns for simple constant-strain-like modes.
    coords = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    modes: dict[str, np.ndarray] = {}
    ux = np.zeros(24)
    uy = np.zeros(24)
    uz = np.zeros(24)
    for a, (x, y, z) in enumerate(coords):
        ux[3 * a + 0] = x
        uy[3 * a + 1] = y
        uz[3 * a + 2] = z
    modes["axial_x"] = ux
    modes["axial_y"] = uy
    modes["axial_z"] = uz
    shear_xy = np.zeros(24)
    shear_xz = np.zeros(24)
    z_slope_x = np.zeros(24)
    for a, (x, y, z) in enumerate(coords):
        shear_xy[3 * a + 0] = y
        shear_xy[3 * a + 1] = x
        shear_xz[3 * a + 0] = z
        shear_xz[3 * a + 2] = x
        z_slope_x[3 * a + 2] = x
    modes["shear_xy"] = shear_xy
    modes["shear_xz"] = shear_xz
    modes["z_slope_x"] = z_slope_x
    ratios = {}
    for name, vec in modes.items():
        ref = float(vec @ ref_ke @ vec)
        ann = float(vec @ ann_ke @ vec)
        ratios[name] = ann / ref if abs(ref) > 1e-30 else float("nan")
    return ratios


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
    print(
        f"Uniform ANN stiffness diagnostic: sub_n={args.sub_n}, "
        f"rho={args.rho}, material={material_value:.12e}"
    )
    shape = predict_shape(args.ann_dir, args.sub_n, material_value)
    ann_ke = accumulate_stiffness(shape, args.sub_n, material_value, args.poisson)
    solid_ke = h8_element_stiffness(1.0, 1.0, 1.0, 1.0, args.poisson)
    ref_ke = material_value * solid_ke
    diff = ann_ke - ref_ke
    print()
    print(f"trace_ratio           = {np.trace(ann_ke) / np.trace(ref_ke):.12e}")
    print(f"frobenius_rel_error   = {np.linalg.norm(diff) / np.linalg.norm(ref_ke):.12e}")
    print(f"max_abs_rel_to_refmax = {np.max(np.abs(diff)) / np.max(np.abs(ref_ke)):.12e}")
    for name, ratio in strain_mode_energy_ratios(ann_ke, ref_ke).items():
        print(f"energy_ratio_{name:>11s} = {ratio:.12e}")


if __name__ == "__main__":
    main()
