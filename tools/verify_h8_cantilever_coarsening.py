#!/usr/bin/env python3
"""Small H8 cantilever check against a beam-theory analytic displacement.

The goal is not to reproduce the control-arm geometry.  It isolates one
question: if the fine density grid is kept visually detailed but the global
displacement grid is coarsened, can a linear H8 solve become much stiffer?

Model:
  * Rectangular cantilever beam, length L, width W, height H.
  * Left x=0 face fully fixed.
  * Total vertical force F is distributed over the right x=L face.
  * Analytic tip displacement uses Euler-Bernoulli bending plus a Timoshenko
    shear correction for a rectangular section.

The finite-element side uses 8-node trilinear bricks with 2x2x2 Gauss
integration, matching the element family used by the production code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


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
XI_NODE = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
ETA_NODE = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float)
ZETA_NODE = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)


@dataclass(frozen=True)
class BeamCase:
    nelx: int
    nely: int
    nelz: int
    length: float
    width: float
    height: float
    young: float
    poisson: float
    force: float


def h8_element_stiffness(hx: float, hy: float, hz: float, young: float, poisson: float) -> np.ndarray:
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    dmat = np.zeros((6, 6))
    dmat[:3, :3] = lam
    np.fill_diagonal(dmat[:3, :3], lam + 2.0 * mu)
    dmat[3, 3] = mu
    dmat[4, 4] = mu
    dmat[5, 5] = mu

    ke = np.zeros((24, 24))
    det_j = hx * hy * hz / 8.0
    gp = 1.0 / np.sqrt(3.0)
    for zeta in (-gp, gp):
        for eta in (-gp, gp):
            for xi in (-gp, gp):
                bmat = np.zeros((6, 24))
                for a in range(8):
                    dxi = 0.125 * XI_NODE[a] * (1.0 + eta * ETA_NODE[a]) * (1.0 + zeta * ZETA_NODE[a])
                    deta = 0.125 * ETA_NODE[a] * (1.0 + xi * XI_NODE[a]) * (1.0 + zeta * ZETA_NODE[a])
                    dzeta = 0.125 * ZETA_NODE[a] * (1.0 + xi * XI_NODE[a]) * (1.0 + eta * ETA_NODE[a])
                    dx = dxi * 2.0 / hx
                    dy = deta * 2.0 / hy
                    dz = dzeta * 2.0 / hz
                    col = 3 * a
                    bmat[0, col + 0] = dx
                    bmat[1, col + 1] = dy
                    bmat[2, col + 2] = dz
                    bmat[3, col + 0] = dy
                    bmat[3, col + 1] = dx
                    bmat[4, col + 1] = dz
                    bmat[4, col + 2] = dy
                    bmat[5, col + 0] = dz
                    bmat[5, col + 2] = dx
                ke += bmat.T @ dmat @ bmat * det_j
    return ke


def node_id(i: int, j: int, k: int, nely: int, nelz: int) -> int:
    return (i * (nely + 1) + j) * (nelz + 1) + k


def solve_beam(case: BeamCase) -> tuple[float, float, int]:
    nnodes = (case.nelx + 1) * (case.nely + 1) * (case.nelz + 1)
    ndof = 3 * nnodes
    hx = case.length / case.nelx
    hy = case.width / case.nely
    hz = case.height / case.nelz
    ke = h8_element_stiffness(hx, hy, hz, case.young, case.poisson)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for ex in range(case.nelx):
        for ey in range(case.nely):
            for ez in range(case.nelz):
                edofs: list[int] = []
                for ox, oy, oz in LOCAL_NODE_OFFSETS:
                    nid = node_id(ex + int(ox), ey + int(oy), ez + int(oz), case.nely, case.nelz)
                    edofs.extend([3 * nid + 0, 3 * nid + 1, 3 * nid + 2])
                for a, ra in enumerate(edofs):
                    for b, cb in enumerate(edofs):
                        rows.append(ra)
                        cols.append(cb)
                        vals.append(ke[a, b])

    kglobal = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    rhs = np.zeros(ndof)
    load_nodes = [
        node_id(case.nelx, j, k, case.nely, case.nelz)
        for j in range(case.nely + 1)
        for k in range(case.nelz + 1)
    ]
    for nid in load_nodes:
        rhs[3 * nid + 2] += -case.force / len(load_nodes)

    fixed = []
    for j in range(case.nely + 1):
        for k in range(case.nelz + 1):
            nid = node_id(0, j, k, case.nely, case.nelz)
            fixed.extend([3 * nid + 0, 3 * nid + 1, 3 * nid + 2])
    fixed = np.array(fixed, dtype=int)
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)

    u = np.zeros(ndof)
    u[free] = spsolve(kglobal[free][:, free], rhs[free])
    compliance = float(rhs @ u)
    tip_disp_from_compliance = compliance / case.force
    avg_tip_uz = float(np.mean([u[3 * nid + 2] for nid in load_nodes]))
    return tip_disp_from_compliance, avg_tip_uz, ndof


def analytic_tip_displacement(case: BeamCase) -> float:
    # Force is along z, so the bending moment is about the y axis.
    area = case.width * case.height
    iy = case.width * case.height**3 / 12.0
    shear_modulus = case.young / (2.0 * (1.0 + case.poisson))
    kappa = 5.0 / 6.0
    bend = case.force * case.length**3 / (3.0 * case.young * iy)
    shear = case.force * case.length / (kappa * shear_modulus * area)
    return bend + shear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--width", type=float, default=0.10)
    parser.add_argument("--height", type=float, default=0.10)
    parser.add_argument("--young", type=float, default=1.0)
    parser.add_argument("--poisson", type=float, default=0.30)
    parser.add_argument("--force", type=float, default=1.0)
    parser.add_argument(
        "--meshes",
        default="4x1x1,8x2x2,16x4x4,32x8x8",
        help="comma separated nelx x nely x nelz list",
    )
    args = parser.parse_args()

    print("H8 cantilever check: left face fixed, total z-force on right face")
    print("Analytic displacement = Euler-Bernoulli bending + Timoshenko shear")
    print()
    print(f"{'mesh':>10} {'dof':>8} {'FE delta':>14} {'analytic':>14} {'FE/analytic':>12}")
    print("-" * 66)
    ratios: list[tuple[str, float]] = []
    for spec in args.meshes.split(","):
        nelx, nely, nelz = [int(part) for part in spec.lower().split("x")]
        case = BeamCase(
            nelx=nelx,
            nely=nely,
            nelz=nelz,
            length=args.length,
            width=args.width,
            height=args.height,
            young=args.young,
            poisson=args.poisson,
            force=args.force,
        )
        fe_delta, avg_uz, ndof = solve_beam(case)
        analytic = analytic_tip_displacement(case)
        ratio = abs(fe_delta) / analytic
        ratios.append((spec, ratio))
        print(f"{spec:>10} {ndof:8d} {abs(fe_delta):14.6e} {analytic:14.6e} {ratio:12.6f}")

    if len(ratios) >= 2:
        coarse_spec, coarse_ratio = ratios[0]
        fine_spec, fine_ratio = ratios[-1]
        print()
        print(
            f"coarse/fine stiffness effect: {coarse_spec} compliance ratio / "
            f"{fine_spec} compliance ratio = {coarse_ratio / fine_ratio:.6f}"
        )


if __name__ == "__main__":
    main()
