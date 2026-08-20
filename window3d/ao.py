"""Baked ambient occlusion.

Ray-marches a voxelised copy of the model from every vertex and stores the
result in the standard glTF ``COLOR_0`` attribute, which multiplies base
colour.  That means the crevice darkening at the mitres, the glazing rebates,
the lath laps and behind the security bars survives export -- Blender,
three.js, Godot and Quick Look all show it without any renderer-specific
tricks.
"""
from __future__ import annotations

import numpy as np


def _voxelise(parts, cell, exclude):
    tris = []
    for m in parts:
        if m.material in exclude:
            continue
        P = np.array(m.v, dtype=np.float32)
        idx = np.array(m.idx, dtype=np.int64).reshape(-1, 3)
        tris.append(P[idx])
    T = np.concatenate(tris, axis=0)

    lo = T.reshape(-1, 3).min(axis=0) - cell * 3
    hi = T.reshape(-1, 3).max(axis=0) + cell * 3
    dims = np.maximum(np.ceil((hi - lo) / cell).astype(int) + 1, 1)
    grid = np.zeros(dims, dtype=bool)

    # scatter enough barycentric samples per triangle to leave no holes
    e1 = T[:, 1] - T[:, 0]
    e2 = T[:, 2] - T[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    n = np.clip(np.ceil(np.sqrt(area) / (cell * 0.4)).astype(int) + 1, 2, 64)

    rng = np.random.default_rng(5)
    for k in np.unique(n):
        sel = n == k
        s = np.linspace(0.0, 1.0, k)
        a, b = np.meshgrid(s, s, indexing="ij")
        keep = (a + b) <= 1.0
        a, b = a[keep], b[keep]
        pts = (T[sel][:, 0][:, None, :]
               + e1[sel][:, None, :] * a[None, :, None]
               + e2[sel][:, None, :] * b[None, :, None]).reshape(-1, 3)
        ijk = np.floor((pts - lo) / cell).astype(np.int32)
        np.clip(ijk, 0, dims - 1, out=ijk)
        grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return grid, lo, dims


def _hemisphere(n_rays, seed=3):
    """Cosine-weighted directions in the +Z hemisphere (low-discrepancy)."""
    i = np.arange(n_rays) + 0.5
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt(i / n_rays)
    z = np.sqrt(np.maximum(1.0 - r * r, 0.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def bake(parts, cell=0.006, rays=28, radius=0.13, strength=0.92,
         steps=16, exclude=("glass",), floor=0.24):
    grid, lo, dims = _voxelise(parts, cell, exclude)
    dirs = _hemisphere(rays)
    ts = np.linspace(cell * 1.6, radius, steps)
    weights = 1.0 - (ts / radius) ** 1.5          # near hits occlude most

    for m in parts:
        P = np.array(m.v, dtype=np.float64)
        N = np.array(m.n, dtype=np.float64)
        if len(P) == 0:
            continue
        # per-vertex tangent frame
        helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(N), 1))
        flip = np.abs(N[:, 2]) > 0.9
        helper[flip] = (1.0, 0.0, 0.0)
        Tx = np.cross(helper, N)
        Tx /= np.maximum(np.linalg.norm(Tx, axis=1, keepdims=True), 1e-9)
        Ty = np.cross(N, Tx)

        origin = P + N * (cell * 1.2)
        occ = np.zeros(len(P))
        wsum = weights.sum()
        for d in dirs:
            ray = Tx * d[0] + Ty * d[1] + N * d[2]
            hit = np.zeros(len(P), bool)
            acc = np.zeros(len(P))
            for t, w in zip(ts, weights):
                p = origin + ray * t
                ijk = np.floor((p - lo) / cell).astype(np.int32)
                inside = np.all((ijk >= 0) & (ijk < dims), axis=1)
                np.clip(ijk, 0, dims - 1, out=ijk)
                solid = grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] & inside & ~hit
                acc += np.where(solid, w, 0.0)
                hit |= solid
            occ += acc
        ao = 1.0 - strength * np.clip(occ / (rays * wsum) * 2.6, 0.0, 1.0)
        ao = np.clip(ao, floor, 1.0)
        m.col = [(float(a), float(a), float(a), 1.0) for a in ao]
    return parts
