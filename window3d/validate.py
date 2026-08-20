"""Geometry sanity checks, run as part of every build.

These are the failures that are invisible in a wireframe but obvious in a
render: a face whose winding disagrees with its shading normal renders black
from the side you can actually see it from, and parts that break the unit's
silhouette read as floating debris.
"""
from __future__ import annotations

import numpy as np
from mesh import tri_normal


def flipped_faces(mesh, tol=-1e-6):
    """Triangles whose vertex order points opposite their shading normal."""
    bad = []
    for t in range(0, len(mesh.idx), 3):
        a, b, c = mesh.idx[t], mesh.idx[t + 1], mesh.idx[t + 2]
        gn = tri_normal(mesh.v[a], mesh.v[b], mesh.v[c])
        sn = mesh.n[a]
        if sum(g * s for g, s in zip(gn, sn)) < tol:
            bad.append(t // 3)
    return bad


def degenerate_faces(mesh, eps=1e-12):
    bad = []
    for t in range(0, len(mesh.idx), 3):
        p = [np.array(mesh.v[mesh.idx[t + k]]) for k in range(3)]
        if np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) < eps:
            bad.append(t // 3)
    return bad


def check(parts, silhouette=None, verbose=True):
    """silhouette = (xmin, xmax, ymin, ymax) the unit must not exceed."""
    problems = []
    for p in parts:
        f = flipped_faces(p)
        if f:
            problems.append(f"{p.name}: {len(f)} face(s) wound against their normal")
        d = degenerate_faces(p)
        if d:
            problems.append(f"{p.name}: {len(d)} degenerate face(s)")
        if silhouette:
            lo, hi = p.bounds()
            x0, x1, y0, y1 = silhouette
            if lo[0] < x0 - 1e-6 or hi[0] > x1 + 1e-6:
                problems.append(f"{p.name}: breaks the silhouette in X "
                                f"({lo[0]:.4f} … {hi[0]:.4f})")
            if lo[1] < y0 - 1e-6 or hi[1] > y1 + 1e-6:
                problems.append(f"{p.name}: breaks the silhouette in Y "
                                f"({lo[1]:.4f} … {hi[1]:.4f})")
    if verbose:
        for m in problems:
            print("  !!", m)
        if not problems:
            print("  geometry checks passed")
    return problems
