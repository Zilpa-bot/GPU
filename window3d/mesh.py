"""Small dependency-light mesh toolkit used to build hard-surface joinery.

Everything is triangles with POSITION / NORMAL / TEXCOORD_0.  UVs are laid out
in *world metres* (1 uv unit == 1 metre) so that the shared procedural detail
maps tile seamlessly across every part of the model regardless of its size.
"""
from __future__ import annotations

import math
import numpy as np


class Mesh:
    def __init__(self, name: str = "mesh", material: str = "default"):
        self.name = name
        self.material = material
        self.v: list[tuple[float, float, float]] = []
        self.n: list[tuple[float, float, float]] = []
        self.uv: list[tuple[float, float]] = []
        self.col: list[tuple[float, float, float, float]] | None = None
        self.idx: list[int] = []

    # ---------------------------------------------------------------- basics
    def add_tri(self, p0, p1, p2, uv0, uv1, uv2, normal=None):
        if normal is None:
            normal = tri_normal(p0, p1, p2)
        base = len(self.v)
        for p, t in ((p0, uv0), (p1, uv1), (p2, uv2)):
            self.v.append(tuple(map(float, p)))
            self.n.append(tuple(map(float, normal)))
            self.uv.append((float(t[0]), float(t[1])))
        self.idx.extend((base, base + 1, base + 2))

    def add_quad(self, p0, p1, p2, p3, uv0, uv1, uv2, uv3, normal=None):
        if normal is None:
            normal = tri_normal(p0, p1, p2)
            if length(normal) < 1e-12:
                normal = tri_normal(p0, p2, p3)
        self.add_tri(p0, p1, p2, uv0, uv1, uv2, normal)
        self.add_tri(p0, p2, p3, uv0, uv2, uv3, normal)

    def add_polygon(self, pts, uvs, normal=None):
        """Convex polygon fan."""
        if normal is None:
            normal = polygon_normal(pts)
        for i in range(1, len(pts) - 1):
            self.add_tri(pts[0], pts[i], pts[i + 1], uvs[0], uvs[i], uvs[i + 1], normal)

    def extend(self, other: "Mesh"):
        base = len(self.v)
        self.v.extend(other.v)
        self.n.extend(other.n)
        self.uv.extend(other.uv)
        if self.col is not None and other.col is not None:
            self.col.extend(other.col)
        self.idx.extend(i + base for i in other.idx)

    def transform(self, mat: np.ndarray):
        """4x4 row-vector-on-the-right transform (mat @ column vector)."""
        if not self.v:
            return self
        p = np.array(self.v, dtype=np.float64)
        p = (mat[:3, :3] @ p.T).T + mat[:3, 3]
        nrm = np.array(self.n, dtype=np.float64)
        rot = np.linalg.inv(mat[:3, :3]).T
        nrm = (rot @ nrm.T).T
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        ln[ln == 0] = 1.0
        nrm = nrm / ln
        self.v = [tuple(x) for x in p]
        self.n = [tuple(x) for x in nrm]
        return self

    def translated(self, dx, dy, dz):
        m = np.eye(4)
        m[:3, 3] = (dx, dy, dz)
        return self.transform(m)

    def smooth_normals(self, angle_deg=32.0, weld_eps=1e-5):
        """Average normals of coincident vertices whose normals agree.

        Keeps hard machined edges crisp while smoothing tube/round surfaces.
        """
        if not self.v:
            return self
        P = np.round(np.array(self.v) / weld_eps).astype(np.int64)
        N = np.array(self.n)
        cos_lim = math.cos(math.radians(angle_deg))
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for i, key in enumerate(map(tuple, P)):
            buckets.setdefault(key, []).append(i)
        out = N.copy()
        for ids in buckets.values():
            if len(ids) < 2:
                continue
            grp = N[ids]
            used = np.zeros(len(ids), bool)
            for a in range(len(ids)):
                if used[a]:
                    continue
                sel = (grp @ grp[a]) >= cos_lim
                sel &= ~used
                if sel.sum() < 2:
                    used[a] = True
                    continue
                avg = grp[sel].sum(axis=0)
                ln = np.linalg.norm(avg)
                if ln > 1e-9:
                    avg = avg / ln
                    for k, s in enumerate(sel):
                        if s:
                            out[ids[k]] = avg
                used |= sel
        self.n = [tuple(x) for x in out]
        return self

    def bounds(self):
        a = np.array(self.v)
        return a.min(axis=0), a.max(axis=0)

    def tri_count(self):
        return len(self.idx) // 3


# -------------------------------------------------------------------- vecmath
def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(a):
    return math.sqrt(dot(a, a))


def norm(a):
    l = length(a)
    return (a[0] / l, a[1] / l, a[2] / l) if l > 1e-12 else (0.0, 0.0, 1.0)


def tri_normal(p0, p1, p2):
    return norm(cross(sub(p1, p0), sub(p2, p0)))


def polygon_normal(pts):
    n = (0.0, 0.0, 0.0)
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        n = add(n, (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))
    return norm(n)


# ------------------------------------------------------------------ primitives
_AXIS_UV = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def box(mesh: Mesh, lo, hi, uv_offset=(0.0, 0.0), skip=()):
    """Axis-aligned box with world-space planar UVs per face."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    faces = {
        "-x": ([(x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)], (-1, 0, 0)),
        "+x": ([(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)], (1, 0, 0)),
        "-y": ([(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)], (0, -1, 0)),
        "+y": ([(x0, y1, z1), (x0, y1, z0), (x1, y1, z0), (x1, y1, z1)], (0, 1, 0)),
        "-z": ([(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)], (0, 0, -1)),
        "+z": ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1)),
    }
    for key, (pts, nrm) in faces.items():
        if key in skip:
            continue
        axis = "xyz".index(key[1])
        ua, va = _AXIS_UV[axis]
        uvs = [(p[ua] + uv_offset[0], p[va] + uv_offset[1]) for p in pts]
        mesh.add_quad(*pts, *uvs, normal=nrm)
    return mesh


def profile_area(profile):
    a = 0.0
    for i in range(len(profile)):
        t0, z0 = profile[i]
        t1, z1 = profile[(i + 1) % len(profile)]
        a += t0 * z1 - t1 * z0
    return a * 0.5


def rect_tube(mesh: Mesh, x0, y0, x1, y1, profile, uv_v0=0.0):
    """Sweep a closed 2-D cross-section around a rectangle with 45 deg miters.

    ``profile`` is a closed polygon of (t, z) where ``t`` is the distance
    measured inward from the rectangle boundary and ``z`` is the world depth.
    The result is a watertight mitred frame exactly like a real extruded
    aluminium window profile.  Corner joints seal perfectly because both
    adjacent members meet on the t == t plane.
    """
    prof = list(profile)
    if profile_area(prof) > 0:          # enforce CW so normals point outward
        prof = prof[::-1]

    # arc-length along the section, used as the V texture coordinate
    vs = [uv_v0]
    for i in range(1, len(prof) + 1):
        t0, z0 = prof[i - 1]
        t1, z1 = prof[i % len(prof)]
        vs.append(vs[-1] + math.hypot(t1 - t0, z1 - z0))

    def side_pts(side, t, z):
        if side == 0:    # bottom, travels +X
            return (x0 + t, y0 + t, z), (x1 - t, y0 + t, z), (x0 + t, x1 - t)
        if side == 1:    # right, travels +Y
            return (x1 - t, y0 + t, z), (x1 - t, y1 - t, z), (y0 + t, y1 - t)
        if side == 2:    # top, travels -X
            return (x1 - t, y1 - t, z), (x0 + t, y1 - t, z), (x1 - t, x0 + t)
        return (x0 + t, y1 - t, z), (x0 + t, y0 + t, z), (y1 - t, y0 + t)  # left

    n = len(prof)
    for side in range(4):
        for k in range(n):
            t0, z0 = prof[k]
            t1, z1 = prof[(k + 1) % n]
            a0, b0, (ua0, ub0) = side_pts(side, t0, z0)
            a1, b1, (ua1, ub1) = side_pts(side, t1, z1)
            v0, v1 = vs[k], vs[k + 1]
            mesh.add_quad(a0, b0, b1, a1,
                          (ua0, v0), (ub0, v0), (ub1, v1), (ua1, v1))
    return mesh


def loft(mesh: Mesh, rings, cap_start=True, cap_end=True, uv_scale=1.0, closed_ring=True):
    """Skin a list of equally-sized point rings (a generalised tube)."""
    rings = [list(r) for r in rings]
    m = len(rings[0])
    # ring-space U coordinate = perimeter arc length
    us = [0.0]
    r0 = rings[0]
    for i in range(1, m + (1 if closed_ring else 0)):
        us.append(us[-1] + length(sub(r0[i % m], r0[i - 1])))
    # along-length V coordinate
    vs = [0.0]
    for i in range(1, len(rings)):
        c0 = centroid(rings[i - 1])
        c1 = centroid(rings[i])
        vs.append(vs[-1] + length(sub(c1, c0)))

    span = m if closed_ring else m - 1
    for i in range(len(rings) - 1):
        for j in range(span):
            k = (j + 1) % m
            p0, p1 = rings[i][j], rings[i][k]
            p2, p3 = rings[i + 1][k], rings[i + 1][j]
            mesh.add_quad(p0, p1, p2, p3,
                          (us[j] * uv_scale, vs[i] * uv_scale),
                          (us[j + 1] * uv_scale, vs[i] * uv_scale),
                          (us[j + 1] * uv_scale, vs[i + 1] * uv_scale),
                          (us[j] * uv_scale, vs[i + 1] * uv_scale))
    if cap_start:
        pts = rings[0][::-1]
        mesh.add_polygon(pts, [(p[0], p[1]) for p in pts])
    if cap_end:
        pts = rings[-1]
        mesh.add_polygon(pts, [(p[0], p[1]) for p in pts])
    return mesh


def centroid(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


def extrude(mesh: Mesh, profile, axis, a, b, cap=True, plane_origin=(0.0, 0.0)):
    """Extrude a closed 2-D polygon along a principal axis from a to b.

    ``profile`` lies in the plane of the two remaining axes, in that order.
    """
    ax = "xyz".index(axis)
    pa, pb = _AXIS_UV[ax]
    if profile_area(profile) < 0:
        profile = profile[::-1]

    def pt(p2, s):
        out = [0.0, 0.0, 0.0]
        out[ax] = s
        out[pa] = p2[0] + plane_origin[0]
        out[pb] = p2[1] + plane_origin[1]
        return tuple(out)

    ring_a = [pt(p, a) for p in profile]
    ring_b = [pt(p, b) for p in profile]
    if b < a:
        ring_a, ring_b = ring_b, ring_a
    loft(mesh, [ring_a, ring_b], cap_start=cap, cap_end=cap)
    return mesh


def circle(radius, segments, phase=0.0):
    return [(radius * math.cos(phase + 2 * math.pi * i / segments),
             radius * math.sin(phase + 2 * math.pi * i / segments)) for i in range(segments)]


def rounded_rect(w, h, r, seg=4):
    """Closed CCW rounded-rectangle polygon centred on the origin."""
    hw, hh = w / 2 - r, h / 2 - r
    pts = []
    for cx, cy, a0 in ((hw, hh, 0.0), (-hw, hh, math.pi / 2),
                       (-hw, -hh, math.pi), (hw, -hh, 1.5 * math.pi)):
        for i in range(seg + 1):
            a = a0 + (math.pi / 2) * i / seg
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def frames_along_polyline(points, up=(0.0, 0.0, 1.0)):
    """Parallel-transport-ish frames for sweeping a section along a path."""
    out = []
    n = len(points)
    for i in range(n):
        if i == 0:
            tan = norm(sub(points[1], points[0]))
        elif i == n - 1:
            tan = norm(sub(points[-1], points[-2]))
        else:
            tan = norm(add(norm(sub(points[i], points[i - 1])), norm(sub(points[i + 1], points[i]))))
        ref = up if abs(dot(tan, up)) < 0.95 else (1.0, 0.0, 0.0)
        side = norm(cross(ref, tan))
        upv = norm(cross(tan, side))
        out.append((points[i], side, upv))
    return out


def tube(mesh: Mesh, points, radius, segments=10, cap=True):
    """Sweep a circular section along a 3-D polyline."""
    rings = []
    sec = circle(radius, segments)
    for p, side, upv in frames_along_polyline(points):
        ring = [add(p, add(mul(side, u), mul(upv, v))) for (u, v) in sec]
        rings.append(ring)
    loft(mesh, rings, cap_start=cap, cap_end=cap)
    return mesh
