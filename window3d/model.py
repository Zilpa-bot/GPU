"""Parametric reconstruction of the aluminium sliding window in the source photo.

Everything below is driven by dimensions read off the reference image, using the
600 mm floor tiles and the standard Israeli window module as the metric anchor
(see MEASUREMENTS in README.md).  Units are metres.

Axes:  +X right, +Y up, +Z toward the room.  z = 0 is the exterior face of the
frame, z = 0.100 is the room-side face.  The origin sits at the centre of the
frame's bottom edge, so the model drops onto a wall opening cleanly.
"""
from __future__ import annotations

import math
from mesh import (Mesh, box, rect_tube, extrude, tube, loft, circle,
                  rounded_rect, add, mul, norm, sub)

# --------------------------------------------------------------- dimensions
W = 1.200            # overall unit width
H_SASH = 1.000       # frame height (below the shutter box)
H_BOX = 0.280        # integrated roller-shutter headbox
H = H_SASH + H_BOX
D = 0.100            # frame depth through the wall

FRAME_FACE = 0.049   # visible frame face width
REBATE = 0.040       # inner wall of the frame rebate

SASH_FACE = 0.042    # sash stile/rail face width
SASH_DEPTH = 0.032
INNER_SASH_Z = (0.050, 0.082)   # room-side track
OUTER_SASH_Z = (0.006, 0.038)   # street-side track
INTERLOCK = 0.045    # how much the two sashes overlap at the meeting stile
SASH_CLEARANCE = 0.002   # running gap between sash and frame rebate

GLASS_T = 0.006

BOX_Z0, BOX_Z1 = -0.062, 0.100

SHUTTER_Z = -0.020           # centre of the roller-shutter curtain
LATH_PITCH = 0.050           # 9 laths across the lowered curtain in the photo
LATH_T = 0.011
LATH_SLOT_PITCH = 0.058      # horizontal spacing of the punched vent slots
SHUTTER_HALF_W = 0.585
SHUTTER_BOTTOM = 0.500       # curtain is lowered to just under half height

GRILLE_Z = -0.092
GRILLE_BAR_SPACING = 0.200
GRILLE_BAR_R = 0.008

UV_TILE = 0.125      # metres covered by one tile of the shared detail maps

HANDLE_LEN = 0.200
HANDLE_Y = 0.660

# ------------------------------------------------------------- cross sections
# (t, z): t is measured inward from the member's outer edge.
FRAME_PROFILE = [
    (0.000, 0.000),
    (0.000, 0.100),
    (0.011, 0.100),
    (0.013, 0.0973),   # shadow groove on the room-side face
    (0.017, 0.0973),
    (0.019, 0.100),
    (0.044, 0.100),
    (0.049, 0.0955),   # front chamfer
    (0.049, 0.0870),
    (0.040, 0.0800),   # inner track rebate
    (0.040, 0.0480),
    (0.049, 0.0430),
    (0.049, 0.0330),
    (0.040, 0.0280),   # outer track rebate
    (0.040, 0.0060),
    (0.046, 0.0000),
]


def sash_profile(z0, z1):
    """Sash tube section with the glazing groove facing the pane."""
    d = z1 - z0
    f = SASH_FACE
    gm = z0 + d * 0.5
    return [
        (0.000, z0),
        (0.000, z1),
        (f - 0.005, z1),
        (f, z1 - 0.005),
        (f, gm + 0.008),
        (f - 0.013, gm + 0.005),   # glazing groove
        (f - 0.013, gm - 0.005),
        (f, gm - 0.008),
        (f, z0 + 0.005),
        (f - 0.005, z0),
    ]


# ------------------------------------------------------------------- helpers
def arc_points(y0, y1, zc, bulge, n=7):
    """Roll-formed lath face: a shallow arc bulging toward the street."""
    pts = []
    for i in range(n):
        s = i / (n - 1)
        y = y0 + (y1 - y0) * s
        pts.append((y, zc - bulge * math.sin(math.pi * s)))
    return pts


def lath(mesh: Mesh, x0, x1, y0, height, zc, u_pitch, thickness=LATH_T):
    """One shutter lath as front/back sheets sharing a UV cell.

    Front and back are given *identical* UVs so a punched slot in the alpha
    map removes both sheets at the same place and daylight reads through,
    exactly like the vented laths in the photograph.
    """
    front = arc_points(y0, y0 + height, zc - thickness / 2, 0.0035, 9)
    back = arc_points(y0, y0 + height, zc + thickness / 2, 0.0035, 9)
    n = len(front)

    def uv(x, i):
        return (x / u_pitch, i / (n - 1))

    for i in range(n - 1):
        (ya, za), (yb, zb) = front[i], front[i + 1]
        mesh.add_quad((x1, ya, za), (x0, ya, za), (x0, yb, zb), (x1, yb, zb),
                      uv(x1, i), uv(x0, i), uv(x0, i + 1), uv(x1, i + 1))
        (ya, za), (yb, zb) = back[i], back[i + 1]
        mesh.add_quad((x0, ya, za), (x1, ya, za), (x1, yb, zb), (x0, yb, zb),
                      uv(x0, i), uv(x1, i), uv(x1, i + 1), uv(x0, i + 1))
    # rolled top / bottom edges and the end caps, kept on solid parts of the UV
    for (fp, bp, v) in ((front[0], back[0], 0.0), (front[-1], back[-1], 1.0)):
        (ya, za), (yb, zb) = fp, bp
        quad = ((x0, ya, za), (x1, ya, za), (x1, yb, zb), (x0, yb, zb))
        if v == 1.0:
            quad = quad[::-1]
        mesh.add_quad(*quad, (0.0, v), (0.25, v), (0.25, v), (0.0, v))
    for x, flip in ((x0, False), (x1, True)):
        pts = [(x, p[0], p[1]) for p in front] + [(x, p[0], p[1]) for p in reversed(back)]
        if flip:
            pts = pts[::-1]
        mesh.add_polygon(pts, [(0.0, 0.0)] * len(pts))
    return mesh


# ------------------------------------------------------------------- parts
def build_frame():
    m = Mesh("frame", "alu_frame")
    rect_tube(m, -W / 2, 0.0, W / 2, H_SASH, FRAME_PROFILE)
    return m


def build_box():
    """Roller-shutter headbox: body, recessed face plate and the curtain slot."""
    m = Mesh("shutter_box", "alu_frame")
    slot_x = SHUTTER_HALF_W
    slot_z0, slot_z1 = SHUTTER_Z - 0.014, SHUTTER_Z + 0.014
    box(m, (-W / 2, H_SASH, BOX_Z0), (W / 2, H, BOX_Z1), skip=("+z", "-y"))

    # bottom face, split around the curtain slot
    y = H_SASH
    for (a, b) in ((BOX_Z0, slot_z0), (slot_z1, BOX_Z1)):
        m.add_quad((-W / 2, y, a), (W / 2, y, a), (W / 2, y, b), (-W / 2, y, b),
                   (-W / 2, a), (W / 2, a), (W / 2, b), (-W / 2, b), normal=(0, -1, 0))
    for (a, b) in ((-W / 2, -slot_x), (slot_x, W / 2)):
        m.add_quad((a, y, slot_z0), (b, y, slot_z0), (b, y, slot_z1), (a, y, slot_z1),
                   (a, slot_z0), (b, slot_z0), (b, slot_z1), (a, slot_z1), normal=(0, -1, 0))

    # raised lip around the recessed face plate
    rect_tube(m, -W / 2, H_SASH, W / 2, H,
              [(0.000, 0.1000), (0.000, 0.1045), (0.017, 0.1045),
               (0.021, 0.0965), (0.000, 0.0965)])

    # face plate with a shadow groove near the top, extruded across the box
    inset = 0.021
    py0, py1 = H_SASH + inset, H - inset
    gy = py1 - 0.052
    plate = [
        (py0, 0.0900), (py0, 0.0965),
        (gy - 0.002, 0.0965), (gy, 0.0943), (gy + 0.004, 0.0943), (gy + 0.006, 0.0965),
        (py1, 0.0965), (py1, 0.0900),
    ]
    extrude(m, plate, "x", -W / 2 + inset, W / 2 - inset)
    return m


def build_slot_cavity():
    m = Mesh("slot_cavity", "dark_recess")
    # run the cavity slightly wider than the slot in the box underside so its
    # end caps end up buried inside the box body instead of showing through
    box(m, (-SHUTTER_HALF_W - 0.006, H_SASH, SHUTTER_Z - 0.014),
        (SHUTTER_HALF_W + 0.006, H_SASH + 0.045, SHUTTER_Z + 0.014),
        skip=("-y", "-x", "+x"))
    return m


def sash_ybounds():
    return REBATE + SASH_CLEARANCE, H_SASH - REBATE - SASH_CLEARANCE


def sash_rects():
    """(x0, x1) of the two sashes.  The clearance keeps the sash faces off the
    frame rebate plane -- coincident faces would z-fight in every renderer."""
    ox0 = -W / 2 + REBATE + SASH_CLEARANCE
    ox1 = W / 2 - REBATE - SASH_CLEARANCE
    span = ox1 - ox0
    sw = (span + INTERLOCK) / 2.0
    left = (ox0, ox0 + sw)
    right = (ox1 - sw, ox1)
    return left, right


def build_sashes():
    meshes = []
    left, right = sash_rects()
    y0, y1 = sash_ybounds()
    for name, (x0, x1), (z0, z1) in (("sash_left", left, OUTER_SASH_Z),
                                     ("sash_right", right, INNER_SASH_Z)):
        m = Mesh(name, "alu_frame")
        rect_tube(m, x0, y0, x1, y1, sash_profile(z0, z1))
        meshes.append(m)

        g = Mesh(name + "_glass", "glass")
        gz = (z0 + z1) / 2
        inset = SASH_FACE - 0.010
        box(g, (x0 + inset, y0 + inset, gz - GLASS_T / 2),
            (x1 - inset, y1 - inset, gz + GLASS_T / 2))
        meshes.append(g)
    return meshes


def build_seals():
    """Brush/pile weather seals in the interlock and around the sash perimeter."""
    m = Mesh("seals", "seal")
    left, right = sash_rects()
    y0, y1 = sash_ybounds()
    zl = (OUTER_SASH_Z[0] + OUTER_SASH_Z[1]) / 2
    zr = (INNER_SASH_Z[0] + INNER_SASH_Z[1]) / 2
    # pile strips on the meeting stiles, filling the gap between the tracks
    box(m, (left[1] - 0.030, y0 + 0.002, OUTER_SASH_Z[1]),
        (left[1] - 0.008, y1 - 0.002, INNER_SASH_Z[0]))
    box(m, (right[0] + 0.008, y0 + 0.002, OUTER_SASH_Z[1]),
        (right[0] + 0.030, y1 - 0.002, INNER_SASH_Z[0]))
    # sill sweeps under each sash
    for (x0, x1), (z0, z1) in ((left, OUTER_SASH_Z), (right, INNER_SASH_Z)):
        box(m, (x0 + 0.010, y0 - 0.006, z0 + 0.006), (x1 - 0.010, y0 + 0.001, z1 - 0.006))
    return m


def build_handles():
    """Vertical pull handles on the two jamb-side stiles."""
    m = Mesh("handles", "handle")
    left, right = sash_rects()
    specs = [(left[0] + SASH_FACE / 2, OUTER_SASH_Z[1]),
             (right[1] - SASH_FACE / 2, INNER_SASH_Z[1])]
    for cx, zf in specs:
        # backing plate
        plate = rounded_rect(0.030, HANDLE_LEN + 0.020, 0.010, 4)
        extrude(m, plate, "z", zf - 0.001, zf + 0.006,
                plane_origin=(cx, HANDLE_Y))
        # grip: rounded section lofted with softened ends
        sec = rounded_rect(0.0155, 0.023, 0.0072, 4)[::-1]   # CW in XZ == outward for +Y
        rings = []
        prof = [(0.000, 0.35), (0.006, 0.72), (0.016, 1.0),
                (HANDLE_LEN - 0.016, 1.0), (HANDLE_LEN - 0.006, 0.72), (HANDLE_LEN, 0.35)]
        zc = zf + 0.006 + 0.013
        for dy, s in prof:
            y = HANDLE_Y - HANDLE_LEN / 2 + dy
            rings.append([(cx + u * s, y, zc + v * s) for (u, v) in sec])
        loft(m, rings)
    m.smooth_normals(38.0)
    return m


def build_shutter():
    """Roller-shutter curtain, partially lowered, with punched vent slots."""
    vented = Mesh("shutter_laths_vented", "lath_vented")
    solid = Mesh("shutter_bottom_rail", "lath_solid")
    x0, x1 = -SHUTTER_HALF_W, SHUTTER_HALF_W
    y = SHUTTER_BOTTOM
    n = int(round((H_SASH - SHUTTER_BOTTOM) / LATH_PITCH))
    for i in range(n):
        yb = y + i * LATH_PITCH
        target = solid if i == 0 else vented
        lath(target, x0, x1, yb, LATH_PITCH + 0.0015, SHUTTER_Z, u_pitch=LATH_SLOT_PITCH)
    vented.smooth_normals(30.0)
    solid.smooth_normals(30.0)

    rail = Mesh("shutter_end_rail", "alu_slat")
    box(rail, (x0, y - 0.004, SHUTTER_Z - 0.008), (x1, y + 0.002, SHUTTER_Z + 0.008))
    return [vented, solid, rail]


def build_guides():
    """Shutter guide channels on the outside of each jamb."""
    m = Mesh("shutter_guides", "alu_frame")
    for sx in (-1, 1):
        # kept inside the frame outline: W/2 = 0.600 is the hard limit
        cx = sx * (SHUTTER_HALF_W + 0.0055)
        # shallow C-channel: just deep enough to capture the curtain, and it
        # stops flush with the frame's exterior face so it never breaks the
        # unit's silhouette
        prof = [(-0.0055, SHUTTER_Z - 0.012), (0.0055, SHUTTER_Z - 0.012),
                (0.0055, SHUTTER_Z + 0.020), (-0.0055, SHUTTER_Z + 0.020),
                (-0.0055, SHUTTER_Z + 0.014), (0.0035, SHUTTER_Z + 0.014),
                (0.0035, SHUTTER_Z - 0.008), (-0.0055, SHUTTER_Z - 0.008)]
        prof = [(cx + (p[0] * sx), p[1]) for p in prof]
        extrude(m, prof, "y", 0.010, H_SASH + 0.030)   # top buried in the box
    return m


def build_grille():
    """Exterior security bars: straight verticals plus a repeating ogee band."""
    m = Mesh("security_grille", "iron")
    half = W / 2 - 0.052
    n_bars = int(round((half * 2) / GRILLE_BAR_SPACING))
    xs = [-half + (i + 0.5) * (half * 2 / n_bars) for i in range(n_bars)]
    for x in xs:
        tube(m, [(x, 0.026, GRILLE_Z), (x, H_SASH - 0.026, GRILLE_Z)], GRILLE_BAR_R, 14)
    for y in (0.030, H_SASH - 0.030):
        tube(m, [(-half, y, GRILLE_Z), (half, y, GRILLE_Z)], GRILLE_BAR_R, 14)

    # two mirrored waves whose crossings land exactly on the vertical bars,
    # producing the pointed-arch lattice seen in the photo
    # a sine and its mirror cross every half period, so the period is twice the
    # bar spacing for the pointed arches to meet exactly on the verticals
    period = 2.0 * (2 * half / n_bars)
    amp = 0.085
    band = 0.330
    yc = 0.20
    while yc < H_SASH - 0.08:
        for sign in (1, -1):
            pts = []
            steps = 120
            for i in range(steps + 1):
                x = -half + (2 * half) * i / steps
                yy = yc + sign * amp * math.sin(2 * math.pi * (x - xs[0]) / period)
                pts.append((x, yy, GRILLE_Z))
            tube(m, pts, 0.0062, 12, cap=True)
        yc += band
    m.smooth_normals(35.0)
    return m


def build_wall(thickness=0.22, size=(2.6, 2.55)):
    """Optional plastered reveal so the extracted window can be shown in situ."""
    m = Mesh("wall", "plaster")
    w, h = size
    zi = D            # room-side wall face
    zo = D - thickness
    ox0, oy0, ox1, oy1 = -W / 2, 0.0, W / 2, H
    for (a, b, c, d) in ((-w / 2, -0.35, ox0, h), (ox1, -0.35, w / 2, h),
                         (ox0, oy1, ox1, h), (ox0, -0.35, ox1, oy0)):
        if c <= a or d <= b:
            continue
        m.add_quad((a, b, zi), (c, b, zi), (c, d, zi), (a, d, zi),
                   (a, b), (c, b), (c, d), (a, d), normal=(0, 0, 1))
    # reveal returns into the opening
    for (p0, p1, nrm) in (((ox0, oy0), (ox0, oy1), (1, 0, 0)),
                          ((ox1, oy1), (ox1, oy0), (-1, 0, 0)),
                          ((ox0, oy1), (ox1, oy1), (0, -1, 0)),
                          ((ox1, oy0), (ox0, oy0), (0, 1, 0))):
        m.add_quad((p0[0], p0[1], zo), (p1[0], p1[1], zo),
                   (p1[0], p1[1], zi), (p0[0], p0[1], zi),
                   (p0[0], p0[1]), (p1[0], p1[1]), (p1[0], p1[1]), (p0[0], p0[1]),
                   normal=nrm)
    return m


def build(include_grille=True, include_wall=False):
    parts = [build_frame(), build_box(), build_slot_cavity()]
    parts += build_sashes()
    parts.append(build_seals())
    parts.append(build_handles())
    parts += build_shutter()
    parts.append(build_guides())
    if include_grille:
        parts.append(build_grille())
    if include_wall:
        parts.append(build_wall())
    # tile the shared detail maps finely; the shutter laths keep their own
    # per-cell UV layout because the punched-slot alpha lives in that space.
    for m in parts:
        if not m.material.startswith("lath"):
            m.uv = [(u / UV_TILE, v / UV_TILE) for (u, v) in m.uv]
    return parts
