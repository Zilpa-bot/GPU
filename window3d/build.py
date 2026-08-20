"""Build the window asset: window.glb, window.obj/.mtl, textures, viewer.html."""
from __future__ import annotations

import argparse
import base64
import os
import sys

import numpy as np

import ao
import model
import textures
from gltf import GLB, weld

_PARTS_CACHE = {}


def get_parts(include_grille=True, include_wall=False, bake_ao=True):
    """Build (and cache) the geometry, with ambient occlusion baked once."""
    key = (include_grille, include_wall, bake_ao)
    if key not in _PARTS_CACHE:
        parts = model.build(include_grille=include_grille, include_wall=include_wall)
        if bake_ao:
            ao.bake(parts)
        _PARTS_CACHE[key] = parts
    return _PARTS_CACHE[key]

HERE = os.path.dirname(os.path.abspath(__file__))


def srgb(hex_or_tuple):
    """sRGB (0-255 or #rrggbb) -> linear, which is what glTF factors expect."""
    if isinstance(hex_or_tuple, str):
        h = hex_or_tuple.lstrip("#")
        c = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    else:
        c = tuple(v / 255 for v in hex_or_tuple)
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


# Colours sampled from the reference photo and converted to intrinsic albedo
# (the photo values include room light, so they are pulled down accordingly).
MATERIALS = {
    #  name          albedo sRGB   metal  rough  extra
    "alu_frame":    dict(color="#63676a", metal=0.32, rough=0.40),
    "alu_slat":     dict(color="#797d80", metal=0.34, rough=0.37),
    "lath_vented":  dict(color="#797d80", metal=0.34, rough=0.37, slat=True, mask=True),
    "lath_solid":   dict(color="#797d80", metal=0.34, rough=0.37, slat=True),
    "handle":       dict(color="#2a2c2e", metal=0.20, rough=0.31),
    "guide":        dict(color="#3b3e40", metal=0.25, rough=0.52),
    "seal":         dict(color="#141516", metal=0.00, rough=0.90),
    "dark_recess":  dict(color="#0e0f10", metal=0.00, rough=0.85),
    "iron":         dict(color="#6a7076", metal=0.55, rough=0.46),
    "plaster":      dict(color="#e8e6e2", metal=0.00, rough=0.92),
    "glass":        dict(color="#aab8bd", metal=0.00, rough=0.040, glass=True),
}

DRAW_LAST = {"glass"}


def build_asset(res=1024, include_grille=True, include_wall=False, bake_ao=True):
    maps, means = textures.build_all(res)
    glb = GLB()
    tex = {k: glb.add_texture(v, k) for k, v in maps.items()}

    base_t = tex["detail_basecolor.png"]
    orm_t = tex["detail_orm.png"]
    nrm_t = tex["detail_normal.png"]

    for name, spec in MATERIALS.items():
        col = srgb(spec["color"])
        alpha = 1.0
        bt, ot, nt = base_t, orm_t, nrm_t
        bmean = means["basecolor"]
        rmean = means["roughness"]
        alpha_mode, double = "OPAQUE", False
        cutoff = 0.5

        if spec.get("slat"):
            bt = tex["slat_basecolor.png"]
            bmean = means["slat"]
            if spec.get("mask"):
                alpha_mode, cutoff, double = "MASK", 0.5, True
        if spec.get("glass"):
            ot = tex["glass_orm.png"]
            rmean = means["glass_roughness"]
            nt = None
            alpha_mode, double, alpha = "BLEND", True, 0.16

        glb.add_material(
            name,
            base_color=tuple(min(c / bmean, 1.0) for c in col) + (alpha,),
            metallic=min(spec["metal"] / means["metallic"], 1.0),
            roughness=min(spec["rough"] / rmean, 1.0),
            base_tex=bt, orm_tex=ot, normal_tex=nt,
            normal_scale=0.35 if spec.get("slat") else 0.55,
            alpha_mode=alpha_mode, alpha_cutoff=cutoff, double_sided=double,
            occlusion_strength=1.0,
        )

    parts = list(get_parts(include_grille, include_wall, bake_ao))
    parts.sort(key=lambda m: m.material in DRAW_LAST)   # transparent last
    stats = []
    for m in parts:
        nv, nt_ = glb.add_mesh(m, m.material)
        stats.append((m.name, m.material, nv, nt_))
    return glb, maps, means, stats


# ------------------------------------------------------------------- OBJ/MTL
def write_obj(parts, path, means):
    lines_v, lines_vn, lines_vt, body = [], [], [], []
    off_v = off_n = off_t = 1
    for m in parts:
        P, N, T, idx, C = weld(m)
        body.append(f"o {m.name}\nusemtl {m.material}\ns 1")
        if C is None:
            lines_v += [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in P]
        else:   # baked AO travels as vertex colour, the usual OBJ extension
            lines_v += [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
                        for p, c in zip(P, C)]
        lines_vn += [f"vn {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" for p in N]
        lines_vt += [f"vt {p[0]:.5f} {p[1]:.5f}" for p in T]
        tri = idx.reshape(-1, 3)
        for a, b, c in tri:
            body.append("f " + " ".join(
                f"{i+off_v}/{i+off_t}/{i+off_n}" for i in (a, b, c)))
        off_v += len(P)
        off_n += len(N)
        off_t += len(T)
    with open(path, "w") as f:
        f.write("# window3d - procedural aluminium sliding window\n")
        f.write(f"mtllib {os.path.basename(path).replace('.obj', '.mtl')}\n")
        f.write("\n".join(lines_v) + "\n")
        f.write("\n".join(lines_vt) + "\n")
        f.write("\n".join(lines_vn) + "\n")
        f.write("\n".join(body) + "\n")


def write_mtl(path, means):
    with open(path, "w") as f:
        f.write("# PBR (PBR extensions: Pr roughness, Pm metallic, norm normal map)\n")
        for name, spec in MATERIALS.items():
            col = srgb(spec["color"])
            base = "textures/slat_basecolor.png" if spec.get("slat") else "textures/detail_basecolor.png"
            bmean = means["slat"] if spec.get("slat") else means["basecolor"]
            f.write(f"\nnewmtl {name}\n")
            f.write("Kd {:.4f} {:.4f} {:.4f}\n".format(*[min(c / bmean, 1.0) for c in col]))
            f.write("Ks 0.5 0.5 0.5\n")
            f.write(f"Pr {min(spec['rough'] / means['roughness'], 1.0):.4f}\n")
            f.write(f"Pm {min(spec['metal'] / means['metallic'], 1.0):.4f}\n")
            f.write(f"map_Kd {base}\n")
            if not spec.get("glass"):
                f.write("norm textures/detail_normal.png\n")
                f.write("map_Pr textures/detail_orm.png\n")
            if spec.get("mask"):
                f.write(f"map_d {base}\n")
            if spec.get("glass"):
                f.write("d 0.16\nNi 1.52\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--no-grille", action="store_true")
    ap.add_argument("--wall", action="store_true")
    ap.add_argument("--no-ao", action="store_true", help="skip the baked AO pass")
    ap.add_argument("--viewer-res", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "textures"), exist_ok=True)
    glb, maps, means, stats = build_asset(args.res, not args.no_grille, args.wall,
                                         not args.no_ao)
    data = glb.serialize()
    glb_path = os.path.join(args.out, "window.glb")
    open(glb_path, "wb").write(data)

    parts = get_parts(not args.no_grille, args.wall, not args.no_ao)
    write_obj(parts, os.path.join(args.out, "window.obj"), means)
    write_mtl(os.path.join(args.out, "window.mtl"), means)
    for k, v in maps.items():
        open(os.path.join(args.out, "textures", k), "wb").write(v)

    # a lighter GLB for the embedded web viewer
    vglb, _, _, _ = build_asset(args.viewer_res, not args.no_grille, args.wall,
                                not args.no_ao)
    vdata = vglb.serialize()
    tpl = open(os.path.join(HERE, "viewer_template.html")).read()
    html = tpl.replace("__GLB_BASE64__", base64.b64encode(vdata).decode())
    open(os.path.join(args.out, "viewer.html"), "w").write(html)

    print(f"{'part':26s} {'material':13s} {'verts':>7s} {'tris':>7s}")
    for n, mat, nv, nt in stats:
        print(f"{n:26s} {mat:13s} {nv:7d} {nt:7d}")
    print(f"\ntotal          {sum(s[2] for s in stats):7d} verts "
          f"{sum(s[3] for s in stats):7d} tris")
    print(f"window.glb     {len(data)/1024:8.1f} KB")
    print(f"viewer.html    {len(html)/1024:8.1f} KB")


if __name__ == "__main__":
    main()
