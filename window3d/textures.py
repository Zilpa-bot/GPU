"""Procedural, seamlessly tileable PBR texture set.

One shared "detail" set (base colour / ORM / normal) is authored around 1.0 so
that it can be *multiplied* by each material's factors.  That keeps the asset
small while every surface still gets real micro-variation: powder-coat orange
peel, roll-formed sheen bands on the shutter laths, paint mottling on the
wrought iron.
"""
from __future__ import annotations

import io
import numpy as np
from PIL import Image

TILE_METRES = 0.5          # one texture tile covers 0.5 m of the model


# ------------------------------------------------------------------ noise
def _smooth(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def value_noise(res, grid, rng):
    """Tileable value noise at `res` px from a `grid` x `grid` lattice."""
    lat = rng.random((grid, grid))
    xs = np.arange(res) * grid / res
    i0 = np.floor(xs).astype(int) % grid
    i1 = (i0 + 1) % grid
    f = _smooth(xs - np.floor(xs))
    a = lat[np.ix_(i0, i0)]
    b = lat[np.ix_(i1, i0)]
    c = lat[np.ix_(i0, i1)]
    d = lat[np.ix_(i1, i1)]
    fx = f[:, None]
    fy = f[None, :]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) +
            c * (1 - fx) * fy + d * fx * fy)


def fbm(res, rng, octaves=5, base_grid=4, gain=0.5):
    out = np.zeros((res, res))
    amp, total, grid = 1.0, 0.0, base_grid
    for _ in range(octaves):
        out += amp * value_noise(res, grid, rng)
        total += amp
        amp *= gain
        grid *= 2
    return out / total


def _to_u8(a):
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _png(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def height_to_normal(h, strength=1.0):
    """Tangent-space normal map from a tileable height field."""
    dx = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * strength
    dy = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * strength
    n = np.stack([-dx, -dy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return _to_u8(n * 0.5 + 0.5)


# ------------------------------------------------------------------ maps
def detail_basecolor(res=1024, seed=7):
    """Near-white multiplier: cloudy pigment drift + fine dust speckle."""
    rng = np.random.default_rng(seed)
    cloud = fbm(res, rng, octaves=4, base_grid=3)
    fine = fbm(res, rng, octaves=3, base_grid=48)
    grain = rng.normal(0.0, 1.0, (res, res))
    v = 1.0 + (cloud - 0.5) * 0.055 + (fine - 0.5) * 0.035 + grain * 0.006
    rgb = np.repeat(v[:, :, None], 3, axis=2)
    # a hint of warm/cool split so the coating is not a dead neutral
    rgb[:, :, 0] *= 1.0 + (cloud - 0.5) * 0.020
    rgb[:, :, 2] *= 1.0 - (cloud - 0.5) * 0.020
    rgb /= rgb.max()                      # keep inside 0..1 so factors stay legal
    return _png(_to_u8(rgb), "RGB"), float(rgb.mean())


def detail_orm(res=1024, seed=11):
    """R = ambient occlusion, G = roughness mult, B = metallic mult."""
    rng = np.random.default_rng(seed)
    peel = fbm(res, rng, octaves=4, base_grid=24)
    broad = fbm(res, rng, octaves=3, base_grid=5)
    ao = np.clip(0.955 + (peel - 0.5) * 0.09, 0, 1)
    # authored just under 1.0 so a material's roughnessFactor stays within spec
    rough = np.clip(0.94 + (peel - 0.5) * 0.16 + (broad - 0.5) * 0.08, 0.72, 1.0)
    metal = np.clip(0.97 + (broad - 0.5) * 0.06, 0, 1)
    return (_png(_to_u8(np.stack([ao, rough, metal], -1)), "RGB"),
            float(rough.mean()), float(metal.mean()))


def detail_normal(res=1024, seed=13):
    """Powder-coat orange peel: shallow, high frequency, isotropic."""
    rng = np.random.default_rng(seed)
    peel = fbm(res, rng, octaves=3, base_grid=26, gain=0.55)
    micro = fbm(res, rng, octaves=2, base_grid=90)
    h = peel * 0.85 + micro * 0.18
    # ~1 degree of slope: enough to break the specular, invisible as relief
    return _png(height_to_normal(h, strength=0.30), "RGB")


def slat_basecolor(res=256, seed=23, slot_w=0.48, slot_h=0.17):
    """One vented shutter-lath cell.

    U spans one 35 mm slot pitch, V spans one lath.  The punched ventilation
    slot is an alpha hole so daylight reads through it exactly like the photo,
    and the metal around it carries the roll-formed brightness gradient.
    """
    rng = np.random.default_rng(seed)
    u = (np.arange(res) + 0.5) / res
    v = (np.arange(res) + 0.5) / res
    V, U = np.meshgrid(v, u, indexing="ij")   # axis 0 is the image row = V

    # roll-formed lath: bright crown, darker toward the rolled edges
    curve = np.cos((V - 0.5) * np.pi * 1.05)
    shade = 0.72 + 0.30 * np.clip(curve, 0, 1) ** 1.5
    # the lap joint between laths reads as a hard dark line in the photo
    shade *= 1.0 - 0.46 * np.exp(-((V - 0.015) / 0.055) ** 2)
    shade *= 1.0 - 0.22 * np.exp(-((V - 0.985) / 0.045) ** 2)
    grain = fbm(res, rng, octaves=3, base_grid=16)
    shade *= 1.0 + (grain - 0.5) * 0.05

    rgb = np.repeat(np.clip(shade, 0, 2)[:, :, None], 3, axis=2)

    # capsule-shaped punched slot: a horizontal dash with rounded ends
    dx = (np.abs(U - 0.5) - (slot_w / 2 - slot_h / 2)).clip(0)
    inside = (dx ** 2 + (V - 0.5) ** 2) < (slot_h / 2) ** 2
    alpha = np.where(inside, 0.0, 1.0)
    # darken the punched rim
    rim = (~inside) & ((dx ** 2 + (V - 0.5) ** 2) < (slot_h / 2 + 0.022) ** 2)
    rgb[rim] *= 0.72

    rgb /= rgb.max()
    out = np.concatenate([_to_u8(rgb), _to_u8(alpha)[:, :, None]], axis=2)
    return _png(out, "RGBA"), float(rgb[~inside].mean())


def glass_orm(res=512, seed=31):
    """Roughness breakup for glass: cleaning smears and a little dust."""
    rng = np.random.default_rng(seed)
    smear = fbm(res, rng, octaves=3, base_grid=6)
    dust = fbm(res, rng, octaves=2, base_grid=90)
    rough = np.clip(0.62 + smear * 0.26 + dust * 0.12, 0, 1)
    ao = np.ones_like(rough)
    metal = np.zeros_like(rough)
    return (_png(_to_u8(np.stack([ao, rough, metal], -1)), "RGB"), float(rough.mean()))


def build_all(res=1024):
    """Returns (maps, means).  `means` lets the exporter divide each material
    factor by the texture's average so factor * texture == the authored value."""
    base, base_mean = detail_basecolor(res)
    orm, rough_mean, metal_mean = detail_orm(res)
    slat, slat_mean = slat_basecolor(256)
    glass, glass_rough_mean = glass_orm(res // 2)
    maps = {
        "detail_basecolor.png": base,
        "detail_orm.png": orm,
        "detail_normal.png": detail_normal(res),
        "slat_basecolor.png": slat,
        "glass_orm.png": glass,
    }
    means = {"basecolor": base_mean, "roughness": rough_mean,
             "metallic": metal_mean, "slat": slat_mean,
             "glass_roughness": glass_rough_mean}
    return maps, means


if __name__ == "__main__":
    import sys, os
    out = sys.argv[1] if len(sys.argv) > 1 else "out/textures"
    os.makedirs(out, exist_ok=True)
    maps, means = build_all(512)
    for k, v in maps.items():
        open(os.path.join(out, k), "wb").write(v)
        print(f"{k:26s} {len(v)/1024:7.1f} KB")
    print("texture means", {k: round(v, 4) for k, v in means.items()})
