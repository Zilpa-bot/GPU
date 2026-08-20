"""Minimal, dependency-free glTF 2.0 / GLB writer (metallic-roughness PBR).

Emits a single self-contained .glb with embedded PNG textures, welded vertices
and generated tangents, so the model drops straight into Blender, three.js,
Unreal, Godot, Windows 3D Viewer, macOS Quick Look, etc.
"""
from __future__ import annotations

import json
import struct
import numpy as np

FLOAT = 5126
UINT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def weld(mesh):
    """Deduplicate vertices on (position, normal, uv) -> arrays."""
    P = np.array(mesh.v, dtype=np.float32)
    N = np.array(mesh.n, dtype=np.float32)
    T = np.array(mesh.uv, dtype=np.float32)
    key = np.concatenate([np.round(P, 6), np.round(N, 4), np.round(T, 5)], axis=1)
    view = np.ascontiguousarray(key).view([("", key.dtype)] * key.shape[1]).ravel()
    _, first, inv = np.unique(view, return_index=True, return_inverse=True)
    idx = inv[np.array(mesh.idx, dtype=np.int64)].astype(np.uint32)
    C = None
    if mesh.col is not None:
        C = np.array(mesh.col, dtype=np.float32)[first]
    return P[first], N[first], T[first], idx, C


def compute_tangents(P, N, T, idx):
    """Per-vertex tangents (vec4 with handedness) from the UV parameterisation."""
    tan = np.zeros((len(P), 3), np.float64)
    bit = np.zeros((len(P), 3), np.float64)
    tri = idx.reshape(-1, 3)
    p0, p1, p2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    u0, u1, u2 = T[tri[:, 0]], T[tri[:, 1]], T[tri[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = u1 - u0, u2 - u0
    den = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(den) < 1e-12, 0.0, 1.0 / np.where(den == 0, 1, den))
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    b = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    for c in range(3):
        np.add.at(tan, tri[:, c], t)
        np.add.at(bit, tri[:, c], b)
    n = N.astype(np.float64)
    # Gram-Schmidt; fall back to any perpendicular where UVs are degenerate
    tt = tan - n * np.sum(n * tan, axis=1, keepdims=True)
    ln = np.linalg.norm(tt, axis=1)
    bad = ln < 1e-9
    if bad.any():
        alt = np.tile(np.array([1.0, 0.0, 0.0]), (bad.sum(), 1))
        flip = np.abs(n[bad][:, 0]) > 0.9
        alt[flip] = (0.0, 1.0, 0.0)
        tt[bad] = np.cross(n[bad], alt)
        ln = np.linalg.norm(tt, axis=1)
    tt /= np.maximum(ln, 1e-12)[:, None]
    w = np.where(np.sum(np.cross(n, tt) * bit, axis=1) < 0.0, -1.0, 1.0)
    return np.concatenate([tt, w[:, None]], axis=1).astype(np.float32)


class GLB:
    def __init__(self):
        self.bin = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.images = []
        self.textures = []
        self.materials = []
        self.material_index = {}
        self.meshes = []
        self.nodes = []
        self.samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]

    # ------------------------------------------------------------- plumbing
    def _pad(self, n=4, fill=b"\x00"):
        while len(self.bin) % n:
            self.bin += fill

    def _view(self, data: bytes, target=None, stride=None):
        self._pad(4)
        off = len(self.bin)
        self.bin += data
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        if stride:
            v["byteStride"] = stride
        self.buffer_views.append(v)
        return len(self.buffer_views) - 1

    def _accessor(self, arr, kind, comp, target):
        data = np.ascontiguousarray(arr).tobytes()
        view = self._view(data, target)
        a = {"bufferView": view, "componentType": comp, "count": int(arr.shape[0]), "type": kind}
        flat = arr.reshape(arr.shape[0], -1)
        a["min"] = [float(x) for x in flat.min(axis=0)]
        a["max"] = [float(x) for x in flat.max(axis=0)]
        self.accessors.append(a)
        return len(self.accessors) - 1

    # -------------------------------------------------------------- content
    def add_texture(self, png_bytes: bytes, name: str):
        view = self._view(png_bytes)
        self.images.append({"bufferView": view, "mimeType": "image/png", "name": name})
        self.textures.append({"sampler": 0, "source": len(self.images) - 1})
        return len(self.textures) - 1

    def add_material(self, name, base_color=(1, 1, 1, 1), metallic=1.0, roughness=1.0,
                     base_tex=None, orm_tex=None, normal_tex=None, normal_scale=1.0,
                     alpha_mode="OPAQUE", alpha_cutoff=0.5, double_sided=False,
                     emissive=(0, 0, 0), extensions=None, occlusion_strength=1.0):
        pbr = {
            "baseColorFactor": list(base_color),
            "metallicFactor": metallic,
            "roughnessFactor": roughness,
        }
        if base_tex is not None:
            pbr["baseColorTexture"] = {"index": base_tex}
        if orm_tex is not None:
            pbr["metallicRoughnessTexture"] = {"index": orm_tex}
        m = {"name": name, "pbrMetallicRoughness": pbr, "doubleSided": bool(double_sided)}
        if orm_tex is not None:
            m["occlusionTexture"] = {"index": orm_tex, "strength": occlusion_strength}
        if normal_tex is not None:
            m["normalTexture"] = {"index": normal_tex, "scale": normal_scale}
        if alpha_mode != "OPAQUE":
            m["alphaMode"] = alpha_mode
            if alpha_mode == "MASK":
                m["alphaCutoff"] = alpha_cutoff
        if any(emissive):
            m["emissiveFactor"] = list(emissive)
        if extensions:
            m["extensions"] = extensions
        self.materials.append(m)
        self.material_index[name] = len(self.materials) - 1
        return len(self.materials) - 1

    def add_mesh(self, mesh, material_name):
        P, N, T, idx, C = weld(mesh)
        TAN = compute_tangents(P, N, T, idx)
        prim = {
            "attributes": {
                "POSITION": self._accessor(P, "VEC3", FLOAT, ARRAY_BUFFER),
                "NORMAL": self._accessor(N, "VEC3", FLOAT, ARRAY_BUFFER),
                "TEXCOORD_0": self._accessor(T, "VEC2", FLOAT, ARRAY_BUFFER),
                "TANGENT": self._accessor(TAN, "VEC4", FLOAT, ARRAY_BUFFER),
            },
            "indices": self._accessor(idx.reshape(-1, 1), "SCALAR", UINT, ELEMENT_ARRAY_BUFFER),
            "material": self.material_index[material_name],
            "mode": 4,
        }
        if C is not None:
            prim["attributes"]["COLOR_0"] = self._accessor(C, "VEC4", FLOAT, ARRAY_BUFFER)
        self.meshes.append({"name": mesh.name, "primitives": [prim]})
        self.nodes.append({"name": mesh.name, "mesh": len(self.meshes) - 1})
        return len(P), len(idx) // 3

    def serialize(self, extensions_used=()):
        gltf = {
            "asset": {"version": "2.0", "generator": "window3d procedural builder"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.bin)}],
        }
        if self.textures:
            gltf["textures"] = self.textures
            gltf["images"] = self.images
            gltf["samplers"] = self.samplers
        if extensions_used:
            gltf["extensionsUsed"] = list(extensions_used)

        js = json.dumps(gltf, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        binc = bytes(self.bin)
        binc += b"\x00" * ((4 - len(binc) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(binc)
        out = bytearray()
        out += struct.pack("<III", 0x46546C67, 2, total)
        out += struct.pack("<II", len(js), 0x4E4F534A) + js
        out += struct.pack("<II", len(binc), 0x004E4942) + binc
        return bytes(out)
