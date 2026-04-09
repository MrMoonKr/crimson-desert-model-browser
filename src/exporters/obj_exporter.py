"""Wavefront OBJ + MTL exporter for parsed mesh models.

Provides:
- ObjExporter class (plugin interface via MeshExporter)
- write_obj() / write_mtl() standalone functions (backward compat with
  pac_export.py, pac_browser.py, pam_export.py legacy Mesh objects)
"""

import os

from exporters.base import MeshExporter, ExportResult, ExportWarning
from model_types import ParsedModel
from pac_parser import material_to_dds_basename


# ---- Standalone functions (backward compat with legacy Mesh objects) ----


def write_mtl(meshes, mtl_path: str, texture_rel_dir: str = "",
              available_textures: set = None, diffuse_overrides: dict = None):
    """Write an MTL file from a list of legacy Mesh objects."""
    with open(mtl_path, 'w') as f:
        f.write(f"# Materials for {os.path.basename(mtl_path).replace('.mtl', '')}\n\n")

        seen = set()
        for mesh in meshes:
            if mesh.material in seen or mesh.material == "(null)":
                continue
            seen.add(mesh.material)

            dds_base = material_to_dds_basename(mesh.material)
            tex_prefix = (".\\" + texture_rel_dir + "\\" + dds_base) if texture_rel_dir else dds_base

            f.write(f"newmtl {mesh.material}\n")
            f.write("Ka 0.2 0.2 0.2\n")
            f.write("Kd 0.8 0.8 0.8\n")
            f.write("Ks 0.5 0.5 0.5\n")
            f.write("Ns 100.0\n")

            def _tex_exists(suffix, _dds_base=dds_base):
                name = f"{_dds_base}{suffix}.dds"
                return available_textures is None or name in available_textures

            override = (diffuse_overrides or {}).get(mesh.material)
            if override:
                rel = (".\\" + texture_rel_dir + "\\" + override) if texture_rel_dir else override
                f.write(f"map_Kd {rel}\n")
            elif _tex_exists(""):
                f.write(f"map_Kd {tex_prefix}.dds\n")
            elif _tex_exists("_ma"):
                f.write(f"map_Kd {tex_prefix}_ma.dds\n")

            if _tex_exists("_n"):
                f.write(f"bump {tex_prefix}_n.dds\n")

            if _tex_exists("_sp"):
                f.write(f"map_Ks {tex_prefix}_sp.dds\n")
            elif _tex_exists("_mg"):
                f.write(f"map_Ks {tex_prefix}_mg.dds\n")

            if _tex_exists("_disp"):
                f.write(f"disp {tex_prefix}_disp.dds\n")

            f.write("\n")


def write_obj(meshes, obj_path: str, mtl_filename: str):
    """Write an OBJ file from a list of legacy Mesh objects.

    UV V-flip (1.0 - v) is applied here in the exporter.
    Uses :.6f fixed precision for all floats.
    """
    with open(obj_path, 'w') as f:
        f.write(f"# Crimson Desert PAC export\n")
        f.write(f"mtllib {mtl_filename}\n\n")

        vert_offset = 0

        for mesh in meshes:
            f.write(f"o {mesh.name}\n")
            f.write(f"usemtl {mesh.material}\n")

            for v in mesh.vertices:
                x, y, z = v.pos
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

            for v in mesh.vertices:
                nx, ny, nz = v.normal
                f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

            for v in mesh.vertices:
                u, v_coord = v.uv
                f.write(f"vt {u:.6f} {1.0 - v_coord:.6f}\n")

            for i in range(0, len(mesh.indices), 3):
                i0 = mesh.indices[i] + vert_offset + 1
                i1 = mesh.indices[i + 1] + vert_offset + 1
                i2 = mesh.indices[i + 2] + vert_offset + 1
                f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

            vert_offset += len(mesh.vertices)
            f.write("\n")


# ---- Fast numpy OBJ writer (for ParsedModel, skips Vertex intermediate) ----


def write_obj_from_model(model: ParsedModel, obj_path: str, mtl_filename: str, lod: int = 0):
    """Write OBJ directly from ParsedModel using numpy. Much faster for batch export."""
    import io
    import numpy as np

    buf = io.StringIO()
    buf.write(f"# Crimson Desert PAC export\nmtllib {mtl_filename}\n\n")

    vert_offset = 0
    for sm in model.submeshes:
        geom = sm.get_geometry(lod)
        if geom is None:
            continue
        vb, ib = geom
        n = vb.count

        buf.write(f"o {sm.name}\nusemtl {sm.material_name}\n")

        # Positions
        pos_buf = io.BytesIO()
        np.savetxt(pos_buf, vb.positions, fmt='v %.6f %.6f %.6f')
        buf.write(pos_buf.getvalue().decode())

        # Normals
        nor_buf = io.BytesIO()
        np.savetxt(nor_buf, vb.normals, fmt='vn %.6f %.6f %.6f')
        buf.write(nor_buf.getvalue().decode())

        # UVs with V-flip
        uvs_flipped = vb.uvs.copy()
        uvs_flipped[:, 1] = 1.0 - uvs_flipped[:, 1]
        uv_buf = io.BytesIO()
        np.savetxt(uv_buf, uvs_flipped, fmt='vt %.6f %.6f')
        buf.write(uv_buf.getvalue().decode())

        # Faces
        tri = ib.indices.reshape(-1, 3).astype(np.int64) + vert_offset + 1
        for a, b, c in tri:
            buf.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")

        vert_offset += n
        buf.write("\n")

    with open(obj_path, 'w') as f:
        f.write(buf.getvalue())


# ---- Plugin class ----


class ObjExporter(MeshExporter):
    format_id = "obj"
    format_name = "Wavefront OBJ"
    file_extension = ".obj"

    def export_to_disk(self, model: ParsedModel, output_dir: str,
                       name_hint: str = "", texture_rel_dir: str = "",
                       available_textures: set = None,
                       diffuse_overrides: dict = None) -> ExportResult:
        from dataclasses import dataclass, field as dfield

        warnings = []

        # Convert ParsedModel submeshes (LOD 0) into lightweight mesh dicts
        # for the standalone write_obj/write_mtl functions.
        @dataclass
        class _Mesh:
            name: str
            material: str
            vertices: list = dfield(default_factory=list)
            indices: list = dfield(default_factory=list)

        @dataclass
        class _Vertex:
            pos: tuple
            uv: tuple
            normal: tuple

        meshes = []
        for sm in model.submeshes:
            lod = sm.best_lod()
            geom = sm.get_geometry(lod)
            if geom is None:
                warnings.append(ExportWarning("warning", "geometry",
                                              f"Submesh '{sm.name}' has no geometry"))
                continue
            vb, ib = geom
            verts = []
            for i in range(vb.count):
                verts.append(_Vertex(
                    pos=tuple(float(x) for x in vb.positions[i]),
                    uv=tuple(float(x) for x in vb.uvs[i]),
                    normal=tuple(float(x) for x in vb.normals[i]),
                ))
            meshes.append(_Mesh(
                name=sm.name,
                material=sm.material_name,
                vertices=verts,
                indices=[int(x) for x in ib.indices],
            ))

        if not meshes:
            return ExportResult(success=False, warnings=[
                ExportWarning("error", "geometry", "No meshes with geometry found")])

        base_name = name_hint or meshes[0].name.lower()
        base_name = base_name.replace(' ', '_')
        obj_filename = base_name + '.obj'
        mtl_filename = base_name + '.mtl'

        os.makedirs(output_dir, exist_ok=True)
        obj_path = os.path.join(output_dir, obj_filename)
        mtl_path = os.path.join(output_dir, mtl_filename)

        write_obj(meshes, obj_path, mtl_filename)
        write_mtl(meshes, mtl_path, texture_rel_dir,
                  available_textures=available_textures,
                  diffuse_overrides=diffuse_overrides)

        total_verts = sum(len(m.vertices) for m in meshes)
        total_tris = sum(len(m.indices) // 3 for m in meshes)

        # Count expected vs available textures
        tex_expected = 0
        tex_found = 0
        for sm in model.submeshes:
            if sm.texture_basename:
                tex_expected += 1
                if available_textures and f"{sm.texture_basename}.dds" in available_textures:
                    tex_found += 1

        return ExportResult(
            success=True,
            output_files=[obj_path, mtl_path],
            stats={
                'meshes': len(meshes),
                'vertices': total_verts,
                'triangles': total_tris,
                'names': [m.name for m in meshes],
            },
            warnings=warnings,
            textures_extracted=tex_found,
            textures_expected=tex_expected,
        )
