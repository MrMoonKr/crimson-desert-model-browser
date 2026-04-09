"""Data loading, mesh conversion, and export functions for the model browser."""

import os

import numpy as np

from PySide6.QtCore import QThread, Signal

from pac_parser import PacParser, decompress_type1_pac, material_to_dds_basename
from pam_parser import PamParser, decompress_pam_geometry
from pac_export import Vertex, Mesh, write_obj, write_mtl
from pam_export import export_pam
from model_types import ParsedModel
from texture_service import TextureService
from browser.models import CatalogEntry, GpuMesh, SceneMesh, SubmeshInfo


# ── PAC / PAM byte readers ────────────────────────────────────────

def read_pac_bytes(entry) -> bytes:
    """Read and decompress raw PAC bytes from PAZ archive."""
    read_size = entry.comp_size if entry.compressed else entry.orig_size
    with open(entry.paz_file, 'rb') as f:
        f.seek(entry.offset)
        raw = f.read(read_size)
    if entry.compressed and entry.compression_type == 1:
        raw = decompress_type1_pac(raw, entry.orig_size)
    return raw


def read_pam_bytes(entry) -> bytes:
    """Read PAM bytes from PAZ archive and decompress internal geometry if needed."""
    read_size = entry.comp_size if entry.compressed else entry.orig_size
    with open(entry.paz_file, 'rb') as f:
        f.seek(entry.offset)
        raw = f.read(read_size)
    return decompress_pam_geometry(raw)


# ── Mesh conversion ───────────────────────────────────────────────

def _model_to_gpu_mesh(model: ParsedModel) -> GpuMesh:
    """Convert a ParsedModel into a GpuMesh for OpenGL preview."""
    all_pos, all_nor, all_idx = [], [], []
    offset = 0
    for sm in model.submeshes:
        geom = sm.get_geometry(sm.best_lod())
        if geom is None:
            continue
        vb, ib = geom
        all_pos.append(vb.positions)
        all_nor.append(vb.normals)
        all_idx.append(ib.indices.astype(np.uint32) + offset)
        offset += vb.count

    positions = np.concatenate(all_pos)
    normals = np.concatenate(all_nor)
    indices = np.concatenate(all_idx)

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    radius = float(np.linalg.norm(positions - center, axis=1).max())
    if radius < 1e-6:
        radius = 1.0

    return GpuMesh(positions, normals, indices, center, radius)


def _model_to_scene_mesh(model: ParsedModel) -> SceneMesh:
    """Convert a ParsedModel into a SceneMesh with per-submesh info."""
    all_pos, all_nor, all_uv, all_idx = [], [], [], []
    submesh_infos = []
    vertex_offset = 0
    index_offset = 0

    for sm in model.submeshes:
        geom = sm.get_geometry(sm.best_lod())
        if geom is None:
            continue
        vb, ib = geom
        all_pos.append(vb.positions)
        all_nor.append(vb.normals)
        all_uv.append(vb.uvs)
        idx = ib.indices.astype(np.uint32) + vertex_offset
        all_idx.append(idx)

        submesh_infos.append(SubmeshInfo(
            name=sm.name,
            material_name=sm.material_name,
            index_offset=index_offset * 4,  # byte offset (uint32 = 4 bytes)
            index_count=ib.count,
        ))

        vertex_offset += vb.count
        index_offset += ib.count

    positions = np.concatenate(all_pos)
    normals = np.concatenate(all_nor)
    uvs = np.concatenate(all_uv)
    indices = np.concatenate(all_idx)

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    radius = float(np.linalg.norm(positions - center, axis=1).max())
    if radius < 1e-6:
        radius = 1.0

    return SceneMesh(
        positions=positions,
        normals=normals,
        uvs=uvs,
        indices=indices,
        submeshes=submesh_infos,
        center=center,
        radius=radius,
        available_lods=list(model.available_lods),
        current_lod=model.submeshes[0].best_lod() if model.submeshes else 0,
    )


def load_pac_mesh(entry) -> SceneMesh:
    raw = read_pac_bytes(entry)
    parser = PacParser()
    model = parser.parse(raw, lods=[0])
    return _model_to_scene_mesh(model)


def load_pam_mesh(entry) -> SceneMesh:
    raw = read_pam_bytes(entry)
    parser = PamParser()
    model = parser.parse(raw)
    return _model_to_scene_mesh(model)


# ── Export (OBJ + MTL + DDS textures) ─────────────────────────────

def export_model_with_textures(entry, output_dir: str,
                               game_dir: str, progress_fn=None,
                               cached_entries=None,
                               apply_dye_colors: bool = False) -> dict:
    """Export PAC model as OBJ + MTL + DDS textures via TextureService."""
    pac_data = read_pac_bytes(entry)
    model_name = os.path.splitext(os.path.basename(entry.path))[0]
    model_dir = os.path.join(output_dir, model_name)
    tex_dir = os.path.join(model_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    model = PacParser().parse(pac_data, lods=[0])

    meshes = []
    for sm in model.submeshes:
        geom = sm.get_geometry(sm.best_lod())
        if geom is None:
            continue
        vb, ib = geom
        verts = [Vertex(pos=tuple(float(x) for x in vb.positions[i]),
                        uv=tuple(float(x) for x in vb.uvs[i]),
                        normal=tuple(float(x) for x in vb.normals[i]))
                 for i in range(vb.count)]
        meshes.append(Mesh(name=sm.name, material=sm.material_name,
                           vertices=verts, indices=[int(x) for x in ib.indices]))

    # Extract textures via TextureService
    tex_svc = TextureService(game_dir, cached_entries=cached_entries)
    tex_basenames = [sm.texture_basename for sm in model.submeshes if sm.texture_basename]
    available, extracted = tex_svc.extract_textures(tex_basenames, tex_dir,
                                                     progress_fn=progress_fn)

    # Dye colors via TextureService
    diffuse_overrides = {}
    if apply_dye_colors:
        submesh_pairs = [(sm.name, sm.material_name) for sm in model.submeshes]
        diffuse_overrides = tex_svc.apply_dye_colors(model_name, submesh_pairs,
                                                      tex_dir, progress_fn=progress_fn)

    # Write OBJ + MTL
    if progress_fn:
        progress_fn("Writing OBJ + MTL...")

    obj_path = os.path.join(model_dir, model_name + '.obj')
    mtl_path = os.path.join(model_dir, model_name + '.mtl')
    write_obj(meshes, obj_path, model_name + '.mtl')
    write_mtl(meshes, mtl_path, texture_rel_dir="textures",
              available_textures=available, diffuse_overrides=diffuse_overrides)

    total_verts_out = sum(len(m.vertices) for m in meshes)
    total_tris = sum(len(m.indices) // 3 for m in meshes)

    return {
        'obj': obj_path, 'mtl': mtl_path,
        'meshes': len(meshes), 'vertices': total_verts_out, 'triangles': total_tris,
        'names': [m.name for m in meshes],
        'textures_extracted': extracted,
        'textures_expected': len(set(sm.texture_basename for sm in model.submeshes if sm.texture_basename)),
        'export_dir': model_dir,
    }


def export_pam_with_textures(entry, output_dir: str,
                              game_dir: str, progress_fn=None,
                              cached_entries=None) -> dict:
    """Export PAM model as OBJ + MTL + DDS textures via TextureService."""
    pam_data = read_pam_bytes(entry)
    model_name = os.path.splitext(os.path.basename(entry.path))[0]
    model_dir = os.path.join(output_dir, model_name)
    tex_dir = os.path.join(model_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    # Parse to get texture names
    model = PamParser().parse(pam_data)
    dds_filenames = [sm.texture_basename + '.dds'
                     for sm in model.submeshes if sm.texture_basename]

    # Extract textures via TextureService
    tex_svc = TextureService(game_dir, cached_entries=cached_entries)
    available, extracted = tex_svc.extract_dds_files(dds_filenames, tex_dir,
                                                     progress_fn=progress_fn)

    # Write OBJ + MTL
    if progress_fn:
        progress_fn("Writing OBJ + MTL...")

    result = export_pam(pam_data, model_dir, name_hint=model_name,
                        texture_rel_dir="textures", available_textures=available)
    result['textures_extracted'] = extracted
    result['textures_expected'] = len(set(dds_filenames))
    result['export_dir'] = model_dir
    return result


# ── Background workers ────────────────────────────────────────────

class LoadWorker(QThread):
    mesh_ready = Signal(object)
    load_error = Signal(str)

    def __init__(self, entry, parent=None):
        super().__init__(parent)
        self._entry = entry

    def run(self):
        try:
            if self._entry.file_type == "pam":
                mesh = load_pam_mesh(self._entry.paz_entry)
            else:
                mesh = load_pac_mesh(self._entry.paz_entry)
            self.mesh_ready.emit(mesh)
        except Exception as e:
            self.load_error.emit(f"{self._entry.filename}: {e}")


class ExportWorker(QThread):
    export_done = Signal(dict)
    export_error = Signal(str)
    progress = Signal(str)

    def __init__(self, entry, output_dir, game_dir, cached_entries=None,
                 apply_dye_colors=False, parent=None):
        super().__init__(parent)
        self._entry = entry
        self._output_dir = output_dir
        self._game_dir = game_dir
        self._cached_entries = cached_entries
        self._apply_dye_colors = apply_dye_colors

    def run(self):
        try:
            if self._entry.file_type == "pam":
                result = export_pam_with_textures(
                    self._entry.paz_entry, self._output_dir,
                    self._game_dir, progress_fn=self.progress.emit,
                    cached_entries=self._cached_entries)
            else:
                result = export_model_with_textures(
                    self._entry.paz_entry, self._output_dir,
                    self._game_dir, progress_fn=self.progress.emit,
                    cached_entries=self._cached_entries,
                    apply_dye_colors=self._apply_dye_colors)
            self.export_done.emit(result)
        except Exception as e:
            self.export_error.emit(str(e))
