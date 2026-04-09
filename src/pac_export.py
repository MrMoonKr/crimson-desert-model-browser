"""Export Crimson Desert PAC skinned meshes to OBJ + MTL.

Parsing logic lives in pac_parser.py. This module provides:
- OBJ/MTL writing (write_obj, write_mtl)
- DDS texture fixing (fix_truncated_dds)
- Dye color extraction and compositing
- CLI entry point
- Re-exports for backward compatibility
"""

import os
import sys
import struct
import argparse
from dataclasses import dataclass, field

try:
    import lz4.block
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

# Re-exports from new modules (tests and pac_browser import these from here)
from pac_parser import (
    PacParser, decompress_type1_pac, material_to_dds_basename,
)
from pac_decode import decode_pac_vertices, decode_indices as _decode_indices_np
from model_types import VertexBuffer, IndexBuffer, BoundingBox, SourceFormat

_parser = PacParser()

# Re-export parser internals under old names for backward compatibility
parse_header = _parser._parse_header
find_mesh_descriptors = _parser._find_mesh_descriptors
_find_section_layout = _parser._find_section_layout
_find_name_strings = _parser._find_name_strings


def decode_vertices(data, section_offset, vertex_count, desc, vertex_start=0):
    """Legacy wrapper returning list[Vertex]. Used by tests and old code."""
    vb = decode_pac_vertices(data, section_offset, vertex_count,
                             desc.center, desc.half_extent, vertex_start)
    verts = []
    for i in range(vb.count):
        verts.append(Vertex(
            pos=tuple(float(x) for x in vb.positions[i]),
            uv=tuple(float(x) for x in vb.uvs[i]),
            normal=tuple(float(x) for x in vb.normals[i]),
        ))
    return verts


def decode_indices(data, section_offset, index_count,
                   total_verts_before=0, index_start=0):
    """Legacy wrapper returning list[int]. Used by tests and old code."""
    ib = _decode_indices_np(data, section_offset + index_start, index_count)
    return [int(x) for x in ib.indices]


# ── Data structures (legacy, still used by write_obj/write_mtl) ──

@dataclass
class MeshDescriptor:
    display_name: str
    material_name: str
    center: tuple
    half_extent: tuple
    vertex_counts: list
    index_counts: list
    bbox_unknowns: tuple = ()


@dataclass
class Vertex:
    pos: tuple
    uv: tuple
    normal: tuple


@dataclass
class Mesh:
    name: str
    material: str
    vertices: list = field(default_factory=list)
    indices: list = field(default_factory=list)


# ── DDS per-mip decompression (type 1 DDS) ──

def fix_truncated_dds(data: bytes) -> bytes:
    """Decompress type 1 DDS files that use per-mip LZ4 compression."""
    if len(data) < 128 or data[:4] != b'DDS ':
        return data

    height = struct.unpack_from('<I', data, 12)[0]
    width = struct.unpack_from('<I', data, 16)[0]
    mips = struct.unpack_from('<I', data, 28)[0]
    fourcc = data[84:88]

    BPB = {
        b'DXT1': 8,  b'DXT3': 16, b'DXT5': 16,
        b'BC4U': 8,  b'BC4S': 8,  b'ATI1': 8,
        b'BC5U': 16, b'BC5S': 16, b'ATI2': 16,
    }
    bpb = BPB.get(fourcc)
    if bpb is None or mips < 1:
        return data

    mip_decomp = []
    w, h = width, height
    for _ in range(mips):
        bw, bh = max(1, w // 4), max(1, h // 4)
        mip_decomp.append(bw * bh * bpb)
        w, h = max(1, w // 2), max(1, h // 2)

    stored = [struct.unpack_from('<I', data, 32 + i * 4)[0] for i in range(4)]

    needs_decompress = False
    for i in range(min(4, mips)):
        if stored[i] > 0 and stored[i] < mip_decomp[i]:
            needs_decompress = True
            break

    if not needs_decompress:
        return data

    import lz4.block
    hdr = bytearray(data[:128])
    for i in range(11):
        struct.pack_into('<I', hdr, 32 + i * 4, 0)

    output = bytearray(hdr)
    offset = 128
    for m in range(mips):
        decomp_size = mip_decomp[m]
        stored_size = stored[m] if m < 4 and stored[m] > 0 else decomp_size
        chunk = data[offset:offset + stored_size]
        if stored_size < decomp_size:
            output.extend(lz4.block.decompress(chunk, uncompressed_size=decomp_size))
        else:
            output.extend(chunk)
        offset += stored_size

    return bytes(output)


# ── Dye color extraction and compositing ──

def parse_pac_xml_colors(xml_data: bytes) -> dict:
    """Parse decrypted pac.xml to extract per-submesh tint colors."""
    import xml.etree.ElementTree as ET
    text = xml_data.lstrip(b'\xef\xbb\xbf')
    root = ET.fromstring(b'<root>' + text + b'</root>')
    result = {}

    for wrapper in root.iter("SkinnedMeshMaterialWrapper"):
        sub_name = wrapper.get("_subMeshName")
        if sub_name is None:
            continue

        tint = {}
        detail = {}
        for param in wrapper.iter("MaterialParameterColor"):
            name = param.get("_name", "")
            value = param.get("_value", "")
            if not value.startswith("#") or len(value) < 7:
                continue
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
            if name.startswith("_tintColor") and name[-1] in "RGB":
                tint[name[-1]] = (r, g, b)
            elif name.startswith("_dyeingDetailLayerColorMask") and name[-1] in "RGB":
                detail[name[-1]] = (r, g, b)

        colors = tint or detail
        if colors:
            result[sub_name] = colors

    return result


def generate_dyed_texture(ma_path: str, tint_colors: dict,
                          output_path: str) -> bool:
    """Composite dye mask + tint colors into a diffuse texture."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return False

    img = Image.open(ma_path)
    arr = np.array(img.convert("RGBA"), dtype=np.float32)

    mask_r = arr[:, :, 0]
    mask_g = arr[:, :, 1]
    mask_b = arr[:, :, 2]

    total = mask_r + mask_g + mask_b
    total = np.maximum(total, 1.0)
    mask_r = mask_r / total
    mask_g = mask_g / total
    mask_b = mask_b / total

    h, w = mask_r.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    for channel, mask in [("R", mask_r), ("G", mask_g), ("B", mask_b)]:
        if channel in tint_colors:
            tr, tg, tb = tint_colors[channel]
            out[:, :, 0] += mask * tr
            out[:, :, 1] += mask * tg
            out[:, :, 2] += mask * tb

    out = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out, "RGB").save(output_path)
    return True


# ── OBJ + MTL writing ──

def write_mtl(meshes: list[Mesh], mtl_path: str, texture_rel_dir: str = "",
              available_textures: set = None, diffuse_overrides: dict = None):
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

            def _tex_exists(suffix):
                name = f"{dds_base}{suffix}.dds"
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


def write_obj(meshes: list[Mesh], obj_path: str, mtl_filename: str):
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


# ── Main export logic ──

def export_pac(pac_data: bytes, output_dir: str, name_hint: str = "",
               texture_rel_dir: str = "", lod: int = 0) -> dict:
    """Export a PAC file to OBJ + MTL using the new parser."""
    model = _parser.parse(pac_data, lods=[lod])

    meshes = []
    for sm in model.submeshes:
        geom = sm.get_geometry(lod)
        if geom is None:
            continue
        vb, ib = geom
        verts = []
        for i in range(vb.count):
            verts.append(Vertex(
                pos=tuple(float(x) for x in vb.positions[i]),
                uv=tuple(float(x) for x in vb.uvs[i]),
                normal=tuple(float(x) for x in vb.normals[i]),
            ))
        meshes.append(Mesh(
            name=sm.name,
            material=sm.material_name,
            vertices=verts,
            indices=[int(x) for x in ib.indices],
        ))

    if not meshes:
        raise ValueError("No meshes with geometry found")

    base_name = name_hint or meshes[0].name.lower()
    base_name = base_name.replace(' ', '_')
    obj_filename = base_name + '.obj'
    mtl_filename = base_name + '.mtl'

    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, obj_filename)
    mtl_path = os.path.join(output_dir, mtl_filename)

    write_obj(meshes, obj_path, mtl_filename)
    write_mtl(meshes, mtl_path, texture_rel_dir)

    total_verts_out = sum(len(m.vertices) for m in meshes)
    total_tris = sum(len(m.indices) // 3 for m in meshes)

    return {
        'obj': obj_path,
        'mtl': mtl_path,
        'meshes': len(meshes),
        'vertices': total_verts_out,
        'triangles': total_tris,
        'names': [m.name for m in meshes],
    }


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Export Crimson Desert PAC meshes to OBJ + MTL")
    parser.add_argument("pac_file", nargs='?', help="Path to .pac file on disk")
    parser.add_argument("-o", "--output", default=".", help="Output directory (default: current)")
    parser.add_argument("--name", help="Output filename base (default: from mesh name)")
    parser.add_argument("--textures", default="", help="Relative path from OBJ to textures dir")
    parser.add_argument("--lod", type=int, default=0, choices=[0,1,2,3],
                        help="LOD level: 0=highest detail (default), 3=lowest")

    parser.add_argument("--paz-dir", help="Game directory with 0.pamt (e.g. .../0009)")
    parser.add_argument("--filter", help="Filter PAC files by glob/substring in PAZ mode")
    parser.add_argument("--batch", action="store_true",
                        help="Export all matching files (not just first match)")

    args = parser.parse_args()

    if args.pac_file:
        with open(args.pac_file, 'rb') as f:
            pac_data = f.read()

        name = args.name or os.path.splitext(os.path.basename(args.pac_file))[0]
        result = export_pac(pac_data, args.output, name_hint=name,
                           texture_rel_dir=args.textures, lod=args.lod)

        print(f"Exported {result['meshes']} mesh(es): {result['vertices']} verts, {result['triangles']} tris")
        for n in result['names']:
            print(f"  - {n}")
        print(f"  {result['obj']}")
        print(f"  {result['mtl']}")

    elif args.paz_dir:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lazorr410-unpacker', 'python'))
            from paz_parse import parse_pamt
            from paz_unpack import extract_entry
        except ImportError:
            print("Error: paz_parse/paz_unpack not found. Ensure lazorr410-unpacker is cloned.", file=sys.stderr)
            sys.exit(1)

        pamt_path = os.path.join(args.paz_dir, '0.pamt')
        if not os.path.exists(pamt_path):
            print(f"Error: {pamt_path} not found", file=sys.stderr)
            sys.exit(1)

        print(f"Parsing {pamt_path}...")
        entries = parse_pamt(pamt_path, paz_dir=args.paz_dir)

        import fnmatch
        pattern = (args.filter or "*.pac").lower()
        matches = [
            e for e in entries
            if e.path.lower().endswith('.pac')
            and (not e.compressed or e.compression_type == 1)
            and (fnmatch.fnmatch(e.path.lower(), f"*{pattern}*")
                 or fnmatch.fnmatch(os.path.basename(e.path).lower(), pattern)
                 or pattern in e.path.lower())
        ]

        if not matches:
            print(f"No PAC files matching '{args.filter}'")
            sys.exit(1)

        if not args.batch:
            matches = matches[:1]

        print(f"Exporting {len(matches)} PAC file(s)...\n")

        for entry in matches:
            try:
                read_size = entry.comp_size if entry.compressed else entry.orig_size
                with open(entry.paz_file, 'rb') as f:
                    f.seek(entry.offset)
                    pac_data = f.read(read_size)

                if entry.compressed and entry.compression_type == 1:
                    pac_data = decompress_type1_pac(pac_data, entry.orig_size)

                pac_name = os.path.splitext(os.path.basename(entry.path))[0]
                result = export_pac(pac_data, args.output, name_hint=args.name or pac_name,
                                   texture_rel_dir=args.textures, lod=args.lod)

                print(f"{pac_name}: {result['meshes']} mesh(es), {result['vertices']} verts, {result['triangles']} tris")
                for n in result['names']:
                    print(f"    {n}")

            except Exception as e:
                print(f"  ERROR {os.path.basename(entry.path)}: {e}", file=sys.stderr)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
