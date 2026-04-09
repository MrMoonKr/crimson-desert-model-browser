"""Export Crimson Desert PAM static meshes to OBJ + MTL.

Parsing logic lives in pam_parser.py. This module provides:
- CLI entry point
- Re-exports for backward compatibility (pac_browser imports from here)
"""

import os
import sys
import struct
import argparse

from pac_export import Vertex, Mesh, write_obj, write_mtl
from pam_parser import PamParser, decompress_pam_geometry

_parser = PamParser()

# Re-exports for backward compatibility (pac_browser.py imports these)
parse_pam_header = _parser._parse_header
parse_pam_submeshes = _parser._parse_submeshes
detect_vertex_stride = _parser._detect_stride

PAM_VERSIONS = PamParser.VERSIONS
PAC_VERSION = PamParser.PAC_VERSION


def decode_pam_vertices(data, geom_off, byte_offset, count, bbox_min, bbox_max, stride=20):
    """Legacy wrapper returning list[Vertex]."""
    from pac_decode import decode_pam_vertices as _decode_np
    vb = _decode_np(data, geom_off, byte_offset, count, bbox_min, bbox_max, stride)
    verts = []
    for i in range(vb.count):
        verts.append(Vertex(
            pos=tuple(float(x) for x in vb.positions[i]),
            uv=tuple(float(x) for x in vb.uvs[i]),
            normal=tuple(float(x) for x in vb.normals[i]),
        ))
    return verts


def decode_pam_indices(data, byte_offset, count):
    """Legacy wrapper returning list[int]."""
    return [struct.unpack_from('<H', data, byte_offset + i * 2)[0] for i in range(count)]


def export_pam(pam_data: bytes, output_dir: str, name_hint: str = "",
               texture_rel_dir: str = "", available_textures: set = None) -> dict:
    """Export a PAM file to OBJ + MTL using the new parser."""
    model = _parser.parse(pam_data)

    meshes = []
    for sm in model.submeshes:
        geom = sm.get_geometry(0)
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

    base_name = name_hint or meshes[0].name.lower().replace(' ', '_')
    obj_filename = base_name + '.obj'
    mtl_filename = base_name + '.mtl'

    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, obj_filename)
    mtl_path = os.path.join(output_dir, mtl_filename)

    write_obj(meshes, obj_path, mtl_filename)
    write_mtl(meshes, mtl_path, texture_rel_dir, available_textures=available_textures)

    return {
        'obj': obj_path, 'mtl': mtl_path,
        'meshes': len(meshes),
        'vertices': sum(len(m.vertices) for m in meshes),
        'triangles': sum(len(m.indices) // 3 for m in meshes),
        'names': [m.name for m in meshes],
    }


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Export Crimson Desert PAM meshes to OBJ + MTL")
    parser.add_argument("pam_file", nargs='?', help="Path to .pam file on disk")
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("--name", help="Output filename base")
    parser.add_argument("--textures", default="", help="Relative path from OBJ to textures dir")
    parser.add_argument("--paz-dir", help="Game directory with 0.pamt")
    parser.add_argument("--filter", help="Filter PAM files by substring")
    parser.add_argument("--batch", action="store_true", help="Export all matching files")

    args = parser.parse_args()

    if args.pam_file:
        with open(args.pam_file, 'rb') as f:
            pam_data = f.read()
        name = args.name or os.path.splitext(os.path.basename(args.pam_file))[0]
        result = export_pam(pam_data, args.output, name_hint=name,
                            texture_rel_dir=args.textures)
        print(f"Exported {result['meshes']} mesh(es): {result['vertices']} verts, {result['triangles']} tris")
        for n in result['names']:
            print(f"  - {n}")

    elif args.paz_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lazorr410-unpacker', 'python'))
        from paz_parse import parse_pamt

        pamt_path = os.path.join(args.paz_dir, '0.pamt')
        print(f"Parsing {pamt_path}...")
        entries = parse_pamt(pamt_path, paz_dir=args.paz_dir)

        import fnmatch
        pattern = (args.filter or "*.pam").lower()
        matches = [
            e for e in entries
            if e.path.lower().endswith('.pam')
            and (not e.compressed or e.compression_type == 1)
            and (pattern in e.path.lower()
                 or fnmatch.fnmatch(os.path.basename(e.path).lower(), pattern))
        ]

        if not matches:
            print(f"No PAM files matching '{args.filter}'")
            sys.exit(1)

        if not args.batch:
            matches = matches[:1]

        print(f"Exporting {len(matches)} PAM file(s)...\n")

        for entry in matches:
            try:
                read_size = entry.comp_size if entry.compressed else entry.orig_size
                with open(entry.paz_file, 'rb') as f:
                    f.seek(entry.offset)
                    raw = f.read(read_size)

                pam_name = os.path.splitext(os.path.basename(entry.path))[0]
                result = export_pam(raw, args.output, name_hint=args.name or pam_name,
                                    texture_rel_dir=args.textures)
                print(f"{pam_name}: {result['meshes']} mesh(es), {result['vertices']} verts, {result['triangles']} tris")

            except Exception as e:
                print(f"  ERROR {os.path.basename(entry.path)}: {e}", file=sys.stderr)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
