"""Shared DDS texture extraction, fixing, and dye color compositing."""

import os
import sys
import tempfile

from pac_export import fix_truncated_dds, parse_pac_xml_colors, generate_dyed_texture
from pac_parser import material_to_dds_basename

DEFAULT_SUFFIXES = ['', '_n', '_sp', '_m', '_mg', '_ma', '_disp', '_o']


class TextureService:
    """Extracts, fixes, and caches DDS textures from PAZ archives.

    Builds a basename->PazEntry index on first use for O(1) lookups
    (vs the current O(n) linear scan per texture).
    """

    def __init__(self, game_dir: str, cached_entries: list = None):
        self._game_dir = game_dir
        self._entries = cached_entries
        self._basename_index = None  # dict[str, list[PazEntry]] -- built lazily

    def _ensure_index(self):
        """Build basename -> [PazEntry] index once."""
        if self._basename_index is not None:
            return
        if self._entries is None:
            src_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(src_dir)
            unpacker_dir = os.path.join(root_dir, "lazorr410-unpacker", "python")
            if unpacker_dir not in sys.path:
                sys.path.insert(0, unpacker_dir)
            from paz_parse import parse_pamt
            dir_0009 = os.path.join(self._game_dir, "0009")
            self._entries = parse_pamt(os.path.join(dir_0009, "0.pamt"), paz_dir=dir_0009)
        self._basename_index = {}
        for e in self._entries:
            bn = os.path.basename(e.path).lower()
            self._basename_index.setdefault(bn, []).append(e)

    def _lookup(self, dds_name: str):
        """Return list of PazEntry matching a DDS filename (case-insensitive)."""
        self._ensure_index()
        return self._basename_index.get(dds_name.lower(), [])

    def _extract_and_flatten(self, paz_extract_entry, entry, output_dir):
        """Extract a single entry, flatten nested dirs, fix truncated DDS."""
        paz_extract_entry(entry, output_dir, decrypt_xml=False)
        nested = os.path.join(output_dir, entry.path.replace('/', os.sep))
        flat = os.path.join(output_dir, os.path.basename(entry.path))
        if os.path.exists(nested) and nested != flat:
            os.replace(nested, flat)
            try:
                d = os.path.dirname(nested)
                while d != output_dir:
                    os.rmdir(d)
                    d = os.path.dirname(d)
            except OSError:
                pass
        if os.path.exists(flat):
            with open(flat, 'rb') as f:
                raw = f.read()
            fixed = fix_truncated_dds(raw)
            if len(fixed) != len(raw):
                with open(flat, 'wb') as f:
                    f.write(fixed)

    def extract_textures(self, texture_basenames: list[str], output_dir: str,
                         suffixes: list[str] = None,
                         progress_fn=None) -> tuple[set[str], int]:
        """Extract and fix DDS textures to disk.

        Args:
            texture_basenames: DDS base names (without suffix/extension,
                e.g. "cd_phw_00_ub_00_0163")
            output_dir: directory to write .dds files
            suffixes: texture suffixes to try (default: standard set)
            progress_fn: optional callback for status messages

        Returns:
            (available_set, extracted_count) -- set of lowercase DDS filenames
            that were successfully extracted, and total count
        """
        from paz_unpack import extract_entry as paz_extract_entry

        if suffixes is None:
            suffixes = DEFAULT_SUFFIXES

        if progress_fn:
            progress_fn("Extracting textures...")

        dds_wanted = set()
        for bn in texture_basenames:
            for suffix in suffixes:
                dds_wanted.add(bn + suffix + '.dds')

        os.makedirs(output_dir, exist_ok=True)

        available = set()
        extracted = 0
        for dds_name in dds_wanted:
            matches = self._lookup(dds_name)
            for m in matches:
                try:
                    self._extract_and_flatten(paz_extract_entry, m, output_dir)
                    available.add(dds_name.lower())
                    extracted += 1
                except Exception:
                    pass

        return available, extracted

    def extract_dds_files(self, dds_filenames: list[str], output_dir: str,
                          progress_fn=None) -> tuple[set[str], int]:
        """Extract pre-built DDS filenames (already include suffix+extension).

        Use this for PAM textures where the full DDS name is already known.

        Returns:
            (available_set, extracted_count)
        """
        from paz_unpack import extract_entry as paz_extract_entry

        if progress_fn:
            progress_fn("Extracting textures...")

        os.makedirs(output_dir, exist_ok=True)

        available = set()
        extracted = 0
        for dds_name in dds_filenames:
            matches = self._lookup(dds_name)
            for m in matches:
                try:
                    self._extract_and_flatten(paz_extract_entry, m, output_dir)
                    available.add(dds_name.lower())
                    extracted += 1
                except Exception:
                    pass

        return available, extracted

    def apply_dye_colors(self, model_name: str,
                         submesh_names_materials: list[tuple[str, str]],
                         tex_dir: str, progress_fn=None) -> dict[str, str]:
        """Find pac.xml, parse colors, generate dyed diffuse PNGs.

        Args:
            model_name: PAC filename without extension (for pac.xml lookup)
            submesh_names_materials: list of (display_name, material_name) tuples
            tex_dir: directory containing extracted _ma.dds files

        Returns:
            {material_name: png_filename} of generated diffuse overrides
        """
        from paz_unpack import extract_entry as paz_extract_entry

        if progress_fn:
            progress_fn("Applying dye colors...")

        xml_name = model_name + '.pac.xml'
        xml_matches = self._lookup(xml_name)
        if not xml_matches:
            return {}

        try:
            with tempfile.TemporaryDirectory() as tmp:
                paz_extract_entry(xml_matches[0], tmp, decrypt_xml=True)
                xml_path = os.path.join(tmp, xml_matches[0].path.replace('/', os.sep))
                with open(xml_path, 'rb') as xf:
                    xml_data = xf.read()
        except Exception:
            return {}

        submesh_colors = parse_pac_xml_colors(xml_data)
        colors_lower = {k.lower(): v for k, v in submesh_colors.items()}

        diffuse_overrides = {}
        seen_materials = set()
        for display_name, material_name in submesh_names_materials:
            if material_name == "(null)" or material_name in seen_materials:
                continue
            seen_materials.add(material_name)

            dn = display_name.lower()
            # Exact match first
            colors = colors_lower.get(dn)
            if not colors:
                # Prefix match: pick entry with the most color channels
                best, best_n = None, 0
                for xname, xcolors in colors_lower.items():
                    if xname.startswith(dn) and len(xcolors) > best_n:
                        best, best_n = xcolors, len(xcolors)
                colors = best
            if not colors:
                continue

            dds_base = material_to_dds_basename(material_name)
            ma_file = os.path.join(tex_dir, dds_base + '_ma.dds')
            if not os.path.exists(ma_file):
                continue
            png_name = dds_base + '_diffuse.png'
            png_path = os.path.join(tex_dir, png_name)
            if generate_dyed_texture(ma_file, colors, png_path):
                diffuse_overrides[material_name] = png_name

        return diffuse_overrides
