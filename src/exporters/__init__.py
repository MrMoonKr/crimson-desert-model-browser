"""Mesh export format plugins.

To add a new format: create a new file (e.g. gltf_exporter.py),
subclass MeshExporter, implement export_to_disk(), and register below.
"""

from exporters.base import MeshExporter, ExportResult, ExportWarning, ExportOptions
from exporters.obj_exporter import ObjExporter

_EXPORTERS = {
    'obj': ObjExporter,
}


def get_exporter(format_id: str) -> MeshExporter:
    if format_id not in _EXPORTERS:
        raise ValueError(f"Unknown format: {format_id}. Available: {list(_EXPORTERS)}")
    return _EXPORTERS[format_id]()


def available_formats() -> list[str]:
    return list(_EXPORTERS.keys())
