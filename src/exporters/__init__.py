"""Mesh export format plugins."""

from exporters.base import MeshExporter, ExportResult, ExportWarning
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
