"""Base class and result types for mesh exporters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from model_types import ParsedModel


@dataclass
class ExportWarning:
    severity: str    # "info", "warning", "error"
    source: str      # "texture", "geometry", "material"
    message: str


@dataclass
class ExportResult:
    success: bool
    output_files: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    warnings: list[ExportWarning] = field(default_factory=list)
    textures_extracted: int = 0
    textures_expected: int = 0


class MeshExporter(ABC):
    format_id: str = ""
    format_name: str = ""
    file_extension: str = ""

    @abstractmethod
    def export_to_disk(self, model: ParsedModel, output_dir: str,
                       name_hint: str = "", texture_rel_dir: str = "",
                       available_textures: set = None,
                       diffuse_overrides: dict = None) -> ExportResult:
        ...

    def export_to_bytes(self, model: ParsedModel, **kwargs) -> bytes:
        raise NotImplementedError(f"{self.format_name} doesn't support in-memory export")
