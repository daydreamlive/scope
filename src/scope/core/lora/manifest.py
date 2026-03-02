import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "lora_manifest.json"


class LoRAProvenance(BaseModel):
    source: Literal["huggingface", "civitai", "url", "local"]
    repo_id: str | None = None
    hf_filename: str | None = None
    model_id: str | None = None
    version_id: str | None = None
    url: str | None = None


class LoRAManifestEntry(BaseModel):
    filename: str
    provenance: LoRAProvenance
    sha256: str
    size_bytes: int
    added_at: datetime


class LoRAManifest(BaseModel):
    version: str = "1.0"
    entries: dict[str, LoRAManifestEntry] = {}


def load_manifest(lora_dir: Path) -> LoRAManifest:
    manifest_path = lora_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return LoRAManifest()
    try:
        data = manifest_path.read_text(encoding="utf-8")
        return LoRAManifest.model_validate_json(data)
    except Exception:
        logger.warning("Failed to load LoRA manifest, starting fresh", exc_info=True)
        return LoRAManifest()


def save_manifest(lora_dir: Path, manifest: LoRAManifest) -> None:
    lora_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = lora_dir / MANIFEST_FILENAME
    tmp_path = manifest_path.with_suffix(".tmp")
    tmp_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(manifest_path)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def add_manifest_entry(
    lora_dir: Path,
    filename: str,
    provenance: LoRAProvenance,
    sha256: str,
    size_bytes: int,
) -> LoRAManifestEntry:
    entry = LoRAManifestEntry(
        filename=filename,
        provenance=provenance,
        sha256=sha256,
        size_bytes=size_bytes,
        added_at=datetime.now(UTC),
    )
    manifest = load_manifest(lora_dir)
    manifest.entries[filename] = entry
    save_manifest(lora_dir, manifest)
    return entry
