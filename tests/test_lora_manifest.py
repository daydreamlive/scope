from datetime import UTC, datetime
from pathlib import Path

from scope.core.lora.manifest import (
    LoRAManifest,
    LoRAManifestEntry,
    LoRAProvenance,
    compute_sha256,
    load_manifest,
    save_manifest,
)


def test_manifest_round_trip(tmp_path: Path):
    entry = LoRAManifestEntry(
        filename="test.safetensors",
        provenance=LoRAProvenance(
            source="huggingface", repo_id="user/repo", hf_filename="test.safetensors"
        ),
        sha256="abc123",
        size_bytes=1024,
        added_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    manifest = LoRAManifest(entries={"test.safetensors": entry})

    save_manifest(tmp_path, manifest)
    loaded = load_manifest(tmp_path)

    assert loaded.version == "1.0"
    assert "test.safetensors" in loaded.entries
    assert loaded.entries["test.safetensors"].sha256 == "abc123"
    assert loaded.entries["test.safetensors"].provenance.source == "huggingface"
    assert loaded.entries["test.safetensors"].provenance.repo_id == "user/repo"


def test_load_manifest_missing_file(tmp_path: Path):
    manifest = load_manifest(tmp_path)
    assert manifest.entries == {}


def test_load_manifest_corrupted_file(tmp_path: Path):
    (tmp_path / "lora_manifest.json").write_text("not json", encoding="utf-8")
    manifest = load_manifest(tmp_path)
    assert manifest.entries == {}


def test_compute_sha256(tmp_path: Path):
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"hello world")
    result = compute_sha256(test_file)
    assert result == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_save_manifest_creates_directory(tmp_path: Path):
    new_dir = tmp_path / "subdir"
    manifest = LoRAManifest()
    save_manifest(new_dir, manifest)
    assert (new_dir / "lora_manifest.json").exists()


def test_provenance_sources():
    for source in ("huggingface", "civitai", "url", "local"):
        prov = LoRAProvenance(source=source)
        assert prov.source == source


def test_add_manifest_entry(tmp_path: Path):
    from scope.core.lora.manifest import add_manifest_entry

    test_file = tmp_path / "model.safetensors"
    test_file.write_bytes(b"test data")

    provenance = LoRAProvenance(source="huggingface", repo_id="user/repo")
    entry = add_manifest_entry(tmp_path, "model.safetensors", provenance, "abc123", 9)

    assert entry.filename == "model.safetensors"
    assert entry.provenance.source == "huggingface"
    assert entry.sha256 == "abc123"
    assert entry.size_bytes == 9
    assert entry.added_at is not None

    # Verify it was persisted
    loaded = load_manifest(tmp_path)
    assert "model.safetensors" in loaded.entries
    assert loaded.entries["model.safetensors"].sha256 == "abc123"
