"""
An artifact represents a resource (e.g., HuggingFace repo) used by a pipeline.
"""

from pydantic import BaseModel


class Artifact(BaseModel):
    """Base class for all artifacts."""

    pass


class HuggingfaceRepoArtifact(Artifact):
    """
    Represents a HuggingFace repo artifact.

    Attributes:
        repo_id: HuggingFace repository ID (e.g., "Wan-AI/Wan2.1-T2V-1.3B")
        files: List of files or directories to download
                Directories should be specified by their name (e.g., "google", "models")
    """

    repo_id: str
    files: list[str]
