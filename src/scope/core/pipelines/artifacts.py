"""
An artifact represents a resource (e.g., HuggingFace repo, Google Drive file) used by a pipeline.
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


class GoogleDriveArtifact(Artifact):
    """
    Represents a Google Drive file artifact.

    Attributes:
        file_id: Google Drive file ID (extracted from share link)
        files: List of filenames to extract from the downloaded file/ZIP (optional).
               If not specified, downloads the file directly using file_id as filename.
        name: Subdirectory name within models directory (optional)
    """

    file_id: str
    files: list[str] | None = None
    name: str | None = None
