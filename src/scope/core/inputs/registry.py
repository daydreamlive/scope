"""Registry for input sources."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import InputSource

logger = logging.getLogger(__name__)


class InputSourceRegistry:
    """Registry for available input source implementations."""

    def __init__(self):
        self._sources: dict[str, type["InputSource"]] = {}

    def register(self, source_class: type["InputSource"]):
        """Register an input source class.

        Args:
            source_class: The InputSource subclass to register.
        """
        source_id = source_class.source_id
        if source_id in self._sources:
            logger.warning(f"Input source '{source_id}' already registered, overwriting")
        self._sources[source_id] = source_class
        logger.info(f"Registered input source: {source_id} ({source_class.source_name})")

    def get(self, source_id: str) -> type["InputSource"] | None:
        """Get an input source class by ID.

        Args:
            source_id: The unique identifier of the input source.

        Returns:
            The input source class, or None if not found.
        """
        return self._sources.get(source_id)

    def list_available(self) -> list[dict]:
        """List all available input sources.

        Returns:
            List of dicts with source metadata, filtered to only include
            sources that are available on this platform.
        """
        available = []
        for source_id, source_class in self._sources.items():
            try:
                if source_class.is_available():
                    available.append(
                        {
                            "source_id": source_id,
                            "source_name": source_class.source_name,
                            "source_description": source_class.source_description,
                        }
                    )
            except Exception as e:
                logger.warning(f"Error checking availability for {source_id}: {e}")
        return available

    def create_instance(self, source_id: str) -> "InputSource | None":
        """Create an instance of an input source.

        Args:
            source_id: The unique identifier of the input source.

        Returns:
            An instance of the input source, or None if not found/available.
        """
        source_class = self.get(source_id)
        if source_class is None:
            logger.error(f"Input source '{source_id}' not found")
            return None

        if not source_class.is_available():
            logger.error(f"Input source '{source_id}' is not available on this platform")
            return None

        try:
            return source_class()
        except Exception as e:
            logger.error(f"Failed to create input source '{source_id}': {e}")
            return None


# Global registry instance
input_source_registry = InputSourceRegistry()

