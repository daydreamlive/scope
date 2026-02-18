"""Plugin manager for discovering and loading Scope plugins."""

import asyncio
import importlib
import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pluggy

from .dependency_validator import DependencyValidator
from .hookspecs import ScopeHookSpec
from .plugins_config import (
    ensure_plugins_dir,
    get_plugins_dir,
    get_plugins_file,
    get_resolved_file,
)

if TYPE_CHECKING:
    from scope.core.pipelines.registry import PipelineRegistry

logger = logging.getLogger(__name__)


def _get_torch_backend_args() -> list[str]:
    """Return torch backend args appropriate for the current platform.

    On Linux/Windows, use CUDA 12.8 backend. On Mac (darwin), omit the flag
    entirely since there are no CUDA wheels available for Mac.
    """
    if sys.platform in ("linux", "win32"):
        return ["--torch-backend", "cu128"]
    return []


class PluginNotFoundError(Exception):
    """Plugin not found."""

    pass


class PluginNotEditableError(Exception):
    """Plugin is not editable."""

    pass


class PluginInUseError(Exception):
    """Plugin pipelines are currently loaded."""

    def __init__(self, message: str, loaded_pipelines: list[str]):
        super().__init__(message)
        self.loaded_pipelines = loaded_pipelines


class PluginNameCollisionError(Exception):
    """Plugin name collision with different source."""

    pass


class PluginDependencyError(Exception):
    """Plugin dependency validation failed."""

    pass


class PluginInstallError(Exception):
    """Plugin installation failed."""

    pass


@dataclass(frozen=True)
class FailedPluginInfo:
    """Information about a plugin entry point that failed to load."""

    package_name: str
    entry_point_name: str
    error_type: str
    error_message: str


class PluginManager:
    """Manager for Scope plugin lifecycle.

    Provides thread-safe operations for discovering, loading, installing,
    uninstalling, and reloading plugins.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._pm = pluggy.PluginManager("scope")
        self._pm.add_hookspecs(ScopeHookSpec)

        # Mapping from pipeline_id to plugin package name
        self._pipeline_to_plugin: dict[str, str] = {}

        # Cache of registered plugin names (package names)
        self._registered_plugins: set[str] = set()

        # Entry points that failed to load
        self._failed_plugins: list[FailedPluginInfo] = []

    def _read_plugins_file(self) -> list[str]:
        """Read plugin specifiers from plugins.txt."""
        plugins_file = get_plugins_file()
        if not plugins_file.exists():
            return []
        return [
            line.strip()
            for line in plugins_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def _write_plugins_file(self, plugins: list[str]) -> None:
        """Write plugin specifiers to plugins.txt."""
        ensure_plugins_dir()
        get_plugins_file().write_text("\n".join(plugins) + "\n" if plugins else "")

    def _generate_constraints(self) -> Path | None:
        """Generate a pip constraints file from uv.lock.

        Finds the root project package in uv.lock (identified by
        ``source.editable``), reads its direct dependency names, then looks up
        each dependency's locked version to produce constraints of the form
        ``name>=locked_version,<next_major`` (e.g. ``transformers>=4.57.5,<5``).
        """
        import tomllib

        lock_file = Path.cwd() / "uv.lock"

        if not lock_file.exists():
            return None

        try:
            lock_text = lock_file.read_text(encoding="utf-8")
            lock_data = tomllib.loads(lock_text)

            packages = lock_data.get("package", [])

            # Find the root project (editable source) and collect all versions
            direct_dep_names: set[str] = set()
            locked_versions: dict[str, str] = {}

            for pkg in packages:
                name = pkg.get("name", "")
                version = pkg.get("version", "")
                source = pkg.get("source", {})

                if name and version:
                    locked_versions[name.lower().replace("-", "_")] = version

                # Root project has source = { editable = "." }
                if isinstance(source, dict) and "editable" in source:
                    for dep in pkg.get("dependencies", []):
                        dep_name = dep.get("name", "")
                        if dep_name:
                            direct_dep_names.add(dep_name.lower().replace("-", "_"))

            if not direct_dep_names:
                return None

            constraints = []
            for norm_name in sorted(direct_dep_names):
                version = locked_versions.get(norm_name)
                if not version:
                    continue

                # Skip platform-specific builds (e.g. 2.9.1+cu128)
                if "+" in version:
                    continue

                major = version.split(".")[0]
                next_major = int(major) + 1
                constraints.append(
                    f"{norm_name.replace('_', '-')}>={version},<{next_major}"
                )

            if not constraints:
                return None

            constraints_file = get_plugins_dir() / "constraints.txt"
            constraints_file.write_text("\n".join(constraints) + "\n")
            return constraints_file
        except Exception:
            logger.warning("Failed to generate constraints", exc_info=True)
            return None

    def _compile_plugins(
        self, upgrade_package: str | None = None
    ) -> tuple[bool, str, str | None]:
        """Compile plugins against project constraints.

        Uses uv pip compile to resolve plugins along with project dependencies,
        ensuring plugins respect project constraints (e.g., torch pin).

        Args:
            upgrade_package: If provided, upgrade only this package

        Returns:
            Tuple of (success, resolved_file_path, error_message)
        """
        plugins_file = get_plugins_file()
        resolved_file = get_resolved_file()
        project_root = Path.cwd()
        pyproject = project_root / "pyproject.toml"

        if not pyproject.exists():
            return False, "", "pyproject.toml not found"

        args = [
            "uv",
            "pip",
            "compile",
            str(pyproject),
            *_get_torch_backend_args(),
            "-o",
            str(resolved_file),
        ]

        # Add plugins file if it exists and has content
        if plugins_file.exists() and plugins_file.read_text().strip():
            args.append(str(plugins_file))

        constraints_file = self._generate_constraints()
        if constraints_file:
            args.extend(["--constraint", str(constraints_file)])

        if upgrade_package:
            args.extend(["--upgrade-package", upgrade_package])

        env = {**os.environ, "PYTHONUTF8": "1"}
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=project_root,
            env=env,
        )

        if result.returncode != 0:
            return False, "", result.stderr

        return True, str(resolved_file), None

    def _sync_plugins(self, resolved_file: str) -> tuple[bool, str | None]:
        """Install packages from resolved.txt.

        Args:
            resolved_file: Path to resolved.txt

        Returns:
            Tuple of (success, error_message)
        """
        args = ["uv", "pip", "install", *_get_torch_backend_args(), "-r", resolved_file]
        logger.info(f"Running: {' '.join(args)}")

        env = {**os.environ, "PYTHONUTF8": "1"}
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        logger.info(f"uv pip install returncode: {result.returncode}")
        if result.stdout:
            logger.info(f"uv pip install stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"uv pip install stderr: {result.stderr}")

        if result.returncode != 0:
            return False, result.stderr

        return True, None

    def _extract_package_name(self, spec: str) -> str:
        """Extract package name from a specifier.

        Handles various formats:
        - git+https://github.com/user/repo.git -> package name from resolved.txt or repo
        - git+https://github.com/user/repo.git@branch -> package name from resolved.txt or repo
        - /path/to/local/dir -> package name from pyproject.toml or dir basename
        - package==1.0 -> package
        - package>=1.0,<2.0 -> package
        - package[extra] -> package
        """
        if spec.startswith("git+"):
            # Try resolved.txt for the authoritative name
            resolved_name = self._get_name_from_resolved(spec)
            if resolved_name:
                return resolved_name
            # Fallback to repo name
            url_part = spec.split("@")[0]  # Remove @branch or @commit
            return url_part.split("/")[-1].replace(".git", "")

        # Handle local directory paths (reuse existing helper)
        path = Path(spec)
        if path.is_dir():
            return self._get_package_name_from_path(path)

        # Handle version specifiers and extras
        # Split on any version specifier chars and take first part
        for sep in ["[", "==", ">=", "<=", "!=", "<", ">", "~="]:
            spec = spec.split(sep)[0]

        return spec.strip()

    def _normalize_package_name(self, name: str) -> str:
        """Normalize package name for comparison.

        Python packaging treats hyphens and underscores as equivalent,
        and comparisons are case-insensitive.
        """
        return name.lower().replace("-", "_")

    @property
    def pm(self) -> pluggy.PluginManager:
        """Get the underlying pluggy PluginManager."""
        return self._pm

    def _prevalidate_entrypoints(self, group: str) -> None:
        """Pre-validate entry points by attempting to load each one.

        Entry points that fail to load are logged, recorded in _failed_plugins,
        and all entry points from that package are blocked in pluggy so that
        load_setuptools_entrypoints skips the entire package.

        Args:
            group: Entry point group name (e.g. "scope")
        """
        from importlib.metadata import distributions

        for dist in distributions():
            try:
                eps = dist.entry_points
                scope_eps = [ep for ep in eps if ep.group == group]
                if not scope_eps:
                    continue

                package_name = dist.metadata.get("Name", "unknown")

                if len(scope_eps) != 1:
                    ep_names = [ep.name for ep in scope_eps]
                    logger.error(
                        f"Plugin '{package_name}' has {len(scope_eps)} entry points "
                        f"in the 'scope' group (expected 1): {ep_names}"
                    )
                    self._failed_plugins.append(
                        FailedPluginInfo(
                            package_name=package_name,
                            entry_point_name=", ".join(ep_names),
                            error_type="InvalidPluginError",
                            error_message=(
                                f"Expected 1 entry point in 'scope' group, "
                                f"found {len(scope_eps)}: {ep_names}"
                            ),
                        )
                    )
                    for ep in scope_eps:
                        self._pm.set_blocked(ep.name)
                    continue

                # Single entry point — validate it loads
                ep = scope_eps[0]
                try:
                    ep.load()
                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    logger.error(
                        f"Failed to load plugin entry point "
                        f"'{ep.name}' from '{package_name}': "
                        f"{error_type}: {error_message}"
                    )
                    self._failed_plugins.append(
                        FailedPluginInfo(
                            package_name=package_name,
                            entry_point_name=ep.name,
                            error_type=error_type,
                            error_message=error_message,
                        )
                    )
                    self._pm.set_blocked(ep.name)
            except Exception as e:
                logger.debug(f"Error checking distribution {dist}: {e}")

    def _is_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed in the current environment.

        Args:
            package_name: Package name (handles hyphen/underscore normalization)

        Returns:
            True if the package is installed, False otherwise
        """
        from importlib.metadata import PackageNotFoundError, distribution

        normalized = self._normalize_package_name(package_name)
        # Try both hyphen and underscore variants
        for name in (normalized, normalized.replace("_", "-")):
            try:
                distribution(name)
                return True
            except PackageNotFoundError:
                continue
        return False

    def ensure_plugins_installed(self) -> None:
        """Re-install plugins from resolved.txt if any are missing.

        This handles venv recreation (e.g., uv upgrade wiping .venv) by
        detecting missing plugin packages and reinstalling them from the
        persisted resolved.txt before plugin discovery runs.
        """
        plugins = self._read_plugins_file()
        if not plugins:
            return

        missing = []
        for spec in plugins:
            name = self._extract_package_name(spec)
            if not self._is_package_installed(name):
                missing.append(name)

        if not missing:
            return

        logger.warning(
            f"Plugins missing from environment (venv may have been recreated): "
            f"{missing}. Re-installing from plugin manifest..."
        )

        # Always recompile to ensure resolved.txt respects current lock constraints
        logger.info("Compiling plugins against current lock constraints...")
        ok, resolved_path, compile_error = self._compile_plugins()
        if not ok:
            # Fall back to existing resolved.txt if compile fails
            resolved_file = get_resolved_file()
            if resolved_file.exists():
                logger.warning(
                    f"Compile failed ({compile_error}), "
                    "falling back to existing resolved.txt"
                )
                resolved_path = str(resolved_file)
            else:
                logger.error(f"Failed to compile plugins: {compile_error}")
                return
        success, error = self._sync_plugins(resolved_path)

        if success:
            logger.info("Successfully re-installed missing plugins")
        else:
            logger.error(f"Failed to re-install plugins: {error}")

    def load_plugins(self) -> None:
        """Discover and load all plugins via entry points."""
        with self._lock:
            self._failed_plugins.clear()
            self._prevalidate_entrypoints("scope")
            self._pm.load_setuptools_entrypoints("scope")
            plugin_count = len(self._pm.get_plugins())
            if self._failed_plugins:
                names = [f.package_name for f in self._failed_plugins]
                logger.warning(
                    f"Loaded {plugin_count} plugin(s), "
                    f"{len(self._failed_plugins)} failed: {names}"
                )
            else:
                logger.info(f"Loaded {plugin_count} plugin(s)")

    def get_failed_plugins(self) -> list[FailedPluginInfo]:
        """Return a copy of the failed plugin info list (thread-safe)."""
        with self._lock:
            return list(self._failed_plugins)

    def register_plugin_pipelines(self, registry: "PipelineRegistry") -> None:
        """Call register_pipelines hook for all plugins.

        Args:
            registry: PipelineRegistry to register pipelines with
        """
        with self._lock:
            # Clear previous mappings
            self._pipeline_to_plugin.clear()

            def register_callback(pipeline_class: Any) -> None:
                """Callback function passed to plugins."""
                config_class = pipeline_class.get_config_class()
                pipeline_id = config_class.pipeline_id
                registry.register(pipeline_id, pipeline_class)

                # Track which plugin owns this pipeline
                # We'll update this mapping after the hook call
                logger.info(f"Registered plugin pipeline: {pipeline_id}")

            self._pm.hook.register_pipelines(register=register_callback)

            # Update pipeline-to-plugin mapping by checking which plugins provide which pipelines
            self._update_pipeline_plugin_mapping(registry)

    def _update_pipeline_plugin_mapping(self, registry: "PipelineRegistry") -> None:
        """Update the mapping of pipeline IDs to plugin names."""
        from importlib.metadata import distributions

        # Get all pipeline IDs currently registered
        all_pipeline_ids = set(registry.list_pipelines())

        # Skip packages whose entry points failed to load
        failed_packages = {fp.package_name for fp in self._failed_plugins}

        # Find which package provides each plugin
        for dist in distributions():
            # Check if this package has scope entry points
            try:
                eps = dist.entry_points
                scope_eps = [ep for ep in eps if ep.group == "scope"]
                if not scope_eps:
                    continue

                package_name = dist.metadata["Name"]
                if package_name in failed_packages:
                    continue
                self._registered_plugins.add(package_name)

                # Try to get pipeline IDs from this plugin
                for ep in scope_eps:
                    try:
                        plugin_module = ep.load()
                        if hasattr(plugin_module, "register_pipelines"):
                            # Call with a tracking callback
                            # Bind package_name to default param to avoid late binding
                            def tracking_callback(
                                pipeline_class: Any, pkg_name: str = package_name
                            ) -> None:
                                config_class = pipeline_class.get_config_class()
                                pipeline_id = config_class.pipeline_id
                                if pipeline_id in all_pipeline_ids:
                                    self._pipeline_to_plugin[pipeline_id] = pkg_name

                            plugin_module.register_pipelines(tracking_callback)
                    except Exception as e:
                        logger.debug(
                            f"Could not track pipelines for {package_name}: {e}"
                        )

            except Exception as e:
                logger.debug(f"Error checking distribution {dist}: {e}")

    def get_plugin_for_pipeline(self, pipeline_id: str) -> str | None:
        """Get the plugin package name that provides a pipeline.

        Args:
            pipeline_id: Pipeline ID to look up

        Returns:
            Plugin package name or None if not found
        """
        with self._lock:
            return self._pipeline_to_plugin.get(pipeline_id)

    def _get_plugin_source(self, dist: Any) -> tuple[str, bool, str | None, str | None]:
        """Determine the source of a plugin installation.

        Args:
            dist: Distribution object from importlib.metadata

        Returns:
            Tuple of (source, editable, editable_path, git_url)
            source is one of: "pypi", "git", "local"
            git_url is the original git URL if source is "git", else None
        """
        # Check for direct_url.json (PEP 610)
        try:
            # Access the dist-info directory to find direct_url.json
            if hasattr(dist, "_path") and dist._path:
                direct_url_path = dist._path / "direct_url.json"
                if direct_url_path.exists():
                    data = json.loads(direct_url_path.read_text())

                    # Check for git source
                    if "vcs_info" in data and data["vcs_info"].get("vcs") == "git":
                        git_url = data.get("url")
                        return ("git", False, None, git_url)

                    # Check for local editable
                    if "dir_info" in data and data["dir_info"].get("editable"):
                        url = data.get("url", "")
                        path = url.replace("file:///", "/").replace("file://", "")
                        # Handle Windows paths
                        if sys.platform == "win32" and path.startswith("/"):
                            path = path[1:]  # Remove leading slash for Windows
                        return ("local", True, path, None)

                    # Local but not editable
                    if "url" in data and data["url"].startswith("file://"):
                        return ("local", False, None, None)
        except Exception as e:
            logger.debug(f"Error reading direct_url.json for {dist}: {e}")

        # Default to PyPI
        return ("pypi", False, None, None)

    async def list_plugins_async(self) -> list[dict[str, Any]]:
        """Get all installed plugins with metadata.

        Returns:
            List of plugin info dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_plugins_sync)

    def _list_plugins_sync(self) -> list[dict[str, Any]]:
        """Synchronous implementation of list_plugins."""
        from importlib.metadata import distributions

        plugins = []

        with self._lock:
            for dist in distributions():
                try:
                    eps = dist.entry_points
                    scope_eps = [ep for ep in eps if ep.group == "scope"]
                    if not scope_eps:
                        continue

                    package_name = dist.metadata["Name"]
                    source, editable, editable_path, git_url = self._get_plugin_source(
                        dist
                    )

                    # Get pipelines provided by this plugin
                    pipelines = []
                    for pipeline_id, plugin_name in self._pipeline_to_plugin.items():
                        if plugin_name == package_name:
                            # Get pipeline metadata
                            from scope.core.pipelines.registry import PipelineRegistry

                            config_class = PipelineRegistry.get_config_class(
                                pipeline_id
                            )
                            if config_class:
                                pipelines.append(
                                    {
                                        "pipeline_id": pipeline_id,
                                        "pipeline_name": config_class.pipeline_name,
                                    }
                                )

                    # For git packages, use the git URL; for PyPI, use package name
                    package_spec = (
                        f"git+{git_url}"
                        if source == "git" and git_url
                        else package_name
                    )

                    # Check for updates (skip local/editable plugins)
                    if source == "local" or editable:
                        latest_version = None
                        update_available = None
                    else:
                        update_info = self._check_plugin_update(
                            package_name, package_spec
                        )
                        latest_version = update_info.get("latest_version")
                        update_available = update_info.get("update_available")

                    plugins.append(
                        {
                            "name": package_name,
                            "version": dist.metadata.get("Version"),
                            "author": dist.metadata.get("Author")
                            or dist.metadata.get("Author-email"),
                            "description": dist.metadata.get("Summary"),
                            "source": source,
                            "editable": editable,
                            "editable_path": editable_path,
                            "pipelines": pipelines,
                            "latest_version": latest_version,
                            "update_available": update_available,
                            "package_spec": package_spec,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error getting plugin info for {dist}: {e}")

        # Deduplicate by normalized package name.
        # When duplicates exist (e.g., stale dist-info), prefer editable/local.
        seen: dict[str, dict[str, Any]] = {}
        for plugin in plugins:
            normalized = self._normalize_package_name(plugin["name"])
            existing = seen.get(normalized)
            if existing is None:
                seen[normalized] = plugin
            elif plugin["editable"] and not existing["editable"]:
                seen[normalized] = plugin
            elif plugin["source"] == "local" and existing["source"] != "local":
                seen[normalized] = plugin

        return list(seen.values())

    def _check_plugin_update(
        self, name: str, package_spec: str | None = None
    ) -> dict[str, Any]:
        """Check for updates using compile --upgrade-package.

        Compares current resolved.txt with a fresh compile using --upgrade-package
        to find if a newer version is available that respects project constraints.

        Args:
            name: Package name (used for version lookup)
            package_spec: Package specifier (not used in compile approach, kept for API compat)

        Returns:
            Dict with latest_version and update_available keys
        """
        import tempfile

        resolved_file = get_resolved_file()

        # Get current version from resolved.txt (if it exists)
        current_version = self._get_version_from_resolved(name, str(resolved_file))

        # If no resolved file exists, we can't check for updates via compile
        if not resolved_file.exists():
            return {"latest_version": None, "update_available": None}

        # Create temp file for upgrade check
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                temp_resolved = f.name

            plugins_file = get_plugins_file()
            project_root = Path.cwd()
            pyproject = project_root / "pyproject.toml"

            if not pyproject.exists():
                return {"latest_version": None, "update_available": None}

            args = [
                "uv",
                "pip",
                "compile",
                str(pyproject),
                *_get_torch_backend_args(),
                "-o",
                temp_resolved,
                "--upgrade-package",
                name,
                "--refresh-package",
                name,  # Force refresh for git packages
            ]

            if plugins_file.exists() and plugins_file.read_text().strip():
                args.append(str(plugins_file))

            env = {**os.environ, "PYTHONUTF8": "1"}
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=project_root,
                env=env,
                timeout=60,
            )

            if result.returncode != 0:
                return {"latest_version": None, "update_available": None}

            # Get new version from temp resolved file
            new_version = self._get_version_from_resolved(name, temp_resolved)

            if new_version and new_version != current_version:
                return {"latest_version": new_version, "update_available": True}

            return {"latest_version": None, "update_available": False}

        except Exception as e:
            logger.warning(f"Failed to check updates for {name}: {e}")
            return {"latest_version": None, "update_available": None}
        finally:
            Path(temp_resolved).unlink(missing_ok=True)

    def _get_version_from_resolved(self, name: str, resolved_file: str) -> str | None:
        """Extract package version/commit from resolved.txt.

        Args:
            name: Package name to look for
            resolved_file: Path to resolved.txt file

        Returns:
            Version string or git commit hash, or None if not found
        """
        import re

        if not Path(resolved_file).exists():
            return None

        content = Path(resolved_file).read_text()

        # Normalize package name for regex (handle - vs _)
        # Use re.sub to avoid replacing characters in the replacement string
        normalized = re.sub(r"[-_]", "[-_]", name)

        # Try version match: package==1.2.3
        match = re.search(
            rf"^{normalized}==([\S]+)", content, re.MULTILINE | re.IGNORECASE
        )
        if match:
            return match.group(1)

        # Try git commit match: package @ git+https://...@commit
        match = re.search(
            rf"^{normalized}\s+@\s+git\+[^@]+@(\w+)",
            content,
            re.MULTILINE | re.IGNORECASE,
        )
        if match:
            return match.group(1)

        return None

    def _get_name_from_resolved(self, git_url: str) -> str | None:
        """Look up the package name for a git URL from resolved.txt."""
        resolved_file = get_resolved_file()
        if not resolved_file.exists():
            return None
        content = resolved_file.read_text(encoding="utf-8")
        # Normalize: strip git+ prefix and .git suffix for comparison
        normalized_url = git_url.removeprefix("git+").removesuffix(".git").lower()
        # Also strip @branch/@commit from the input URL
        normalized_url = normalized_url.split("@")[0].removesuffix(".git")
        for line in content.splitlines():
            if " @ git+" not in line:
                continue
            # "flashvsr @ git+https://...@commit"
            name_part, _, url_part = line.partition(" @ ")
            url_base = (
                url_part.removeprefix("git+").split("@")[0].removesuffix(".git").lower()
            )
            if url_base == normalized_url:
                return name_part.strip()
        return None

    async def get_plugin_info_async(self, name: str) -> dict[str, Any] | None:
        """Get info for a specific plugin.

        Args:
            name: Plugin package name

        Returns:
            Plugin info dictionary or None if not found
        """
        plugins = await self.list_plugins_async()
        for plugin in plugins:
            if plugin["name"] == name:
                return plugin
        return None

    async def check_updates_async(self) -> list[dict[str, Any]]:
        """Check for available updates for all installed plugins.

        Returns:
            List of update info dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._check_updates_sync)

    def _check_updates_sync(self) -> list[dict[str, Any]]:
        """Synchronous implementation of check_updates."""
        import urllib.request

        plugins = self._list_plugins_sync()
        updates = []

        for plugin in plugins:
            name = plugin["name"]
            source = plugin["source"]
            installed_version = plugin["version"] or "unknown"

            if source == "local":
                # Skip local/editable plugins
                updates.append(
                    {
                        "name": name,
                        "installed_version": installed_version,
                        "latest_version": None,
                        "update_available": None,
                        "source": source,
                    }
                )
                continue

            if source == "pypi":
                # Query PyPI for latest version
                try:
                    url = f"https://pypi.org/pypi/{name}/json"
                    with urllib.request.urlopen(url, timeout=10) as response:
                        data = json.loads(response.read().decode())
                        latest_version = data["info"]["version"]
                        update_available = latest_version != installed_version

                        updates.append(
                            {
                                "name": name,
                                "installed_version": installed_version,
                                "latest_version": latest_version,
                                "update_available": update_available,
                                "source": source,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to check PyPI updates for {name}: {e}")
                    updates.append(
                        {
                            "name": name,
                            "installed_version": installed_version,
                            "latest_version": None,
                            "update_available": None,
                            "source": source,
                        }
                    )

            elif source == "git":
                # For git, we can't easily check without the original URL
                # Would need to store it or run a dry-run upgrade
                updates.append(
                    {
                        "name": name,
                        "installed_version": installed_version,
                        "latest_version": None,
                        "update_available": None,
                        "source": source,
                    }
                )

        return updates

    async def validate_install_async(
        self, packages: list[str]
    ) -> tuple[bool, str | None]:
        """Validate that packages can be installed without conflicts.

        Args:
            packages: List of package specifiers

        Returns:
            Tuple of (is_valid, error_message)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._validate_install_sync, packages)

    def _validate_install_sync(self, packages: list[str]) -> tuple[bool, str | None]:
        """Synchronous implementation of validate_install."""
        validator = DependencyValidator()
        result = validator.validate_install(packages)
        return (result.is_valid, result.error_message)

    async def install_plugin_async(
        self,
        package: str,
        editable: bool = False,
        upgrade: bool = False,
        pre: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Install a plugin.

        Args:
            package: Package specifier (PyPI name, git URL, or local path)
            editable: Install in editable mode
            upgrade: Upgrade if already installed
            pre: Include pre-release versions
            force: Skip dependency validation

        Returns:
            Installation result dictionary

        Raises:
            PluginDependencyError: If dependency validation fails
            PluginInstallError: If installation fails
            PluginNameCollisionError: If plugin with same name exists from different source
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._install_plugin_sync, package, editable, upgrade, pre, force
        )

    def _install_plugin_sync(
        self,
        package: str,
        editable: bool = False,
        upgrade: bool = False,
        pre: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Synchronous implementation of install_plugin.

        Uses compile-based resolution to ensure plugins respect project
        constraints (e.g., torch pin). Includes freeze-based rollback to
        restore venv state if installation fails after compile succeeds.
        """
        from .venv_snapshot import VenvSnapshot

        # For editable installs, use direct pip install (local dev)
        if editable:
            return self._install_editable_plugin(package)

        # Extract package name for tracking
        package_base = self._extract_package_name(package)
        normalized_base = self._normalize_package_name(package_base)

        # Read current plugins and check if already in list
        plugins = self._read_plugins_file()
        existing_idx = None
        for i, p in enumerate(plugins):
            if (
                self._normalize_package_name(self._extract_package_name(p))
                == normalized_base
            ):
                existing_idx = i
                break

        # Store original for rollback
        original_plugins = plugins.copy()

        # Capture venv state before installation for rollback
        snapshot = VenvSnapshot()
        snapshot_captured = snapshot.capture()
        if not snapshot_captured:
            logger.warning(
                "Failed to capture venv snapshot, proceeding without rollback support"
            )

        # Update plugins list
        if existing_idx is not None:
            plugins[existing_idx] = package  # Update specifier
        else:
            plugins.append(package)

        self._write_plugins_file(plugins)

        # Compile with project constraints
        logger.info(f"Compiling plugins with package: {package}")
        success, resolved_file, error = self._compile_plugins(
            upgrade_package=package_base if upgrade else None
        )

        if not success:
            # Rollback plugins.txt on failure (no venv changes yet)
            self._write_plugins_file(original_plugins)
            snapshot.discard()  # Clean up snapshot files
            raise PluginDependencyError(f"Dependency resolution failed: {error}")

        # Install resolved packages
        logger.info(f"Installing from resolved file: {resolved_file}")
        success, error = self._sync_plugins(resolved_file)

        if not success:
            # Rollback plugins.txt
            self._write_plugins_file(original_plugins)

            # Attempt venv rollback if we have a snapshot
            if snapshot_captured:
                logger.info("Attempting venv rollback from snapshot...")
                restore_success, restore_error = snapshot.restore()
                if restore_success:
                    logger.info("Successfully rolled back venv state")
                    snapshot.discard()  # Clean up backup files after successful restore
                else:
                    logger.error(
                        f"Failed to rollback venv state: {restore_error}. "
                        "Manual recovery may be required: delete .venv and run 'uv sync'"
                    )
            else:
                snapshot.discard()

            raise PluginInstallError(f"Installation failed: {error}")

        # Success - clean up snapshot files
        snapshot.discard()

        # Don't try to reload plugins in-process - the server restart will
        # handle loading the new plugin with a clean slate (no caching issues)
        return {
            "success": True,
            "message": f"Successfully installed {package}. Restart server to load.",
            "plugin": {"name": package_base},
        }

    def _install_editable_plugin(self, package: str) -> dict[str, Any]:
        """Install a plugin in editable mode (for local development).

        Editable installs bypass compile-based resolution since they're
        used for local development where the developer manages dependencies.
        Still uses snapshot-based rollback to protect venv integrity.
        """
        from .venv_snapshot import VenvSnapshot

        # Resolve package name early so we can clean up plugins.txt
        # Convert Git Bash POSIX-style paths (e.g. /c/Users/...) to Windows paths
        if (
            sys.platform == "win32"
            and len(package) >= 3
            and package[0] == "/"
            and package[1].isalpha()
            and package[2] == "/"
        ):
            package = f"{package[1].upper()}:{package[2:]}"
        package_path = Path(package).resolve()
        package_name = self._get_package_name_from_path(package_path)
        normalized_name = self._normalize_package_name(package_name)

        # Remove any existing entry for this package from plugins.txt
        # (e.g., a prior git/PyPI install) to avoid stale entries
        plugins = self._read_plugins_file()
        new_plugins = [
            p
            for p in plugins
            if self._normalize_package_name(self._extract_package_name(p))
            != normalized_name
        ]
        if len(new_plugins) != len(plugins):
            self._write_plugins_file(new_plugins)
            logger.info(f"Removed existing entry for {package_name} from plugins.txt")

        # Capture venv state before installation for rollback
        snapshot = VenvSnapshot()
        snapshot_captured = snapshot.capture()
        if not snapshot_captured:
            logger.warning(
                "Failed to capture venv snapshot, proceeding without rollback support"
            )

        args = [
            "uv",
            "pip",
            "install",
            *_get_torch_backend_args(),
            "--editable",
            str(package_path),
        ]

        env = {**os.environ, "PYTHONUTF8": "1"}
        logger.info(f"Running: {' '.join(args)}")
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        logger.info(f"uv pip install returncode: {result.returncode}")
        if result.stdout:
            logger.info(f"uv pip install stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"uv pip install stderr: {result.stderr}")

        if result.returncode != 0:
            # Attempt venv rollback if we have a snapshot
            if snapshot_captured:
                logger.info("Attempting venv rollback from snapshot...")
                restore_success, restore_error = snapshot.restore()
                if restore_success:
                    logger.info("Successfully rolled back venv state")
                    snapshot.discard()
                else:
                    logger.error(
                        f"Failed to rollback venv state: {restore_error}. "
                        "Manual recovery may be required: delete .venv and run 'uv sync'"
                    )
            else:
                snapshot.discard()

            raise PluginInstallError(
                f"Installation failed: {result.stderr or result.stdout}"
            )

        # Success - clean up snapshot files
        snapshot.discard()

        return {
            "success": True,
            "message": f"Successfully installed {package} (editable). Restart server to load.",
            "plugin": {"name": package_name},
        }

    def _get_package_name_from_path(self, path: Path) -> str:
        """Extract package name from a local path."""
        # Try pyproject.toml first
        pyproject = path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib

                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                    return data.get("project", {}).get("name", path.name)
            except Exception:
                pass

        # Fall back to directory name
        return path.name

    async def uninstall_plugin_async(
        self, name: str, pipeline_manager: Any = None
    ) -> dict[str, Any]:
        """Uninstall a plugin.

        Args:
            name: Plugin package name
            pipeline_manager: Optional PipelineManager to unload pipelines

        Returns:
            Uninstallation result dictionary

        Raises:
            PluginNotFoundError: If plugin not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._uninstall_plugin_sync, name, pipeline_manager
        )

    def _uninstall_plugin_sync(
        self, name: str, pipeline_manager: Any = None
    ) -> dict[str, Any]:
        """Synchronous implementation of uninstall_plugin."""
        from scope.core.pipelines.registry import PipelineRegistry

        # Check if plugin exists
        plugin_info = None
        plugins_list = self._list_plugins_sync()
        logger.debug(f"Found {len(plugins_list)} installed plugins")
        for plugin in plugins_list:
            if plugin["name"] == name:
                plugin_info = plugin
                break

        if not plugin_info:
            raise PluginNotFoundError(f"Plugin '{name}' not found")

        logger.debug(f"Found plugin: {plugin_info}")

        # Get pipelines from this plugin
        plugin_pipelines = [p["pipeline_id"] for p in plugin_info.get("pipelines", [])]
        if plugin_pipelines:
            logger.debug(f"Plugin pipelines to unload: {plugin_pipelines}")

        # Unload pipelines if pipeline_manager provided
        unloaded_pipelines = []
        if pipeline_manager and plugin_pipelines:
            for pipeline_id in plugin_pipelines:
                try:
                    pipeline_manager.unload_pipeline_by_id(pipeline_id)
                    unloaded_pipelines.append(pipeline_id)
                    logger.info(f"Unloaded pipeline: {pipeline_id}")
                except Exception as e:
                    logger.warning(f"Failed to unload pipeline {pipeline_id}: {e}")

        # Unregister pipelines from registry
        with self._lock:
            for pipeline_id in plugin_pipelines:
                PipelineRegistry.unregister(pipeline_id)
                if pipeline_id in self._pipeline_to_plugin:
                    del self._pipeline_to_plugin[pipeline_id]
                logger.info(f"Unregistered pipeline from registry: {pipeline_id}")

        # Remove from plugins.txt
        plugins = self._read_plugins_file()
        normalized_name = self._normalize_package_name(name)
        new_plugins = [
            p
            for p in plugins
            if self._normalize_package_name(self._extract_package_name(p))
            != normalized_name
        ]

        # Check if plugin was in plugins.txt (non-editable plugins)
        plugins_file_changed = len(new_plugins) != len(plugins)
        if plugins_file_changed:
            self._write_plugins_file(new_plugins)
            logger.info(f"Removed {name} from plugins.txt")
        else:
            logger.warning(
                f"Plugin {name} not found in plugins.txt. "
                f"Searched for normalized name '{normalized_name}' in: {plugins}"
            )

        # Run uv pip uninstall
        args = ["uv", "pip", "uninstall", name]
        logger.info(f"Running: {' '.join(args)}")
        env = {**os.environ, "PYTHONUTF8": "1"}

        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        logger.info(f"uv pip uninstall returncode: {result.returncode}")
        if result.stdout:
            logger.info(f"uv pip uninstall stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"uv pip uninstall stderr: {result.stderr}")

        if result.returncode != 0:
            raise PluginInstallError(
                f"Uninstallation failed: {result.stderr or result.stdout}"
            )

        # Re-compile to update resolved.txt (if plugins.txt was changed)
        if plugins_file_changed:
            self._compile_plugins()
            logger.info("Re-compiled plugins after uninstall")

        return {
            "success": True,
            "message": f"Successfully uninstalled {name}",
            "unloaded_pipelines": unloaded_pipelines,
        }

    async def reload_plugin_async(
        self,
        name: str,
        force: bool = False,
        pipeline_manager: Any = None,
    ) -> dict[str, Any]:
        """Reload an editable plugin.

        Args:
            name: Plugin package name
            force: Force reload even if pipelines are loaded
            pipeline_manager: Optional PipelineManager to check/unload pipelines

        Returns:
            Reload result dictionary

        Raises:
            PluginNotFoundError: If plugin not found
            PluginNotEditableError: If plugin is not editable
            PluginInUseError: If pipelines are loaded and force=False
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._reload_plugin_sync, name, force, pipeline_manager
        )

    def _reload_plugin_sync(
        self,
        name: str,
        force: bool = False,
        pipeline_manager: Any = None,
    ) -> dict[str, Any]:
        """Synchronous implementation of reload_plugin."""
        from scope.core.pipelines.registry import PipelineRegistry

        # Get plugin info
        plugin_info = None
        plugins = self._list_plugins_sync()
        for plugin in plugins:
            if plugin["name"] == name:
                plugin_info = plugin
                break

        if not plugin_info:
            raise PluginNotFoundError(f"Plugin '{name}' not found")

        if not plugin_info.get("editable"):
            raise PluginNotEditableError(
                f"Plugin '{name}' is not installed in editable mode"
            )

        # Get pipelines from this plugin before reload
        old_pipeline_ids = {p["pipeline_id"] for p in plugin_info.get("pipelines", [])}

        # Check if any pipelines are loaded
        loaded_pipelines = []
        if pipeline_manager:
            for pipeline_id in old_pipeline_ids:
                try:
                    pipeline_manager.get_pipeline_by_id(pipeline_id)
                    loaded_pipelines.append(pipeline_id)
                except Exception:
                    pass

        if loaded_pipelines and not force:
            raise PluginInUseError(
                f"Plugin '{name}' has loaded pipelines: {loaded_pipelines}. Use force=true to unload them.",
                loaded_pipelines,
            )

        # Unload pipelines if force=True
        if loaded_pipelines and force:
            for pipeline_id in loaded_pipelines:
                try:
                    pipeline_manager.unload_pipeline_by_id(pipeline_id)
                except Exception as e:
                    logger.warning(f"Failed to unload pipeline {pipeline_id}: {e}")

        # Unregister old pipelines
        with self._lock:
            for pipeline_id in old_pipeline_ids:
                PipelineRegistry.unregister(pipeline_id)
                if pipeline_id in self._pipeline_to_plugin:
                    del self._pipeline_to_plugin[pipeline_id]

        # Get the plugin module from pluggy and unregister it
        plugin_to_unregister = None
        for plugin in self._pm.get_plugins():
            plugin_name = getattr(plugin, "__name__", "")
            if name in plugin_name or plugin_name in name:
                plugin_to_unregister = plugin
                break

        if plugin_to_unregister:
            try:
                self._pm.unregister(plugin_to_unregister)
            except Exception as e:
                logger.warning(f"Failed to unregister plugin {name}: {e}")

        # Reload the plugin module
        editable_path = plugin_info.get("editable_path")
        if editable_path:
            self._reload_module_tree(name, editable_path)

        # Re-load plugins via entry points (with prevalidation)
        self._failed_plugins.clear()
        self._prevalidate_entrypoints("scope")
        self._pm.load_setuptools_entrypoints("scope")

        # Re-register pipelines
        self.register_plugin_pipelines(PipelineRegistry)

        # Get new pipeline IDs
        new_plugin_info = None
        for plugin in self._list_plugins_sync():
            if plugin["name"] == name:
                new_plugin_info = plugin
                break

        new_pipeline_ids: set[str] = set()
        if new_plugin_info:
            new_pipeline_ids = {
                p["pipeline_id"] for p in new_plugin_info.get("pipelines", [])
            }

        # Calculate diff
        reloaded = old_pipeline_ids & new_pipeline_ids
        added = new_pipeline_ids - old_pipeline_ids
        removed = old_pipeline_ids - new_pipeline_ids

        return {
            "success": True,
            "message": f"Successfully reloaded {name}",
            "reloaded_pipelines": list(reloaded),
            "added_pipelines": list(added),
            "removed_pipelines": list(removed),
        }

    def _reload_module_tree(self, package_name: str, package_path: str) -> None:
        """Reload all modules from a package.

        Args:
            package_name: Name of the package
            package_path: Path to the package
        """
        # Normalize the package name to module form
        module_name = package_name.replace("-", "_")

        # Find all modules from this package
        modules_to_reload = []
        for name, module in list(sys.modules.items()):
            if name.startswith(module_name) or name == module_name:
                modules_to_reload.append((name, module))

        # Sort by depth (reload deepest first)
        modules_to_reload.sort(key=lambda x: x[0].count("."), reverse=True)

        # Reload each module
        for name, module in modules_to_reload:
            try:
                importlib.reload(module)
                logger.debug(f"Reloaded module: {name}")
            except Exception as e:
                logger.warning(f"Failed to reload module {name}: {e}")
                # Remove from sys.modules to force fresh import
                if name in sys.modules:
                    del sys.modules[name]


# Module-level singleton
_plugin_manager: PluginManager | None = None
_init_lock = threading.Lock()


def get_plugin_manager() -> PluginManager:
    """Get the singleton PluginManager instance."""
    global _plugin_manager
    with _init_lock:
        if _plugin_manager is None:
            _plugin_manager = PluginManager()
        return _plugin_manager


# Backward-compatible module-level exports
# These use the singleton under the hood
def _get_pm() -> pluggy.PluginManager:
    """Get the pluggy PluginManager instance."""
    return get_plugin_manager().pm


# Create a property-like access for pm
class _PMProxy:
    """Proxy to access the pluggy PluginManager."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_plugin_manager().pm, name)


pm = _PMProxy()


def ensure_plugins_installed() -> None:
    """Re-install plugins if any are missing from the environment."""
    get_plugin_manager().ensure_plugins_installed()


def load_plugins() -> None:
    """Discover and load all plugins via entry points."""
    get_plugin_manager().load_plugins()


def register_plugin_pipelines(registry: "PipelineRegistry") -> None:
    """Call register_pipelines hook for all plugins.

    Args:
        registry: PipelineRegistry to register pipelines with
    """
    get_plugin_manager().register_plugin_pipelines(registry)
