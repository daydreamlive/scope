"""Plugin manager for discovering and loading Scope plugins."""

import asyncio
import importlib
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pluggy

from .dependency_validator import DependencyValidator
from .hookspecs import ScopeHookSpec

if TYPE_CHECKING:
    from scope.core.pipelines.registry import PipelineRegistry

logger = logging.getLogger(__name__)


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

    @property
    def pm(self) -> pluggy.PluginManager:
        """Get the underlying pluggy PluginManager."""
        return self._pm

    def load_plugins(self) -> None:
        """Discover and load all plugins via entry points."""
        with self._lock:
            self._pm.load_setuptools_entrypoints("scope")
            plugin_count = len(self._pm.get_plugins())
            logger.info(f"Loaded {plugin_count} plugin(s)")

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

        # Find which package provides each plugin
        for dist in distributions():
            # Check if this package has scope entry points
            try:
                eps = dist.entry_points
                scope_eps = [ep for ep in eps if ep.group == "scope"]
                if not scope_eps:
                    continue

                package_name = dist.metadata["Name"]
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

        return plugins

    def _check_plugin_update(
        self, name: str, package_spec: str | None = None
    ) -> dict[str, Any]:
        """Check for updates using uv pip install --dry-run.

        This works for both PyPI and git packages uniformly.

        Args:
            name: Package name (used for logging and regex matching)
            package_spec: Package specifier for uv (e.g., "package-name" for PyPI,
                         "git+https://..." for git). Defaults to name if not provided.

        Output when update available (PyPI):
            Would install 1 package
             + package==1.2.3

        Output when update available (git):
            Would install 1 package
             + package @ git+https://github.com/user/repo.git@commit_hash

        Output when up to date:
            Would make no changes
        """
        import re

        if package_spec is None:
            package_spec = name

        try:
            env = {**os.environ, "PYTHONUTF8": "1"}
            result = subprocess.run(
                ["uv", "pip", "install", "--dry-run", "--upgrade", package_spec],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=env,
            )

            output = result.stdout + result.stderr

            # Check if no changes needed
            if "Would make no changes" in output:
                return {"latest_version": None, "update_available": False}

            # Parse the new version from output:
            # PyPI format: "+ package==1.2.3"
            # Git format: "+ package @ git+https://...@commit_hash"
            pattern = r"\+ " + re.escape(name) + r"(?:==| @ )([\S]+)"
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                latest_version = match.group(1)
                return {"latest_version": latest_version, "update_available": True}

            # If we see "Would install" but couldn't parse version, still mark as update available
            if "Would install" in output:
                return {"latest_version": None, "update_available": True}

            return {"latest_version": None, "update_available": None}

        except Exception as e:
            logger.warning(f"Failed to check updates for {name}: {e}")
            return {"latest_version": None, "update_available": None}

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
        """Synchronous implementation of install_plugin."""
        # Validate dependencies unless forced
        packages_to_validate = [package]
        if not force:
            is_valid, error_message = self._validate_install_sync(packages_to_validate)
            if not is_valid:
                raise PluginDependencyError(error_message or "Dependency conflict")

        # Build uv pip install command
        args = ["uv", "pip", "install", "--torch-backend", "cu128"]

        if upgrade:
            args.append("--upgrade")
        if editable:
            args.extend(["--editable", package])
        else:
            args.append(package)
        if pre:
            args.append("--pre")

        # Set PYTHONUTF8=1 for proper Unicode handling
        env = {**os.environ, "PYTHONUTF8": "1"}

        # Run installation
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
            raise PluginInstallError(
                f"Installation failed: {result.stderr or result.stdout}"
            )

        # Reload plugins to pick up the new one
        self._reload_all_plugins()

        # Get info about the installed plugin
        # For editable installs, extract package name from path
        if editable:
            # Try to get package name from pyproject.toml or setup.py
            package_path = Path(package).resolve()
            package_name = self._get_package_name_from_path(package_path)
        else:
            # For PyPI/git, extract base package name
            package_name = (
                package.split("[")[0].split("==")[0].split(">=")[0].split("<=")[0]
            )
            package_name = (
                package_name.replace("git+", "").split("/")[-1].replace(".git", "")
            )

        # Find the installed plugin
        plugins = self._list_plugins_sync()
        installed_plugin = None
        for plugin in plugins:
            if plugin["name"].lower() == package_name.lower():
                installed_plugin = plugin
                break

        return {
            "success": True,
            "message": f"Successfully installed {package}",
            "plugin": installed_plugin,
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
        plugins = self._list_plugins_sync()
        logger.debug(f"Found {len(plugins)} installed plugins")
        for plugin in plugins:
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

        # Re-load plugins via entry points
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

    def _reload_all_plugins(self) -> None:
        """Reload all plugins from entry points.

        This clears the module cache for all registered plugins to ensure
        new code is loaded (important for upgrades).
        """
        # Clear module cache for all registered plugins
        for package_name in list(self._registered_plugins):
            self._clear_package_modules(package_name)

        # Clear existing plugins from pluggy
        for plugin in list(self._pm.get_plugins()):
            try:
                self._pm.unregister(plugin)
            except Exception:
                pass

        # Reload entry points
        self._pm.load_setuptools_entrypoints("scope")

        # Update pipeline mapping
        from scope.core.pipelines.registry import PipelineRegistry

        self.register_plugin_pipelines(PipelineRegistry)

    def _clear_package_modules(self, package_name: str) -> None:
        """Clear all modules from a package from sys.modules.

        Args:
            package_name: Package name (will be normalized to module form)
        """
        # Normalize the package name to module form (e.g., my-package -> my_package)
        module_name = package_name.replace("-", "_")

        # Find and remove all modules from this package
        modules_to_remove = [
            name
            for name in sys.modules
            if name == module_name or name.startswith(f"{module_name}.")
        ]

        for name in modules_to_remove:
            del sys.modules[name]
            logger.debug(f"Cleared module from cache: {name}")


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


def load_plugins() -> None:
    """Discover and load all plugins via entry points."""
    get_plugin_manager().load_plugins()


def register_plugin_pipelines(registry: "PipelineRegistry") -> None:
    """Call register_pipelines hook for all plugins.

    Args:
        registry: PipelineRegistry to register pipelines with
    """
    get_plugin_manager().register_plugin_pipelines(registry)
