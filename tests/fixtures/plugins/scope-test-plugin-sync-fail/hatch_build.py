"""Custom build hook that intentionally fails during installation."""

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        # Fail during the build/install process
        raise RuntimeError(
            "INTENTIONAL FAILURE: This plugin is designed to fail for testing "
            "venv rollback functionality. Do not use in production."
        )
