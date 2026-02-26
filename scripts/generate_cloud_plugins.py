#!/usr/bin/env python3
"""
Fetch plugins from the daydream API and generate cloud-plugins.txt for Docker builds.

This fetches the same plugin list shown in the Discover tab of the UI.
Run this during CI/CD to generate an up-to-date plugin list for cloud builds.
"""

import json
import urllib.request
from pathlib import Path

DAYDREAM_API_BASE = "https://api.daydream.live"
OUTPUT_FILE = Path(__file__).parent.parent / "cloud-plugins.txt"


def fetch_discover_plugins() -> list[dict]:
    """Fetch plugins from the daydream API (same as Discover tab)."""
    url = f"{DAYDREAM_API_BASE}/v1/plugins?limit=50&sortBy=popularity"
    print(f"Fetching plugins from {url}")

    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode())

    return data.get("plugins", [])


def main():
    plugins = fetch_discover_plugins()

    # Filter to plugins with repository URLs (installable via git)
    installable = [
        p for p in plugins
        if p.get("repositoryUrl")
    ]

    print(f"Found {len(plugins)} plugins, {len(installable)} installable via git")

    # Generate plugin specifiers
    lines = [
        "# Auto-generated cloud plugin list from daydream API",
        "# Do not edit manually - regenerate with: python scripts/generate_cloud_plugins.py",
        "",
    ]

    for plugin in installable:
        repo_url = plugin["repositoryUrl"]
        name = plugin.get("name", "unknown")
        
        # Clean the URL: remove fragments (#...) and trailing slashes
        repo_url = repo_url.split("#")[0].rstrip("/")
        
        # Ensure .git suffix for GitHub repos (helps with caching)
        if "github.com" in repo_url and not repo_url.endswith(".git"):
            repo_url = repo_url + ".git"
        
        # Format: git+https://github.com/user/repo.git
        spec = f"git+{repo_url}" if not repo_url.startswith("git+") else repo_url
        lines.append(f"{spec}  # {name}")

    OUTPUT_FILE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(installable)} plugins to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
