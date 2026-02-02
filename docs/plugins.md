# Plugins

The Scope plugin system enables third-party extensions to provide custom pipelines. This guide covers how to install, manage, and develop plugins. The technical details on the plugin system can be found in the [architecture doc](./architecture/plugins.md).

## Using Plugins

### Using Plugins from the Desktop App

#### Opening Plugin Settings

1. Click the gear button in the app header to open the Settings dialog.

<img width="1382" height="86" alt="Screenshot 2026-02-02 164822" src="https://github.com/user-attachments/assets/0ca5d0ce-0fb7-4a5d-a342-d4d395db0ff8" />

2. Navigate to the Plugins tab.

<img width="657" height="399" alt="Screenshot 2026-02-02 164848" src="https://github.com/user-attachments/assets/de5d17f8-3f1d-41c1-953a-7c5b406032b8" />

#### Installing a Plugin

1. In the Plugins tab, locate the installation input field.

2. Enter a package spec (see [Plugin Sources](#plugin-sources) for format options) or browse for a local plugin directory.

3. Click the **Install** button.

4. Wait for the installation to complete and the server to restart.

<img width="414" height="112" alt="Screenshot 2026-02-02 164941" src="https://github.com/user-attachments/assets/aeac6653-7395-4a5f-a31f-d2b2e7e4730f" />

<img width="423" height="126" alt="Screenshot 2026-02-02 165047" src="https://github.com/user-attachments/assets/15ce6b29-2d9f-46d4-a3e1-86387aac8abb" />

#### Viewing Installed Plugins

The Plugins tab displays a list of all installed plugins.

#### Uninstalling a Plugin

1. Find the plugin you want to remove in the installed plugins list.

2. Click the **Uninstall** button next to the plugin.

<img width="481" height="177" alt="trashcan" src="https://github.com/user-attachments/assets/f68ac436-23d0-4b55-b4bd-d9256e551aaa" />

3. Wait for the uninstallation to complete and the server to restart.

#### Reloading a Plugin (Local Plugins Only)

When using a local plugin directory, you can reload it after making code changes without reinstalling:

1. Find your locally installed plugin in the list.

2. Click the **Reload** button next to the plugin.

<img width="481" height="177" alt="reload" src="https://github.com/user-attachments/assets/3b22f1c8-d1a7-4fce-a322-116a2873945c" />

3. Wait for the server to restart.

### Using Plugins via Manual Installation

#### Key Differences from Desktop

The experience of using plugins with a manual installation of Scope is very similar to the experience in the desktop app with the following exceptions:

- No deep link support eg a website cannot auto-open the UI to facilitate plugin installation
- No local plugin support

## Plugin Sources

Plugins can be installed from three sources:

### Git (Recommended)

Install directly from a Git repository:

```
git+https://github.com/user/plugin-repo.git
```

You can also specify a branch, tag, or commit:

```
git+https://github.com/user/plugin-repo.git@v1.0.0
git+https://github.com/user/plugin-repo.git@main
git+https://github.com/user/plugin-repo.git@abc1234
```

### PyPI

Install from the Python Package Index:

```
my-scope-plugin
```

You can optionally specify a version:

```
my-scope-plugin==1.0.0
```

### Local

Install from a local directory (useful for development):

```
/path/to/my-plugin
```

On Windows:

```
C:\Users\username\projects\my-plugin
```

Local plugins are installed in editable mode, meaning changes to the source code take effect after reloading the plugin.

## Developing Plugins

_TBD_
