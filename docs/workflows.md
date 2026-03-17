# Workflow Import/Export

Scope allows you to export your complete pipeline configuration as a shareable workflow file and import workflows from others. A workflow captures the pipeline, parameters, LoRAs, prompt timeline, and plugin dependencies so that someone else can reproduce your exact setup.

## Exporting a Workflow

1. Configure your pipeline, parameters, LoRAs, and prompt timeline as desired.
2. Click the **Export** button in the app header.
3. Enter a name for the workflow.
4. Click **Export** to download a `.scope-workflow.json` file.

The exported file contains:

- Pipeline ID and source (builtin or plugin)
- All parameter values (denoising steps, guidance scale, resolution, etc.)
- LoRA configurations with weights, merge modes, and provenance (download source)
- Prompt timeline entries with interpolation settings
- Preprocessor and postprocessor pipelines, if configured
- Scope version metadata

## Importing a Workflow

1. Click the **Import** button in the app header.
2. Select or drag-and-drop a `.scope-workflow.json` file.
3. Review the dependency resolution summary.
4. Resolve any missing dependencies (see below).
5. Click **Load** to apply the workflow.

> **Note:** Loading a workflow replaces your current settings and timeline.

### Dependency Resolution

When you import a workflow, Scope checks your environment against the workflow's requirements and shows the status of each dependency:

| Dependency | What is checked |
|------------|-----------------|
| Pipeline | Whether the required pipeline is registered (builtin or via plugin) |
| Plugin | Whether the required plugin is installed and version-compatible |
| LoRA | Whether the required LoRA file exists in `~/.daydream-scope/models/lora` |

Each item shows one of three statuses:

- **OK** — available and compatible
- **Missing** — not found; may be auto-resolvable
- **Version mismatch** — installed but different version than expected

### Installing Missing Dependencies

**LoRAs** — If the workflow includes provenance information (HuggingFace repo, CivitAI model, or URL), you can download missing LoRAs directly from the import dialog. Click **Download** next to individual LoRAs or **Download All Missing LoRAs** to fetch them all.

**Plugins** — Missing plugins that have an install spec can be installed from the import dialog. Click **Install** next to a plugin and confirm the installation. The server restarts automatically after each plugin install.

> [!WARNING]
> Only install plugins from sources you trust. Plugin installation runs `pip install` with the package spec from the workflow file.

The **Load** button is disabled until all required dependencies are resolved.

## Workflow File Format

Workflow files use the `.scope-workflow.json` extension and contain a JSON object with the following top-level fields:

| Field | Description |
|-------|-------------|
| `format` | Always `"scope-workflow"` |
| `format_version` | Schema version (currently `"1.0"`) |
| `metadata` | Name, creation timestamp, and Scope version |
| `pipelines` | Array of pipeline configurations with params, LoRAs, and source info |
| `timeline` | Prompt timeline entries with interpolation settings |
| `prompts` | Top-level prompt list (used when no timeline is present) |

### Limitations

The following are not yet included in workflow files:

- Reference images, first frame, and last frame images
- Plugin version resolution
- Cloud-hosted workflow sharing
