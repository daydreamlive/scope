export interface InstalledPlugin {
  name: string;
  version: string | null;
  author: string | null;
  description: string | null;
  source?: "pypi" | "git" | "local";
  editable?: boolean;
  latest_version?: string | null;
  update_available?: boolean | null;
  package_spec?: string | null;
  bundled?: boolean;
  /** Plugin kind from the package's `__scope_kind__` attribute.
   *  `"source"` means the plugin runs on the local machine even in
   *  cloud mode. */
  kind?: string | null;
  /** Where this plugin record came from (only set in cloud mode). */
  origin?: "local" | "cloud" | null;
}
