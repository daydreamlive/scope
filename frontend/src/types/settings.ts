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
}
