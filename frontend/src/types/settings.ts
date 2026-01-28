export interface InstalledPlugin {
  id: string;
  name: string;
  version: string;
  author: string;
  description: string;
}

export const MOCK_PLUGINS: InstalledPlugin[] = [
  {
    id: "1",
    name: "scope-overworld",
    version: "0.1.0",
    author: "daydreamlive",
    description: "Overworld world models like Waypoint-1",
  },
  {
    id: "2",
    name: "scope-flashvsr2",
    version: "0.1.0",
    author: "daydreamlive",
    description: "Real-time upscaling with flashvsr2",
  },
];
