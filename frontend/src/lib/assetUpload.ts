import {
  getAssetUrl,
  listAssets,
  uploadAsset as uploadAssetFile,
  type AssetFileInfo,
} from "./api";

export type AssetKind = "image" | "video" | "audio";

export interface UploadCache {
  assetsByKind: Partial<Record<AssetKind, AssetFileInfo[]>>;
  uploads: Map<string, Promise<string>>;
}

export interface AssetFieldSpec {
  key: string;
  kind: AssetKind;
}

const WINDOWS_ABSOLUTE_PATH_RE = /^[A-Za-z]:[\\/]/;

export function createUploadCache(): UploadCache {
  return {
    assetsByKind: {},
    uploads: new Map(),
  };
}

function getBasename(assetPath: string): string {
  const parts = assetPath.split(/[/\\]/);
  return parts[parts.length - 1] || assetPath;
}

function isFilesystemAssetPath(assetPath: string): boolean {
  if (!assetPath) return false;
  if (
    assetPath.startsWith("blob:") ||
    assetPath.startsWith("data:") ||
    assetPath.startsWith("http://") ||
    assetPath.startsWith("https://")
  ) {
    return false;
  }
  if (WINDOWS_ABSOLUTE_PATH_RE.test(assetPath)) return true;
  if (!assetPath.startsWith("/")) return false;
  return true;
}

async function getAssetsForKind(
  kind: AssetKind,
  cache: UploadCache
): Promise<AssetFileInfo[]> {
  const cached = cache.assetsByKind[kind];
  if (cached) {
    return cached;
  }
  const response = await listAssets(kind);
  cache.assetsByKind[kind] = response.assets;
  return response.assets;
}

function guessMimeType(assetPath: string, kind: AssetKind): string {
  const ext = assetPath.split(".").pop()?.toLowerCase() ?? "";
  if (kind === "image") {
    if (ext === "png") return "image/png";
    if (ext === "webp") return "image/webp";
    if (ext === "bmp") return "image/bmp";
    return "image/jpeg";
  }
  if (kind === "audio") {
    if (ext === "wav") return "audio/wav";
    if (ext === "flac") return "audio/flac";
    if (ext === "ogg") return "audio/ogg";
    return "audio/mpeg";
  }
  if (ext === "webm") return "video/webm";
  if (ext === "mov") return "video/quicktime";
  return "video/mp4";
}

export async function assetPath(
  path: string,
  kind: AssetKind,
  cache: UploadCache
): Promise<string> {
  if (!path || !isFilesystemAssetPath(path)) {
    return path;
  }

  const assets = await getAssetsForKind(kind, cache);
  const exactMatch = assets.find(asset => asset.path === path);
  if (exactMatch) {
    return exactMatch.path;
  }

  const uploadKey = `${kind}\0${path}`;
  const existingUpload = cache.uploads.get(uploadKey);
  if (existingUpload) {
    return existingUpload;
  }

  const uploadPromise = (async () => {
    const response = await fetch(getAssetUrl(path));
    if (!response.ok) {
      throw new Error(
        `Could not read local asset '${path}' (${response.status})`
      );
    }

    const blob = await response.blob();
    const file = new File([blob], getBasename(path), {
      type: blob.type || guessMimeType(path, kind),
    });
    const uploaded = await uploadAssetFile(file);

    if (cache.assetsByKind[kind]) {
      cache.assetsByKind[kind] = [uploaded, ...cache.assetsByKind[kind]!];
    }

    return uploaded.path;
  })();

  cache.uploads.set(uploadKey, uploadPromise);

  try {
    return await uploadPromise;
  } catch (error) {
    cache.uploads.delete(uploadKey);
    throw error;
  }
}

export async function rewriteAssetFields<T extends Record<string, unknown>>(
  params: T,
  specs: AssetFieldSpec[],
  cache: UploadCache
): Promise<T> {
  const nextParams: Record<string, unknown> = { ...params };

  for (const spec of specs) {
    const value = nextParams[spec.key];
    if (value == null) continue;

    if (Array.isArray(value)) {
      const rewritten = await Promise.all(
        value.map(item =>
          typeof item === "string" ? assetPath(item, spec.kind, cache) : item
        )
      );
      nextParams[spec.key] = rewritten;
      continue;
    }

    if (typeof value !== "string") continue;
    nextParams[spec.key] = await assetPath(value, spec.kind, cache);
  }

  return nextParams as T;
}
