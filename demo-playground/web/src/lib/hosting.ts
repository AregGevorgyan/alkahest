/** True when built for static hosting (e.g. GitHub Pages). */
export const isStaticHosting = process.env.NEXT_PUBLIC_STATIC_HOSTING === 'true';

/** Base path prefix (e.g. `/playground` on GitHub Pages). */
export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

/** Resolve a public asset path (worker, etc.) under basePath. */
export function publicAssetPath(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${basePath}${p}`;
}
