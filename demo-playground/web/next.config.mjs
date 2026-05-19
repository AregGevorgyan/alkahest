/** @type {import('next').NextConfig} */
const isStaticHosting = process.env.NEXT_PUBLIC_STATIC_HOSTING === 'true';
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

const nextConfig = {
  ...(isStaticHosting
    ? {
        output: 'export',
        basePath: basePath || undefined,
        trailingSlash: true,
        images: { unoptimized: true },
      }
    : {}),
  // COOP/COEP for Pyodide threading when running the Next dev/prod server (not on static GH Pages).
  ...(!isStaticHosting
    ? {
        async headers() {
          return [
            {
              source: '/(.*)',
              headers: [
                { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
                { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
              ],
            },
          ];
        },
      }
    : {}),
};

export default nextConfig;
