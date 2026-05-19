#!/usr/bin/env node
import { execSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { rename, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const apiDir = join(root, 'src/app/api');
const apiBak = join(root, 'src/app/api.__build_bak');

let moved = false;
try {
  if (existsSync(apiDir)) {
    await rename(apiDir, apiBak);
    moved = true;
  }

  execSync('pnpm exec next build', {
    cwd: root,
    stdio: 'inherit',
    env: {
      ...process.env,
      NEXT_PUBLIC_STATIC_HOSTING: 'true',
      NEXT_PUBLIC_BASE_PATH: '/playground',
    },
  });

  await writeFile(join(root, 'out/.nojekyll'), '');
  console.log('Static export written to web/out/');
} finally {
  if (moved && existsSync(apiBak)) {
    await rename(apiBak, apiDir);
  }
}
