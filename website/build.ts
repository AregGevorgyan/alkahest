#!/usr/bin/env bun

import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";

const repoRoot = import.meta.dir;
const srcRoot = join(repoRoot, "src");
const artifacts = ["index.html", "styles.css"];

for (const name of artifacts) {
  const from = join(srcRoot, name);
  const to = join(repoRoot, name);
  await mkdir(dirname(to), { recursive: true });
  await copyFile(from, to);
}

console.warn("Copied", artifacts.join(", "), "→ website/");
