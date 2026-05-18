import path from 'path';
import fs from 'fs';
import chalk from 'chalk';
import { chromium } from 'playwright';
import { startCommand } from './start.js';

export async function demoCommand(
  prompt: string,
  opts: {
    output: string;
    url: string;
    server: string;
    width: string;
    height: string;
    wait: string;
    start: boolean;
  },
) {
  const outputPath = path.resolve(opts.output);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  console.log(chalk.bold('\nalkahest agent demo\n'));
  console.log(chalk.dim(`  Prompt: "${prompt}"`));
  console.log(chalk.dim(`  Output: ${outputPath}\n`));

  // ── Optionally start servers ─────────────────────────────────────────────
  if (opts.start) {
    console.log(chalk.cyan('→ Starting servers…'));
    // Fire and forget — start runs a blocking await internally
    startCommand({ webPort: '3000', serverPort: '8000', open: false }).catch(() => {});
    // Wait for servers to be ready
    await waitForUrl(`${opts.url}/api/health-check-noop`, 30_000).catch(() => {});
    await waitForUrl(`${opts.server}/health`, 30_000).catch(() => {});
    await delay(2000);
  }

  const videoDir = fs.mkdtempSync('/tmp/alkahest-demo-');

  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({
    viewport: { width: Number(opts.width), height: Number(opts.height) },
    recordVideo: {
      dir: videoDir,
      size: { width: Number(opts.width), height: Number(opts.height) },
    },
  });

  const page = await context.newPage();

  console.log(chalk.cyan('→ Navigating to agent page…'));
  await page.goto(`${opts.url}/agent`, { waitUntil: 'networkidle' });
  await delay(2000);

  // Wait for the input box
  await page.waitForSelector('input[placeholder*="agent"]', { timeout: 15_000 });
  await delay(500);

  // Type the prompt
  console.log(chalk.cyan('→ Sending prompt to agent…'));
  const input = page.locator('input[placeholder*="agent"]');
  await input.click();
  await typeSlowly(page, prompt, 30);
  await delay(500);

  // Submit
  await page.keyboard.press('Enter');

  // Wait for agent to finish (no more loading indicators)
  console.log(chalk.cyan('→ Waiting for agent response…'));
  const maxWait = Number(opts.wait);
  const deadline = Date.now() + maxWait;

  while (Date.now() < deadline) {
    await delay(2000);
    const isLoading = await page.$('button:has-text("Stop")');
    if (!isLoading) break;
  }

  // A bit more to show the final state
  await delay(3000);

  await context.close();
  await browser.close();

  // Move video
  const videos = fs.readdirSync(videoDir).filter((f) => f.endsWith('.webm'));
  if (!videos.length) {
    console.error(chalk.red('No video captured.'));
    process.exit(1);
  }

  const src = path.join(videoDir, videos[0]);
  fs.copyFileSync(src, outputPath);
  fs.unlinkSync(src);
  try { fs.rmdirSync(videoDir); } catch {}

  console.log(chalk.green(`\n✓ Demo saved: ${outputPath}\n`));
}

async function waitForUrl(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return;
    } catch {}
    await delay(1000);
  }
}

async function typeSlowly(page: import('playwright').Page, text: string, delayMs: number) {
  for (const char of text) {
    await page.keyboard.type(char);
    await delay(delayMs);
  }
}

function delay(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
