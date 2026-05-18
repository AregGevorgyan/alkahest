import path from 'path';
import fs from 'fs';
import chalk from 'chalk';
import { chromium } from 'playwright';

export async function recordCommand(
  opts: {
    code?: string;
    output: string;
    url: string;
    width: string;
    height: string;
    delay: string;
    headless?: boolean;
  },
) {
  const outputPath = path.resolve(opts.output);
  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });

  const videoDir = fs.mkdtempSync('/tmp/alkahest-rec-');
  const headless = opts.headless ?? !process.env.DISPLAY;

  console.log(chalk.bold('\nRecording notebook demo'));
  console.log(chalk.dim(`  URL:      ${opts.url}`));
  console.log(chalk.dim(`  Output:   ${outputPath}`));
  console.log(chalk.dim(`  Headless: ${headless}\n`));

  // Encode cells into ?demo= URL param so the Notebook pre-populates them
  let targetUrl = opts.url;
  let numCells = 0;
  if (opts.code) {
    const raw = fs.readFileSync(opts.code, 'utf-8');
    const cellCodes = raw.split(/\n# ---\n/).map((c) => c.trim()).filter(Boolean);
    numCells = cellCodes.length;
    const encoded = Buffer.from(JSON.stringify(cellCodes)).toString('base64');
    // Force server mode so continuation cells (no import statement) still hit the kernel
    targetUrl = `${opts.url}?demo=${encoded}&mode=server`;
    console.log(chalk.dim(`  Cells: ${numCells}\n`));
  }

  const browser = await chromium.launch({ headless });
  const context = await browser.newContext({
    viewport: { width: Number(opts.width), height: Number(opts.height) },
    recordVideo: {
      dir: videoDir,
      size: { width: Number(opts.width), height: Number(opts.height) },
    },
  });

  const page = await context.newPage();
  await page.goto(targetUrl, { waitUntil: 'networkidle', timeout: 30_000 });
  await page.waitForSelector('.cm-editor', { timeout: 20_000 });
  console.log(chalk.cyan('  Page loaded'));

  // Brief pause so the first frame shows the loaded cells
  await delay(1500);

  // Click "Run all" to execute all cells with the notebook's natural stagger
  await page.click('button:has-text("Run all")');
  console.log(chalk.cyan('  Running cells…'));

  // Wait until every cell spinner is gone (all cells done)
  await page.waitForFunction(() => {
    return document.querySelectorAll('.animate-spin').length === 0;
  }, { timeout: 60_000, polling: 500 }).catch(() => {
    console.log(chalk.yellow('  Warning: timed out waiting for cells to finish'));
  });

  // Extra pause — wait for any async output rendering (KaTeX, images)
  await delay(1500);

  console.log(chalk.green('  All cells done — holding final frame'));
  await delay(2000);

  await context.close();
  await browser.close();

  // Move video — use copy+delete to handle cross-device filesystems
  const videos = fs.readdirSync(videoDir).filter((f) => f.endsWith('.webm'));
  if (videos.length === 0) {
    console.error(chalk.red('No video captured.'));
    process.exit(1);
  }

  const src = path.join(videoDir, videos[0]);
  fs.copyFileSync(src, outputPath);
  fs.unlinkSync(src);
  try { fs.rmdirSync(videoDir); } catch {}

  console.log(chalk.green(`\n✓ Saved: ${outputPath}`));
}

function delay(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
