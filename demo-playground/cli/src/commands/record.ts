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
  },
) {
  const outputPath = path.resolve(opts.output);
  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });

  // Playwright records to a directory; we'll move the file afterwards
  const videoDir = fs.mkdtempSync('/tmp/alkahest-rec-');

  console.log(chalk.bold('\nRecording notebook demo'));
  console.log(chalk.dim(`  URL: ${opts.url}`));
  console.log(chalk.dim(`  Output: ${outputPath}\n`));

  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({
    viewport: { width: Number(opts.width), height: Number(opts.height) },
    recordVideo: {
      dir: videoDir,
      size: { width: Number(opts.width), height: Number(opts.height) },
    },
  });

  const page = await context.newPage();
  await page.goto(opts.url, { waitUntil: 'networkidle' });

  // Wait for the first CodeMirror editor to appear
  await page.waitForSelector('.cm-editor', { timeout: 15_000 });
  await delay(1000);

  if (opts.code) {
    const code = fs.readFileSync(opts.code, 'utf-8');
    const cells = code.split('\n# ---\n'); // split on `# ---` delimiters

    for (let i = 0; i < cells.length; i++) {
      const cellCode = cells[i].trim();

      // Add a new cell if this isn't the first one
      if (i > 0) {
        await page.click('button[title="Add cell below"]');
        await delay(300);
      }

      // Focus the last editor
      const editors = await page.$$('.cm-editor');
      const editor = editors[editors.length - 1];
      await editor.click();
      await delay(200);

      // Type code character by character for visual effect
      await typeSlowly(page, cellCode, Number(opts.delay));
      await delay(500);

      // Run the cell
      await page.keyboard.press('Meta+Enter');
      await delay(2000); // wait for output

      // Wait for "done" state (execution count badge appears)
      await page.waitForSelector('[class*="execution"]', { timeout: 30_000 }).catch(() => {});
      await delay(1000);
    }
  }

  await delay(2000);

  // Close context — this finalizes the video
  await context.close();
  await browser.close();

  // Find and move the recorded video
  const videos = fs.readdirSync(videoDir).filter((f) => f.endsWith('.webm'));
  if (videos.length === 0) {
    console.error(chalk.red('No video file found.'));
    process.exit(1);
  }

  fs.renameSync(path.join(videoDir, videos[0]), outputPath);
  fs.rmdirSync(videoDir, { recursive: true } as never);

  console.log(chalk.green(`\n✓ Saved recording: ${outputPath}`));
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
