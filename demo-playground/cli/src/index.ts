#!/usr/bin/env node
import { Command } from 'commander';
import chalk from 'chalk';
import { startCommand } from './commands/start.js';
import { recordCommand } from './commands/record.js';
import { demoCommand } from './commands/demo.js';

const program = new Command();

program
  .name('alkahest-demo')
  .description(
    chalk.bold('Alkahest Demo Playground CLI') +
    '\nLaunch, record, and orchestrate demos of the Alkahest CAS library.',
  )
  .version('0.1.0');

program
  .command('start')
  .description('Start the web app and Python execution server')
  .option('-p, --web-port <port>', 'Next.js port', '3000')
  .option('-s, --server-port <port>', 'Python server port', '8000')
  .option('--no-open', 'Do not open the browser')
  .action(startCommand);

program
  .command('record')
  .description('Record a scripted notebook demo as a video')
  .option('-c, --code <file>', 'Python file to inject into cells')
  .option('-o, --output <file>', 'Output video file', `alkahest-demo-${Date.now()}.webm`)
  .option('--url <url>', 'Playground URL', 'http://localhost:3000')
  .option('--width <px>', 'Viewport width', '1280')
  .option('--height <px>', 'Viewport height', '720')
  .option('--delay <ms>', 'Delay between typing characters (ms)', '40')
  .action(recordCommand);

program
  .command('demo <prompt>')
  .description('Tell an AI agent to demonstrate something and capture the result as a video')
  .option('-o, --output <file>', 'Output video file', `alkahest-agent-demo-${Date.now()}.webm`)
  .option('--url <url>', 'Playground URL', 'http://localhost:3000')
  .option('--server <url>', 'Python server URL', 'http://localhost:8000')
  .option('--width <px>', 'Viewport width', '1280')
  .option('--height <px>', 'Viewport height', '720')
  .option('--wait <ms>', 'Max time to wait for agent response (ms)', '120000')
  .option('--no-start', 'Do not auto-start servers (assume they are already running)')
  .action(demoCommand);

program.parse();
