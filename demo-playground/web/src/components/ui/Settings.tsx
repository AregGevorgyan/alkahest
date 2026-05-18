'use client';

import { useEffect, useState } from 'react';

const PROVIDERS = ['anthropic', 'openai', 'google', 'mistral'] as const;
const EXECUTION_MODES = ['auto', 'wasm', 'server'] as const;

export type ProviderName = (typeof PROVIDERS)[number];
export type ExecutionMode = (typeof EXECUTION_MODES)[number];

export interface PlaygroundConfig {
  serverHttpUrl: string;
  serverWsUrl: string;
  executionMode: ExecutionMode;
  aiProvider: ProviderName;
  aiModel: string;
}

const DEFAULT_CONFIG: PlaygroundConfig = {
  serverHttpUrl: 'http://localhost:8000',
  serverWsUrl: 'ws://localhost:8000',
  executionMode: 'auto',
  aiProvider: 'anthropic',
  aiModel: 'claude-sonnet-4-6',
};

const STORAGE_KEY = 'alkahest-playground-config';

export function loadConfig(): PlaygroundConfig {
  if (typeof window === 'undefined') return DEFAULT_CONFIG;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? { ...DEFAULT_CONFIG, ...JSON.parse(raw) } : DEFAULT_CONFIG;
  } catch {
    return DEFAULT_CONFIG;
  }
}

export function saveConfig(cfg: PlaygroundConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
}

interface SettingsProps {
  onClose: () => void;
}

export default function Settings({ onClose }: SettingsProps) {
  const [cfg, setCfg] = useState<PlaygroundConfig>(DEFAULT_CONFIG);
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'ok' | 'fail'>('idle');

  useEffect(() => {
    setCfg(loadConfig());
  }, []);

  function update<K extends keyof PlaygroundConfig>(key: K, value: PlaygroundConfig[K]) {
    setCfg((prev) => {
      const next = { ...prev, [key]: value };
      // Keep WS URL in sync with HTTP URL if user edits HTTP
      if (key === 'serverHttpUrl') {
        next.serverWsUrl = (value as string).replace(/^https?/, (p) => (p === 'https' ? 'wss' : 'ws'));
      }
      return next;
    });
  }

  async function testConnection() {
    setTestStatus('testing');
    try {
      const res = await fetch(`${cfg.serverHttpUrl}/health`, { signal: AbortSignal.timeout(3000) });
      setTestStatus(res.ok ? 'ok' : 'fail');
    } catch {
      setTestStatus('fail');
    }
  }

  function handleSave() {
    saveConfig(cfg);
    onClose();
    // Reload to apply config changes
    window.location.reload();
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-lg border border-ak-border bg-ak-bg p-6 shadow-xl">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-base font-semibold">Settings</h2>
          <button onClick={onClose} className="text-ak-muted hover:text-ak-fg">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-4">
          {/* Backend */}
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ak-muted">
              Execution backend
            </h3>
            <label className="block text-sm mb-1">Server URL</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={cfg.serverHttpUrl}
                onChange={(e) => update('serverHttpUrl', e.target.value)}
                className="flex-1 rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ak-brand"
              />
              <button
                onClick={testConnection}
                className="rounded border border-ak-border px-3 py-1.5 text-xs hover:bg-ak-code-bg"
              >
                {testStatus === 'testing' ? '…' : testStatus === 'ok' ? '✓ OK' : testStatus === 'fail' ? '✗ fail' : 'Test'}
              </button>
            </div>
          </section>

          <section>
            <label className="block text-sm mb-1">Execution mode</label>
            <select
              value={cfg.executionMode}
              onChange={(e) => update('executionMode', e.target.value as ExecutionMode)}
              className="w-full rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ak-brand"
            >
              <option value="auto">Auto (WASM for pure Python, server for alkahest)</option>
              <option value="wasm">WASM only (Pyodide)</option>
              <option value="server">Server only</option>
            </select>
          </section>

          {/* AI */}
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ak-muted">
              Agent AI
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm mb-1">Provider</label>
                <select
                  value={cfg.aiProvider}
                  onChange={(e) => update('aiProvider', e.target.value as ProviderName)}
                  className="w-full rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-ak-brand"
                >
                  {PROVIDERS.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm mb-1">Model</label>
                <input
                  type="text"
                  value={cfg.aiModel}
                  onChange={(e) => update('aiModel', e.target.value)}
                  className="w-full rounded border border-ak-border bg-ak-code-bg px-3 py-1.5 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ak-brand"
                  placeholder="e.g. claude-sonnet-4-6"
                />
              </div>
            </div>
          </section>
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded border border-ak-border px-4 py-1.5 text-sm hover:bg-ak-code-bg"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="rounded bg-ak-brand px-4 py-1.5 text-sm font-medium text-white hover:bg-ak-brand-dark"
          >
            Save & reload
          </button>
        </div>
      </div>
    </div>
  );
}
