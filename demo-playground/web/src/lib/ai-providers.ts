/** AI providers supported by the demo playground (Vercel AI SDK). */

export const AI_PROVIDERS = [
  { id: 'anthropic', label: 'Anthropic', defaultModel: 'claude-sonnet-4-6' },
  { id: 'openai', label: 'OpenAI', defaultModel: 'gpt-4o' },
  { id: 'google', label: 'Google Generative AI', defaultModel: 'gemini-2.0-flash' },
  { id: 'mistral', label: 'Mistral', defaultModel: 'mistral-large-latest' },
  { id: 'groq', label: 'Groq', defaultModel: 'llama-3.3-70b-versatile' },
  { id: 'xai', label: 'xAI (Grok)', defaultModel: 'grok-3-mini' },
  { id: 'deepseek', label: 'DeepSeek', defaultModel: 'deepseek-chat' },
  { id: 'together', label: 'Together.ai', defaultModel: 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo' },
  { id: 'fireworks', label: 'Fireworks', defaultModel: 'accounts/fireworks/models/llama-v3p1-8b-instruct' },
  { id: 'cerebras', label: 'Cerebras', defaultModel: 'llama-3.3-70b' },
  {
    id: 'openai-compatible',
    label: 'OpenAI-compatible (custom)',
    defaultModel: '',
    customEndpoint: true,
  },
] as const;

export type ProviderId = (typeof AI_PROVIDERS)[number]['id'];

export function getProviderMeta(id: string) {
  return AI_PROVIDERS.find((p) => p.id === id);
}

export function defaultModelForProvider(id: string): string {
  return getProviderMeta(id)?.defaultModel ?? '';
}

export function isKnownProvider(id: string): id is ProviderId {
  return AI_PROVIDERS.some((p) => p.id === id);
}
