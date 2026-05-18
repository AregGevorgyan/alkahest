import { streamText, tool } from 'ai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createMistral } from '@ai-sdk/mistral';
import { z } from 'zod';
import { ALKAHEST_SYSTEM_PROMPT } from '@/lib/alkahest-skill';
import type { OutputItem } from '@/lib/execution';

export const runtime = 'nodejs';
export const maxDuration = 120;

function getLanguageModel(provider: string, model: string) {
  switch (provider) {
    case 'openai': {
      const client = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });
      return client(model);
    }
    case 'google': {
      const client = createGoogleGenerativeAI({ apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY });
      return client(model);
    }
    case 'mistral': {
      const client = createMistral({ apiKey: process.env.MISTRAL_API_KEY });
      return client(model as Parameters<ReturnType<typeof createMistral>>[0]);
    }
    case 'anthropic':
    default: {
      const client = createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
      return client(model as Parameters<ReturnType<typeof createAnthropic>>[0]);
    }
  }
}

export async function POST(req: Request) {
  const body = await req.json() as {
    messages: unknown[];
    provider?: string;
    model?: string;
    serverHttpUrl?: string;
    sessionId?: string;
  };

  const provider = body.provider ?? process.env.AI_PROVIDER ?? 'anthropic';
  const model = body.model ?? process.env.AI_MODEL ?? 'claude-sonnet-4-6';
  const serverHttpUrl = body.serverHttpUrl ?? process.env.PYTHON_SERVER_URL ?? 'http://localhost:8000';
  const sessionId = body.sessionId;

  const languageModel = getLanguageModel(provider, model);

  const result = streamText({
    model: languageModel,
    system: ALKAHEST_SYSTEM_PROMPT,
    messages: body.messages as Parameters<typeof streamText>[0]['messages'],
    maxSteps: 12,
    tools: {
      run_python: tool({
        description:
          'Execute Python code on the alkahest server kernel. Use this to run alkahest, SymPy, numpy, matplotlib, or any Python computation. The kernel is stateful — variables persist between calls.',
        parameters: z.object({
          code: z.string().describe('Python code to execute'),
        }),
        execute: async ({ code }) => {
          if (!sessionId) {
            return { outputs: [{ type: 'error', ename: 'NoSession', evalue: 'No kernel session available.', traceback: [] }] as OutputItem[] };
          }

          try {
            const res = await fetch(`${serverHttpUrl}/sessions/${sessionId}/run`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ code }),
              signal: AbortSignal.timeout(60_000),
            });

            if (!res.ok) {
              const text = await res.text();
              return {
                outputs: [{ type: 'error', ename: 'ServerError', evalue: text, traceback: [] }] as OutputItem[],
              };
            }

            const data = await res.json() as { outputs: OutputItem[] };
            return data;
          } catch (e) {
            return {
              outputs: [{ type: 'error', ename: 'NetworkError', evalue: String(e), traceback: [] }] as OutputItem[],
            };
          }
        },
      }),
    },
  });

  return result.toDataStreamResponse();
}
