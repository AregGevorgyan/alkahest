import { createAnthropic } from '@ai-sdk/anthropic';
import { createCerebras } from '@ai-sdk/cerebras';
import { createDeepSeek } from '@ai-sdk/deepseek';
import { createFireworks } from '@ai-sdk/fireworks';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createGroq } from '@ai-sdk/groq';
import { createMistral } from '@ai-sdk/mistral';
import { createOpenAI } from '@ai-sdk/openai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { createTogetherAI } from '@ai-sdk/togetherai';
import { createXai } from '@ai-sdk/xai';
export interface LanguageModelOptions {
  /** Base URL for openai-compatible provider (settings or OPENAI_COMPATIBLE_BASE_URL). */
  customBaseUrl?: string;
  /** API key override from settings (falls back to provider env vars). */
  customApiKey?: string;
}

export function getLanguageModel(
  provider: string,
  model: string,
  options: LanguageModelOptions = {},
) {
  switch (provider) {
    case 'openai': {
      const client = createOpenAI({
        apiKey: options.customApiKey ?? process.env.OPENAI_API_KEY,
      });
      return client(model);
    }
    case 'google': {
      const client = createGoogleGenerativeAI({
        apiKey: options.customApiKey ?? process.env.GOOGLE_GENERATIVE_AI_API_KEY,
      });
      return client(model);
    }
    case 'mistral': {
      const client = createMistral({
        apiKey: options.customApiKey ?? process.env.MISTRAL_API_KEY,
      });
      return client(model as Parameters<ReturnType<typeof createMistral>>[0]);
    }
    case 'groq': {
      const client = createGroq({
        apiKey: options.customApiKey ?? process.env.GROQ_API_KEY,
      });
      return client(model);
    }
    case 'xai': {
      const client = createXai({
        apiKey: options.customApiKey ?? process.env.XAI_API_KEY,
      });
      return client(model);
    }
    case 'deepseek': {
      const client = createDeepSeek({
        apiKey: options.customApiKey ?? process.env.DEEPSEEK_API_KEY,
      });
      return client(model);
    }
    case 'together': {
      const client = createTogetherAI({
        apiKey: options.customApiKey ?? process.env.TOGETHER_API_KEY,
      });
      return client(model);
    }
    case 'fireworks': {
      const client = createFireworks({
        apiKey: options.customApiKey ?? process.env.FIREWORKS_API_KEY,
      });
      return client(model);
    }
    case 'cerebras': {
      const client = createCerebras({
        apiKey: options.customApiKey ?? process.env.CEREBRAS_API_KEY,
      });
      return client(model);
    }
    case 'openai-compatible': {
      const baseURL =
        options.customBaseUrl?.trim() || process.env.OPENAI_COMPATIBLE_BASE_URL?.trim();
      const apiKey =
        options.customApiKey?.trim() ||
        process.env.OPENAI_COMPATIBLE_API_KEY?.trim() ||
        process.env.OPENAI_API_KEY?.trim();
      if (!baseURL) {
        throw new Error(
          'OpenAI-compatible provider requires a base URL (Settings or OPENAI_COMPATIBLE_BASE_URL).',
        );
      }
      if (!apiKey) {
        throw new Error(
          'OpenAI-compatible provider requires an API key (Settings or OPENAI_COMPATIBLE_API_KEY).',
        );
      }
      const client = createOpenAICompatible({
        name: 'openai-compatible',
        baseURL,
        apiKey,
      });
      return client(model);
    }
    case 'anthropic':
    default: {
      const client = createAnthropic({
        apiKey: options.customApiKey ?? process.env.ANTHROPIC_API_KEY,
      });
      return client(model as Parameters<ReturnType<typeof createAnthropic>>[0]);
    }
  }
}
