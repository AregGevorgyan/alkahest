export type ServerBackend = 'alkahest' | 'jupyter';

export interface ServerConnection {
  backend: ServerBackend;
  httpUrl: string;
  wsUrl: string;
  token: string;
}

export interface ServerConfigSlice {
  serverBackend?: ServerBackend;
  serverHttpUrl: string;
  serverWsUrl: string;
  serverToken?: string;
}

export function connectionFromConfig(cfg: ServerConfigSlice): ServerConnection {
  return {
    backend: cfg.serverBackend ?? 'alkahest',
    httpUrl: cfg.serverHttpUrl.replace(/\/$/, ''),
    wsUrl: cfg.serverWsUrl.replace(/\/$/, ''),
    token: (cfg.serverToken ?? '').trim(),
  };
}

/** Auth headers for the alkahest FastAPI backend. */
export function alkahestAuthHeaders(token: string): HeadersInit {
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

/** Append Jupyter token query param when needed. */
export function jupyterUrlWithToken(url: string, token: string): string {
  if (!token) return url;
  const u = new URL(url);
  u.searchParams.set('token', token);
  return u.toString();
}

export function healthPath(backend: ServerBackend): string {
  return backend === 'jupyter' ? '/api' : '/health';
}
