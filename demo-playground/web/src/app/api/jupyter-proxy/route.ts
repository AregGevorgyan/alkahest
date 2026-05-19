export const runtime = 'nodejs';

function isAllowedBaseUrl(raw: string): boolean {
  try {
    const u = new URL(raw);
    return u.protocol === 'http:' || u.protocol === 'https:';
  } catch {
    return false;
  }
}

export async function POST(req: Request) {
  const body = await req.json() as {
    baseUrl?: string;
    token?: string;
    path?: string;
    method?: string;
    body?: unknown;
  };

  const baseUrl = body.baseUrl?.replace(/\/$/, '');
  const path = body.path ?? '/api';
  const method = (body.method ?? 'GET').toUpperCase();

  if (!baseUrl || !isAllowedBaseUrl(baseUrl)) {
    return new Response(JSON.stringify({ error: 'Invalid baseUrl' }), { status: 400 });
  }

  const target = new URL(path.startsWith('/') ? path : `/${path}`, baseUrl);
  if (body.token) target.searchParams.set('token', body.token);

  const headers: Record<string, string> = { Accept: 'application/json' };
  if (body.token) headers.Authorization = `token ${body.token}`;

  let fetchBody: string | undefined;
  if (body.body !== undefined && method !== 'GET' && method !== 'HEAD') {
    headers['Content-Type'] = 'application/json';
    fetchBody = JSON.stringify(body.body);
  }

  try {
    const upstream = await fetch(target.toString(), {
      method,
      headers,
      body: fetchBody,
      signal: AbortSignal.timeout(30_000),
    });

    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { 'Content-Type': upstream.headers.get('content-type') ?? 'application/json' },
    });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return new Response(JSON.stringify({ error: message }), { status: 502 });
  }
}
