import { API_BASE } from './config';
import type { BrowsePayload, Film, SimilarPayload } from './types';

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { signal });
  if (!r.ok) throw new Error(`${path} responded ${r.status}`);
  return r.json() as Promise<T>;
}

export const api = {
  browse: (signal?: AbortSignal) => get<BrowsePayload>('/api/browse', signal),

  similar: (id: number, k = 12, signal?: AbortSignal) =>
    get<SimilarPayload>(`/api/similar/${id}?k=${k}`, signal),

  search: (q: string, signal?: AbortSignal) =>
    get<{ results: Film[]; filters: Record<string, unknown>; latency_ms: number }>(
      `/api/search?q=${encodeURIComponent(q)}&limit=48`, signal),

  suggest: (q: string, signal?: AbortSignal) =>
    get<{ results: Film[] }>(`/api/suggest?q=${encodeURIComponent(q)}`, signal),

  blend: async (seeds: number[], k = 20): Promise<{ results: Film[] }> => {
    const r = await fetch(`${API_BASE}/api/blend`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ seeds, k }),
    });
    if (!r.ok) throw new Error(`blend responded ${r.status}`);
    return r.json();
  },
};
