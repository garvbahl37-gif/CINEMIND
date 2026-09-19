// Same-origin in production: the API ships as Vercel Python Functions alongside
// this app, so there is no cross-origin hop and no separate backend to wake up.
export const API_BASE = import.meta.env.VITE_API_BASE ?? '';
export const IMG = 'https://image.tmdb.org/t/p';

export const poster = (p?: string | null, size: 'w185' | 'w342' | 'w500' = 'w342') =>
  p ? `${IMG}/${size}${p}` : null;
export const backdrop = (p?: string | null, size: 'w780' | 'original' = 'w780') =>
  p ? `${IMG}/${size}${p}` : null;
