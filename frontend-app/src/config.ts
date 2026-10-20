// The API ships as Vercel Python Functions alongside this app, so it is
// same-origin: no cross-origin hop, and no separate backend to wake up.
// In development Vite proxies /api (see vite.config.ts).
export const API_BASE = import.meta.env.VITE_API_BASE ?? '';
export const TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p';
