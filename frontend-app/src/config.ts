// The API ships as Vercel Python Functions alongside this app, so it is
// same-origin: no cross-origin hop, and no separate backend to wake up.
// In development Vite proxies /api (see vite.config.ts).
export const API_BASE = import.meta.env.VITE_API_BASE ?? '';

/** An API read's URL for this build: a new deploy asks new URLs, so no browser keeps
 *  showing the previous deploy's cached answers. */
export const versioned = (path: string) => `${API_BASE}${path}${path.includes('?') ? '&' : '?'}v=${__BUILD_ID__}`;
export const TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p';
