/**
 * Everything the dashboard reads from the API, through one normaliser.
 *
 * The API already carries artwork, synopses, runtime and credits for every
 * film, so nothing here calls TMDB directly — the previous cards each made
 * their own TMDB request, which on the home page was several hundred.
 */
import { API_BASE, TMDB_IMAGE_BASE } from '../config';
import { Movie } from '../types';

type Size = 'w185' | 'w342' | 'w500' | 'w780' | 'w1280' | 'original';

export const img = (path: string | undefined, size: Size = 'w342') =>
    !path ? undefined : path.startsWith('http') ? path : `${TMDB_IMAGE_BASE}/${size}${path}`;

/** Accepts the native shape and the compat shape the chat endpoint returns. */
export const toMovie = (m: any): Movie => ({
    item_id: m.item_id,
    tmdbId: m.tmdbId,
    title: m.title,
    year: m.year ?? (m.releaseDate ? Number(String(m.releaseDate).slice(0, 4)) : undefined),
    posterPath: m.poster_path ?? undefined,
    backdropPath: m.backdrop_path ?? undefined,
    overview: m.overview,
    genres: m.genres ?? [],
    tags: m.tags ?? [],
    runtime: m.runtime || undefined,
    media_type: m.media_type,
    original_language: m.original_language,
    rating: m.rating_avg ?? m.vote_average ?? undefined,
    rating_count: m.rating_count ?? m.vote_count,
    directors: m.directors ?? [],
    cast: m.cast ?? [],
    score: m.score ?? undefined,
    shared_genres: m.shared_genres,
    shared_tags: m.shared_tags,
});

async function get(path: string, init?: RequestInit) {
    const res = await fetch(`${API_BASE}${path}`, init);
    if (!res.ok) throw new Error(`${path} returned ${res.status}`);
    // A misrouted deploy serves index.html with a 200 for every /api path, so
    // check what came back rather than trusting the status.
    if (!(res.headers.get('content-type') ?? '').includes('json')) {
        throw new Error(`${path} did not return JSON`);
    }
    return res.json();
}

export interface Browse {
    hero: Movie[];
    top50: Movie[];
    tv: Movie[];
    rows: { genre: string; items: Movie[] }[];
}

export async function fetchBrowse(): Promise<Browse> {
    const d = await get('/api/browse');
    return {
        hero: d.hero.map(toMovie),
        top50: d.top50.map(toMovie),
        tv: d.tv.map(toMovie),
        rows: d.rows.map((r: any) => ({ genre: r.genre, items: r.items.map(toMovie) })),
    };
}

export async function fetchSimilar(itemId: number, k = 24, signal?: AbortSignal) {
    const d = await get(`/api/similar/${itemId}?k=${k}`, { signal });
    return { source: toMovie(d.source), matches: (d.similar_items ?? []).map(toMovie) as Movie[] };
}

/** Recommendations from several films at once. */
export async function fetchBlend(seeds: number[], k = 24, signal?: AbortSignal): Promise<Movie[]> {
    const d = await get('/api/blend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seeds, k }),
        signal,
    });
    return (d.results ?? []).map(toMovie);
}
