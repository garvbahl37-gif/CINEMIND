/**
 * Everything the dashboard reads from the API, through one normaliser.
 *
 * The API already carries artwork, synopses, runtime and credits for every
 * film, so nothing here calls TMDB directly — the previous cards each made
 * their own TMDB request, which on the home page was several hundred.
 */
import { API_BASE, TMDB_IMAGE_BASE, versioned } from '../config';
import { MatchBasis, Movie } from '../types';
import type { TasteSignal } from './taste';

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
    imdb_id: m.imdb_id ?? undefined,
    rating_backed: m.rating_backed,
    match_basis: m.match_basis,
    why: m.why,
    reason: m.reason,
    evidence: m.evidence,
    confidence: m.confidence,
    confidence_label: m.confidence_label,
    exploring: m.exploring,
    propensity: m.propensity,
});

async function get(path: string, init?: RequestInit) {
    const res = await fetch(!init?.method || init.method === 'GET' ? versioned(path) : `${API_BASE}${path}`, init);
    if (!res.ok) throw new Error(`${path} returned ${res.status}`);
    // A misrouted deploy serves index.html with a 200 for every /api path, so
    // check what came back rather than trusting the status.
    if (!(res.headers.get('content-type') ?? '').includes('json')) {
        throw new Error(`${path} did not return JSON`);
    }
    return res.json();
}

export interface Browse {
    /** Films in the catalogue. */
    total?: number;
    hero: Movie[];
    top50: Movie[];
    tv: Movie[];
    rows: { genre: string; items: Movie[] }[];
}

/** The home page's data: a static file built with the site (pipelines/build_home.py), so the
 *  site opens without waiting for the API to start; /api/browse answers the same if it's missing. */
async function home(): Promise<any> {
    try {
        const res = await fetch(`/home.json?v=${__BUILD_ID__}`);
        // A dev server answers a missing file with index.html, so check what came back.
        if (res.ok && (res.headers.get('content-type') ?? '').includes('json')) {
            const d = await res.json();
            if (Array.isArray(d?.rows)) {
                // Start the API now, while the page is being looked at, so the first search doesn't wait for it.
                fetch(versioned('/api/health')).catch(() => { });
                return d;
            }
        }
    } catch { /* fall through to the API */ }
    return get('/api/browse');
}

export async function fetchBrowse(): Promise<Browse> {
    const d = await home();
    return {
        total: d.total,
        hero: d.hero.map(toMovie),
        top50: d.top50.map(toMovie),
        tv: d.tv.map(toMovie),
        rows: d.rows.map((r: any) => ({ genre: r.genre, items: r.items.map(toMovie) })),
    };
}

export async function fetchSimilar(itemId: number, k = 24, signal?: AbortSignal) {
    const d = await get(`/api/similar/${itemId}?k=${k}`, { signal });
    return {
        source: toMovie(d.source),
        matches: (d.similar_items ?? []).map(toMovie) as Movie[],
        basis: (d.match_basis ?? 'collaborative') as MatchBasis,
    };
}

export interface ForYou {
    results: Movie[];
    /** Films nobody has rated yet, matched on what they are about — a shelf of their own. */
    unrated: Movie[];
    profile: {
        signals: number; liked: number; skipped: number; searches: number;
        long_term: string[]; lately: string[]; drift: number; adventurousness: number; cold_start: boolean;
    };
    policy: { version?: string; epsilon?: number; diversity?: number; drift?: number; candidates?: number; cold_start?: boolean };
}

/** Personal recommendations from this browser's signals, by the adaptive engine. */
export async function fetchForYou(body: {
    signals: TasteSignal[]; searches: { q: string; ts: number }[]; explore: number; k?: number;
}, signal?: AbortSignal): Promise<ForYou> {
    const d = await get('/api/foryou', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            signals: body.signals.map(s => ({ item_id: s.item_id, kind: s.kind, value: s.value, ts: s.ts })),
            searches: body.searches,
            explore: body.explore,
            k: body.k ?? 24,
        }),
        signal,
    });
    return {
        results: (d.results ?? []).map(toMovie), unrated: (d.unrated ?? []).map(toMovie),
        profile: d.profile, policy: d.policy ?? {},
    };
}

export interface Discovery {
    query: string;
    understood: {
        /** The description as read: anchor, mood, runtime, year, language, genre or avoid, with a label. */
        chips: { kind: string; label: string }[];
        anchors: { item_id: number; title: string }[];
        mood: string | null;
        avoid: string | null;
    };
    results: Movie[];
    total: number;
    notes: string[];
}

/** Films that fit a description: "something uplifting under two hours". */
export async function fetchDiscover(q: string, limit = 36, signal?: AbortSignal): Promise<Discovery> {
    const d = await get(`/api/discover?q=${encodeURIComponent(q)}&limit=${limit}`, { signal });
    return { ...d, results: (d.results ?? []).map(toMovie), notes: d.notes ?? [] };
}

/** The offline comparison of recommenders, written by pipelines/evaluate.py. */
export const fetchEvaluation = (signal?: AbortSignal) => get('/api/evaluation', { signal });
