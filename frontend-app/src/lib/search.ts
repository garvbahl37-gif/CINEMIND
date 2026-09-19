/**
 * One place that knows how to talk to the search API.
 *
 * The query box and the filter bar are two ways of expressing the same
 * request, so they build the same URL through here rather than each assembling
 * their own — which is how the two ends of a faceted search usually drift apart.
 */
import { API_BASE, TMDB_IMAGE_BASE } from '../config';
import { Movie } from '../types';

export type Sort = 'relevance' | 'rating' | 'newest' | 'oldest' | 'title';

export interface Filters {
    genres: string[];
    /** Inclusive decade bounds, e.g. 1990–1999. */
    yearMin?: number;
    yearMax?: number;
    lang?: string;
    sort: Sort;
}

export interface Facet { value: string | number; label?: string; count: number }
export interface Facets { genres: Facet[]; decades: Facet[]; languages: Facet[] }

export interface Category {
    kind: 'genre' | 'language' | 'filter' | 'person' | 'tag';
    label: string;
    sublabel: string;
    /** What selecting it applies: filter fields, or `q` for free text. */
    filters: { genres?: string[]; lang?: string; year_min?: number; year_max?: number; q?: string };
}

export const EMPTY: Filters = { genres: [], sort: 'relevance' };

export const SORTS: { value: Sort; label: string }[] = [
    { value: 'relevance', label: 'Best match' },
    { value: 'rating', label: 'Top rated' },
    { value: 'newest', label: 'Newest' },
    { value: 'oldest', label: 'Oldest' },
    { value: 'title', label: 'A–Z' },
];

export const isActive = (f: Filters) =>
    f.genres.length > 0 || !!f.lang || !!f.yearMin || f.sort !== 'relevance';

/** Turn a suggestion's payload into a full filter set. */
export const fromCategory = (c: Category, base: Filters = EMPTY): Filters => ({
    ...base,
    genres: c.filters.genres ?? [],
    lang: c.filters.lang,
    yearMin: c.filters.year_min,
    yearMax: c.filters.year_max,
});

const poster = (m: any): string | null => {
    const p = m.poster ?? m.poster_path;
    if (!p) return null;
    return p.startsWith('http') ? p : `${TMDB_IMAGE_BASE}/w500${p}`;
};

export interface SearchResponse {
    results: Movie[];
    /** What the server applied — the query's own structure when none was sent. */
    filters: Filters;
    facets: Facets | null;
    total: number;
}

/**
 * @param filters `null` lets the query speak for itself ("korean thrillers"
 *        becomes ko + Thriller server-side); an object overrides it entirely.
 */
export async function searchFilms(
    q: string,
    filters: Filters | null,
    opts: { limit?: number; facets?: boolean; signal?: AbortSignal } = {},
): Promise<SearchResponse> {
    const p = new URLSearchParams();
    if (q) p.set('q', q);
    p.set('limit', String(opts.limit ?? 60));
    if (opts.facets !== false) p.set('facets', 'true');
    if (filters) {
        // An empty string is meaningful: it means "no genres", as distinct from
        // "I didn't say", which is what leaving the parameter out means.
        p.set('genres', filters.genres.join(','));
        if (filters.yearMin) p.set('year_min', String(filters.yearMin));
        if (filters.yearMax) p.set('year_max', String(filters.yearMax));
        if (filters.lang) p.set('lang', filters.lang);
        if (filters.sort !== 'relevance') p.set('sort', filters.sort);
    }

    const res = await fetch(`${API_BASE}/api/movies/search?${p}`, { signal: opts.signal });
    if (!res.ok) throw new Error(`Search failed: ${res.status}`);
    const d = await res.json();

    return {
        results: (d.results ?? []).map((m: any) => ({ ...m, poster: poster(m), id: m.tmdbId ?? m.item_id })),
        filters: {
            genres: d.filters?.genres ?? [],
            yearMin: d.filters?.year_min,
            yearMax: d.filters?.year_max,
            lang: d.filters?.lang,
            sort: filters?.sort ?? 'relevance',
        },
        facets: d.facets ?? null,
        total: d.total ?? d.count ?? 0,
    };
}

export async function fetchCategories(): Promise<Facets> {
    const res = await fetch(`${API_BASE}/api/categories`);
    if (!res.ok) throw new Error(`Categories failed: ${res.status}`);
    return res.json();
}

export async function fetchSuggestions(q: string, signal?: AbortSignal) {
    const res = await fetch(`${API_BASE}/api/suggest?q=${encodeURIComponent(q)}`, { signal });
    if (!res.ok) throw new Error(`Suggest failed: ${res.status}`);
    const d = await res.json();
    return {
        categories: (d.categories ?? []) as Category[],
        titles: (d.results ?? []).map((m: any) => ({ ...m, poster: poster(m), id: m.tmdbId })) as Movie[],
    };
}
