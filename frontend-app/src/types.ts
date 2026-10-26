/** A film as every screen reads it, whichever endpoint it came from. */
export interface Movie {
    /** The model's index for this film — what /api/similar is keyed by. */
    item_id?: number;
    tmdbId?: number;
    title: string;
    year?: number;
    /** TMDB image paths; turn them into URLs with `img()` at the size you need. */
    posterPath?: string;
    backdropPath?: string;
    overview?: string;
    genres?: string[];
    tags?: string[];
    runtime?: number;
    media_type?: string;
    original_language?: string;
    /** MovieLens average, out of 5. */
    rating?: number;
    rating_count?: number;
    directors?: string[];
    cast?: string[];

    /** Set on neighbours: match strength 0–1, top match = 1. */
    score?: number;
    shared_genres?: string[];
    shared_tags?: string[];
    imdb_id?: string;
    /** False for metadata-only films: nobody in MovieLens has rated them. */
    rating_backed?: boolean;
    /** How a film's matches were found: co-ratings, a mix, or content alone. */
    match_basis?: MatchBasis;

    /** Set on discovery results: why the film fits the description. */
    why?: string[];
    /** Set on personal recommendations by the adaptive engine. */
    reason?: string;
    evidence?: Evidence[];
    confidence?: number;
    confidence_label?: 'High' | 'Medium' | 'Low';
    /** An exploration slot, and the probability it was shown with. */
    exploring?: boolean;
    propensity?: number;
}

export type MatchBasis = 'collaborative' | 'hybrid' | 'content';

/** One piece of evidence behind a recommendation. */
export interface Evidence {
    kind: 'liked' | 'search' | 'drift' | 'explore' | 'quality';
    text: string;
    strength: number;
    item_id?: number;
    title?: string;
    query?: string;
}
