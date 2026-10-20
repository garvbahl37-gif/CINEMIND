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
}
