export interface Film {
  item_id: number;
  title: string;
  year: number | null;
  genres: string[];
  tags: string[];
  tmdbId: number | null;
  movieId: number;
  poster_path: string | null;
  backdrop_path: string | null;
  overview: string;
  runtime: number | null;
  media_type: string;
  original_language: string | null;
  rating_avg: number | null;
  rating_count: number;
  tmdb_vote: number | null;
  /** Relative match strength, 0–1, only present on recommendation results. */
  score?: number;
  shared_genres?: string[];
  shared_tags?: string[];
}

export interface Shelf { genre: string; items: Film[] }

export interface BrowsePayload {
  hero: Film[];
  top50: Film[];
  tv: Film[];
  rows: Shelf[];
  latency_ms: number;
}

export interface SimilarPayload {
  item_id: number;
  source: Film;
  similar_items: Film[];
  latency_ms: number;
}
