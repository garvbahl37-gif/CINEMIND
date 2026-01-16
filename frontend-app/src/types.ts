export interface Movie {
    tmdbId: number;
    title: string;
    poster: string;
    backdrop?: string;
    overview: string;
    vote_average?: number;
    releaseDate?: string;
    genres?: string[];
    runtime?: number;
    media_type?: string;
    poster_path?: string;
    belongs_to_collection?: {
        id: number;
        name: string;
        poster_path: string;
        backdrop_path: string;
    };
}

export interface User {
    id: string;
    name: string;
    avatar?: string;
}
