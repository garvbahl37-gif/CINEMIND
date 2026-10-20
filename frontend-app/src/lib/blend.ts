import { useCallback, useEffect, useState } from 'react';
import { Movie } from '../types';

/**
 * Your blend: a handful of films the dashboard recommends from at once,
 * through /api/blend. It lives in this browser only — there are no accounts —
 * and storage can be unavailable (private windows), so every access is guarded
 * and the blend simply starts empty when it is.
 */
const KEY = 'cinemind.blend.v1';
/** The API reads at most twelve seeds. */
const MAX = 12;

const slim = (m: Movie): Movie => ({
    item_id: m.item_id, title: m.title, year: m.year, posterPath: m.posterPath,
});

const load = (): Movie[] => {
    try {
        const v = JSON.parse(localStorage.getItem(KEY) ?? '[]');
        return Array.isArray(v) ? v.filter(m => typeof m?.item_id === 'number') : [];
    } catch {
        return [];
    }
};

export function useBlend() {
    const [films, setFilms] = useState<Movie[]>(load);

    useEffect(() => {
        try { localStorage.setItem(KEY, JSON.stringify(films)); } catch { /* storage off */ }
    }, [films]);

    const has = useCallback((m?: Movie | null) => !!m && films.some(f => f.item_id === m.item_id), [films]);

    const toggle = useCallback((m: Movie) => {
        if (m.item_id == null) return;
        setFilms(fs => fs.some(f => f.item_id === m.item_id)
            ? fs.filter(f => f.item_id !== m.item_id)
            : [...fs, slim(m)].slice(-MAX));
    }, []);

    const clear = useCallback(() => setFilms([]), []);

    return { films, has, toggle, clear };
}

export type Blend = ReturnType<typeof useBlend>;
