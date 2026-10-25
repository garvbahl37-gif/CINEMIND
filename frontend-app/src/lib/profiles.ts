import { useCallback, useEffect, useState } from 'react';

/**
 * Who is watching. There are no accounts, so a profile is a name and a storage
 * slot in this browser: each one keeps its own likes, watchlist, passes,
 * searches and feedback log under its own key, and only the active profile's
 * signals are ever sent to the API. Another browser, or a private window,
 * starts with nothing at all.
 */
export interface Profile { id: string; name: string }

interface Book { active: string; list: Profile[] }

const KEY = 'cinemind.profiles.v1';
export const MAX_PROFILES = 6;

/** Where a profile's taste lives. */
export const tasteKey = (id: string) => `cinemind.taste.v1.${id}`;

const newId = () => {
    try { return crypto.randomUUID().slice(0, 8); } catch { return Math.random().toString(36).slice(2, 10); }
};

function load(): Book {
    try {
        const b = JSON.parse(localStorage.getItem(KEY) ?? 'null');
        const list: Profile[] = Array.isArray(b?.list)
            ? b.list.filter((p: any) => typeof p?.id === 'string' && /^[\w-]{1,32}$/.test(p.id) && typeof p?.name === 'string')
            : [];
        if (list.length) return { active: list.some(p => p.id === b.active) ? b.active : list[0].id, list };
    } catch { /* storage off or corrupt */ }
    return { active: 'main', list: [{ id: 'main', name: 'You' }] };
}

function save(b: Book) {
    try { localStorage.setItem(KEY, JSON.stringify(b)); } catch { /* storage off */ }
}

export function useProfiles() {
    const [book, setBook] = useState<Book>(load);

    useEffect(() => { save(book); }, [book]);

    // Another tab added, renamed or switched profiles.
    useEffect(() => {
        const on = (e: StorageEvent) => { if (e.key === KEY) setBook(load()); };
        window.addEventListener('storage', on);
        return () => window.removeEventListener('storage', on);
    }, []);

    const active = book.list.find(p => p.id === book.active) ?? book.list[0];

    const select = useCallback((id: string) =>
        setBook(b => b.list.some(p => p.id === id) ? { ...b, active: id } : b), []);

    /** Add a profile and switch to it. Its taste starts empty. */
    const add = useCallback((name: string) => {
        const n = name.trim().slice(0, 24);
        if (!n) return;
        setBook(b => {
            if (b.list.length >= MAX_PROFILES) return b;
            const p = { id: newId(), name: n };
            return { active: p.id, list: [...b.list, p] };
        });
    }, []);

    /** Delete a profile and everything it stored. The last one can't go. */
    const remove = useCallback((id: string) => {
        setBook(b => {
            if (b.list.length <= 1) return b;
            try { localStorage.removeItem(tasteKey(id)); } catch { /* storage off */ }
            const list = b.list.filter(p => p.id !== id);
            return { active: b.active === id ? list[0].id : b.active, list };
        });
    }, []);

    return { list: book.list, active, select, add, remove };
}

export type Profiles = ReturnType<typeof useProfiles>;
