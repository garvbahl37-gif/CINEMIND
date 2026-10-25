import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Movie } from '../types';
import { tasteKey } from './profiles';

/**
 * Your taste: everything the adaptive engine learns from, kept in this browser.
 *
 * There are no accounts, so each profile's taste lives in localStorage under
 * its own key (see profiles.ts) and travels with each /api/foryou request; the
 * server keeps nothing and is never told who is asking. Storage can be missing
 * (private windows, blocked site data), so every access is guarded and the
 * profile simply starts empty when it is.
 *
 * Alongside the signals sits a feedback log: which recommendations were shown,
 * where, whether each was an exploration slot and with what probability it was
 * shown — and what happened next. That is what counterfactual evaluation needs,
 * so it is recorded from the start.
 */
export type SignalKind = 'like' | 'skip' | 'watchlist' | 'view' | 'rating';

export interface TasteSignal {
    item_id: number;
    kind: SignalKind;
    value?: number;
    ts: number;
    title: string;
    year?: number;
    posterPath?: string;
}

export interface LogEvent {
    ts: number;
    type: 'shown' | 'opened' | 'like' | 'skip' | 'watchlist';
    item_id: number;
    position?: number;
    exploring?: boolean;
    propensity?: number;
    policy?: string;
}

interface State {
    signals: TasteSignal[];
    searches: { q: string; ts: number }[];
    explore: number;
    log: LogEvent[];
}

const OLD_BLEND = 'cinemind.blend.v1';
const MAX_SIGNALS = 300, MAX_SEARCHES = 30, MAX_LOG = 600;
const EMPTY: State = { signals: [], searches: [], explore: 0.3, log: [] };

function parse(raw: string | null): State | null {
    if (!raw) return null;
    try {
        const s = JSON.parse(raw);
        return {
            signals: Array.isArray(s.signals) ? s.signals.filter((x: any) => typeof x?.item_id === 'number') : [],
            searches: Array.isArray(s.searches) ? s.searches.filter((x: any) => typeof x?.q === 'string') : [],
            explore: typeof s.explore === 'number' && Number.isFinite(s.explore) ? Math.min(1, Math.max(0, s.explore)) : 0.3,
            log: Array.isArray(s.log) ? s.log : [],
        };
    } catch { return null; }
}

function load(id: string): State {
    try {
        const saved = parse(localStorage.getItem(tasteKey(id)));
        if (saved) return saved;
        // The blend this replaced: its films become the first profile's likes.
        if (id !== 'main') return EMPTY;
        const old = JSON.parse(localStorage.getItem(OLD_BLEND) ?? '[]');
        if (Array.isArray(old) && old.length) {
            const now = Date.now();
            return {
                ...EMPTY,
                signals: old.filter((m: any) => typeof m?.item_id === 'number').map((m: any, i: number) => ({
                    item_id: m.item_id, kind: 'like' as const, ts: now - i * 1000,
                    title: m.title, year: m.year, posterPath: m.posterPath,
                })),
            };
        }
    } catch { /* storage off or corrupt: start empty */ }
    return EMPTY;
}

const slim = (m: Movie) => ({ item_id: m.item_id!, title: m.title, year: m.year, posterPath: m.posterPath });

export function useTaste(profile: string) {
    // The state carries the profile it belongs to, so a save can never land in
    // another profile's slot while the switch is under way.
    const [box, setBox] = useState(() => ({ id: profile, s: load(profile) }));
    if (box.id !== profile) setBox({ id: profile, s: load(profile) });
    const state = box.s;
    const setState = useCallback((f: (s: State) => State) => setBox(b => ({ ...b, s: f(b.s) })), []);
    const written = useRef('');

    useEffect(() => {
        const raw = JSON.stringify(box.s);
        if (raw === written.current) return;
        written.current = raw;
        try {
            localStorage.setItem(tasteKey(box.id), raw);
            if (box.id === 'main') localStorage.removeItem(OLD_BLEND);
        } catch { /* storage off */ }
    }, [box]);

    // The same profile open in another tab: take its latest state instead of overwriting it.
    useEffect(() => {
        const on = (e: StorageEvent) => {
            if (e.key !== tasteKey(box.id)) return;
            const next = parse(e.newValue) ?? EMPTY;
            written.current = e.newValue ?? '';
            setBox(b => b.id === box.id ? { ...b, s: next } : b);
        };
        window.addEventListener('storage', on);
        return () => window.removeEventListener('storage', on);
    }, [box.id]);

    /** The latest explicit stance on a film — like, skip, watchlist or rating — if any. */
    const stance = useCallback((m?: Movie | null): TasteSignal | undefined => {
        if (!m || m.item_id == null) return undefined;
        for (let i = state.signals.length - 1; i >= 0; i--) {
            const s = state.signals[i];
            if (s.item_id === m.item_id && s.kind !== 'view') return s;
        }
        return undefined;
    }, [state.signals]);

    const has = useCallback((kind: SignalKind, m?: Movie | null) => stance(m)?.kind === kind, [stance]);

    /** Like, skip or watchlist a film — or undo it if that's already the stance. */
    const toggle = useCallback((kind: 'like' | 'skip' | 'watchlist', m: Movie) => {
        if (m.item_id == null) return;
        setState(s => {
            const current = [...s.signals].reverse().find(x => x.item_id === m.item_id && x.kind !== 'view');
            const rest = s.signals.filter(x => !(x.item_id === m.item_id && x.kind !== 'view'));
            const signals = current?.kind === kind ? rest
                : [...rest, { ...slim(m), kind, ts: Date.now() }].slice(-MAX_SIGNALS);
            const log = current?.kind === kind ? s.log
                : [...s.log, { ts: Date.now(), type: kind, item_id: m.item_id! }].slice(-MAX_LOG);
            return { ...s, signals, log };
        });
    }, []);

    /** A film was opened: a weak signal, recorded at most once every half hour. */
    const view = useCallback((m: Movie) => {
        if (m.item_id == null) return;
        setState(s => {
            const recent = s.signals.some(x => x.item_id === m.item_id && Date.now() - x.ts < 30 * 60_000);
            if (recent) return s;
            const views = s.signals.filter(x => x.kind === 'view');
            const others = s.signals.filter(x => x.kind !== 'view');
            const keptViews = [...views, { ...slim(m), kind: 'view' as const, ts: Date.now() }].slice(-100);
            return { ...s, signals: [...others, ...keptViews].sort((a, b) => a.ts - b.ts).slice(-MAX_SIGNALS) };
        });
    }, []);

    const search = useCallback((q: string) => {
        const t = q.trim();
        if (t.length < 3) return;
        setState(s => ({
            ...s,
            searches: [...s.searches.filter(x => x.q.toLowerCase() !== t.toLowerCase()), { q: t, ts: Date.now() }]
                .slice(-MAX_SEARCHES),
        }));
    }, []);

    const setExplore = useCallback((v: number) => setState(s => ({ ...s, explore: Math.min(1, Math.max(0, v)) })), []);

    const remove = useCallback((item_id: number) =>
        setState(s => ({ ...s, signals: s.signals.filter(x => x.item_id !== item_id) })), []);

    const clear = useCallback(() => setState(s => ({ ...EMPTY, explore: s.explore })), []);

    /** Record what was shown — with each slot's exploration flag and propensity. */
    const logShown = useCallback((films: Movie[], policy: string) => setState(s => ({
        ...s,
        log: [...s.log, ...films.map((m, i) => ({
            ts: Date.now(), type: 'shown' as const, item_id: m.item_id!, position: i,
            exploring: !!m.exploring, propensity: m.propensity ?? 1, policy,
        }))].slice(-MAX_LOG),
    })), []);

    const logOpened = useCallback((m: Movie) => {
        if (m.item_id == null) return;
        setState(s => ({ ...s, log: [...s.log, { ts: Date.now(), type: 'opened' as const, item_id: m.item_id! }].slice(-MAX_LOG) }));
    }, []);

    const { liked, watchlist, skipped } = useMemo(() => {
        const of = (kind: SignalKind) => state.signals.filter(x => x.kind === kind);
        return { liked: of('like'), watchlist: of('watchlist'), skipped: of('skip') };
    }, [state.signals]);

    /** The feedback loop so far: of the recommendations shown, how many were opened or liked. */
    const stats = useMemo(() => {
        const shown = new Set(state.log.filter(e => e.type === 'shown').map(e => e.item_id));
        const opened = new Set(state.log.filter(e => e.type === 'opened' && shown.has(e.item_id)).map(e => e.item_id));
        const likedFromRecs = new Set(state.log.filter(e => e.type === 'like' && shown.has(e.item_id)).map(e => e.item_id));
        const explored = state.log.filter(e => e.type === 'shown' && e.exploring).length;
        return { shown: shown.size, opened: opened.size, liked: likedFromRecs.size, explored };
    }, [state.log]);

    /** Positive or negative signals the engine can use: anything but bare views counts. */
    const active = liked.length + watchlist.length + skipped.length > 0 || state.searches.length > 0;

    return {
        signals: state.signals, searches: state.searches, explore: state.explore, log: state.log,
        liked, watchlist, skipped, stats, active,
        stance, has, toggle, view, search, setExplore, remove, clear, logShown, logOpened,
    };
}

export type Taste = ReturnType<typeof useTaste>;
