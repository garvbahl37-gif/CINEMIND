import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Plus, X } from 'lucide-react';
import { Movie } from '../types';
import { Category } from '../lib/search';
import { Taste } from '../lib/taste';
import { MAX_PROFILES, Profiles } from '../lib/profiles';
import { img } from '../lib/api';
import { short } from '../lib/format';
import { cn } from '../lib/utils';
import { Mark } from './Marks';
import SearchCommand from './SearchCommand';

export type View = 'home' | 'discover' | 'results' | 'top50' | 'tv' | 'about';

const NAV: { id: View; label: string }[] = [
    { id: 'home', label: 'Browse' },
    { id: 'discover', label: 'Discover' },
    { id: 'top50', label: 'Top 50' },
    { id: 'tv', label: 'Series' },
    { id: 'about', label: 'How it works' },
];

interface Props {
    view: View;
    onNavigate: (v: View) => void;
    query: string;
    onQuery: (q: string) => void;
    onSearch: (q: string) => void;
    onCategory: (c: Category) => void;
    onMovie: (m: Movie) => void;
    onDiscover: (q: string) => void;
    taste: Taste;
    profiles: Profiles;
    onForYou: () => void;
}

type Pile = 'like' | 'watchlist' | 'skip';
const SHELVES: { id: Pile; label: string; empty: string }[] = [
    { id: 'like', label: 'Liked', empty: 'Nothing liked yet. Press Like on any film.' },
    { id: 'watchlist', label: 'Watchlist', empty: 'Your watchlist is empty. Use the bookmark on any film.' },
    { id: 'skip', label: 'Not for me', empty: 'Nothing passed on. The thumbs-down keeps a film out of your picks.' },
];

/**
 * Your taste, in a tray: what you liked, saved and passed on, how adventurous
 * the picks should be, and how the picks have landed so far.
 */
/** "Your taste" for the default profile, "Sam's taste" for Sam's. */
const whose = (name: string) => (name.trim().toLowerCase() === 'you' ? 'Your' : `${name}’s`);

/**
 * Who is watching: one pill per profile, a way to add one, and — for the
 * profile in use — a way to delete it and everything it stored.
 */
const Who = ({ profiles }: { profiles: Profiles }) => {
    const [adding, setAdding] = useState(false);
    const [name, setName] = useState('');
    const [confirm, setConfirm] = useState(false);
    const { list, active } = profiles;
    const add = () => { if (name.trim()) { profiles.add(name); setName(''); setAdding(false); } };

    return (
        <div className="mt-3">
            <div className="flex flex-wrap items-center gap-1.5" role="radiogroup" aria-label="Who's watching">
                {list.map(p => (
                    <button key={p.id} type="button" role="radio" aria-checked={p.id === active.id}
                        onClick={() => { profiles.select(p.id); setConfirm(false); }}
                        className={cn('pill h-8 pl-1 pr-3 normal-case text-[12px] tracking-normal gap-2',
                            p.id === active.id ? 'bg-white text-black' : 'bg-white/[0.08] text-white hover:bg-white/[0.14]')}>
                        <span aria-hidden="true" className={cn('grid place-items-center w-6 h-6 rounded-full text-[11px] font-semibold',
                            p.id === active.id ? 'bg-ultra text-white' : 'bg-white/15')}>{p.name.charAt(0).toUpperCase()}</span>
                        <span className="max-w-[9rem] truncate">{p.name}</span>
                    </button>
                ))}
                {list.length < MAX_PROFILES && !adding && (
                    <button type="button" onClick={() => setAdding(true)} aria-label="Add a profile"
                        className="round-btn w-8 h-8 border-white/20"><Plus className="w-3.5 h-3.5" /></button>
                )}
            </div>
            {adding && (
                <form className="mt-2 flex gap-1.5" onSubmit={(e) => { e.preventDefault(); add(); }}>
                    <input autoFocus value={name} onChange={e => setName(e.target.value)} maxLength={24}
                        onKeyDown={e => { if (e.key === 'Escape') { e.stopPropagation(); setAdding(false); } }}
                        aria-label="Profile name" placeholder="Name"
                        className="flex-1 min-w-0 h-9 px-3.5 rounded-full bg-white/[0.1] text-[13px] outline-none focus-visible:outline-2 focus-visible:outline-white placeholder:text-white/45" />
                    <button type="submit" className="pill pill-white" disabled={!name.trim()}>Add</button>
                    <button type="button" className="pill pill-ghost" onClick={() => setAdding(false)}>Cancel</button>
                </form>
            )}
            {list.length > 1 && (
                confirm ? (
                    <p className="mt-2 flex flex-wrap items-center gap-2 text-[12px] text-white/70">
                        Delete this profile and everything saved in it?
                        <button type="button" className="underline underline-offset-2 text-white" onClick={() => { profiles.remove(active.id); setConfirm(false); }}>Delete</button>
                        <button type="button" className="underline underline-offset-2" onClick={() => setConfirm(false)}>Keep</button>
                    </p>
                ) : (
                    <button type="button" className="mt-2 text-[12px] text-white/50 hover:text-white/80 underline underline-offset-2"
                        onClick={() => setConfirm(true)}>Delete this profile</button>
                )
            )}
        </div>
    );
};

const TasteTray = ({ taste, profiles, onMovie, onForYou }: {
    taste: Taste; profiles: Profiles; onMovie: (m: Movie) => void; onForYou: () => void;
}) => {
    const [open, setOpen] = useState(false);
    const [tab, setTab] = useState<Pile>('like');
    const many = profiles.list.length > 1;
    const host = useRef<HTMLDivElement>(null);
    const n = taste.liked.length;
    const lists = { like: taste.liked, watchlist: taste.watchlist, skip: taste.skipped };
    const films = [...lists[tab]].reverse();
    const s = taste.stats;

    useEffect(() => {
        if (!open) return;
        const away = (e: MouseEvent) => { if (!host.current?.contains(e.target as Node)) setOpen(false); };
        const esc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
        document.addEventListener('mousedown', away);
        document.addEventListener('keydown', esc);
        return () => { document.removeEventListener('mousedown', away); document.removeEventListener('keydown', esc); };
    }, [open]);

    return (
        <div ref={host} className="relative shrink-0">
            <button type="button" onClick={() => setOpen(o => !o)} aria-expanded={open}
                aria-label={`${whose(profiles.active.name)} taste, ${n} liked ${n === 1 ? 'film' : 'films'}`}
                className="flex items-center gap-2 h-11 pl-1.5 pr-4 rounded-full bg-white/[0.13] hover:bg-white/[0.2] transition-colors">
                {many && (
                    <span aria-hidden="true" className="grid place-items-center w-8 h-8 rounded-full bg-white text-ultra text-[13px] font-semibold">
                        {profiles.active.name.charAt(0).toUpperCase()}
                    </span>
                )}
                <span className={cn('flex -space-x-2.5', many && n === 0 && 'hidden')}>
                    {n === 0 && <span className="w-8 h-8 rounded-full border border-dashed border-white/50" />}
                    {taste.liked.slice(-3).map(m => (
                        <span key={m.item_id} className="w-8 h-8 rounded-full overflow-hidden ring-2 ring-ultra bg-panel-hi">
                            {m.posterPath && <img src={img(m.posterPath, 'w185')} alt="" className="w-full h-full object-cover" />}
                        </span>
                    ))}
                </span>
                <span className="text-[12px] font-medium uppercase">
                    Taste{n > 0 && <span className="ml-1.5 tabular-nums">{n}</span>}
                </span>
            </button>

            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.14 }}
                        className="absolute right-0 top-full mt-2 z-50 w-[min(23rem,calc(100vw-2rem))] rounded-[20px] bg-black border border-white/12 p-4 shadow-[0_24px_60px_-12px_rgba(0,0,0,.9)]">
                        <p className="text-[15px] font-medium">{whose(profiles.active.name)} taste</p>
                        <p className="mt-1 text-[13px] leading-snug text-white/60">
                            What you like, save and pass on shapes the For you shelf. Each profile keeps its own, in
                            this browser only; no name or account is ever sent.
                        </p>
                        <Who profiles={profiles} />

                        <div role="tablist" aria-label="Your films" className="mt-3 flex gap-1.5">
                            {SHELVES.map(t => (
                                <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
                                    className={cn('pill h-8 px-3 normal-case text-[12px] tracking-normal',
                                        tab === t.id ? 'bg-white text-black' : 'bg-white/[0.08] text-white hover:bg-white/[0.14]')}>
                                    {t.label} <span className="tabular-nums opacity-60">{lists[t.id].length}</span>
                                </button>
                            ))}
                        </div>

                        {films.length === 0 ? (
                            <p className="mt-4 mb-1 text-[13px] text-white/75">{SHELVES.find(t => t.id === tab)!.empty}</p>
                        ) : (
                            <ul role="tabpanel" className="mt-2 max-h-[14rem] overflow-y-auto -mx-1">
                                {films.map(sig => {
                                    const m: Movie = { item_id: sig.item_id, title: sig.title, year: sig.year, posterPath: sig.posterPath };
                                    return (
                                        <li key={sig.item_id} className="flex items-center gap-3 px-1 py-1.5">
                                            <button type="button" onClick={() => { setOpen(false); onMovie(m); }}
                                                className="flex items-center gap-3 flex-1 min-w-0 text-left rounded-lg hover:bg-white/[0.06]">
                                                <span className="w-8 h-12 rounded-[6px] overflow-hidden bg-panel-hi shrink-0">
                                                    {m.posterPath && <img src={img(m.posterPath, 'w185')} alt="" className="w-full h-full object-cover" />}
                                                </span>
                                                <span className="min-w-0">
                                                    <span className="block text-[14px] truncate">{short(m.title)}</span>
                                                    <span className="block text-[12px] text-white/50">{m.year}</span>
                                                </span>
                                            </button>
                                            <button type="button" onClick={() => taste.toggle(tab, m)}
                                                aria-label={`Remove ${short(m.title)} from ${SHELVES.find(t => t.id === tab)!.label.toLowerCase()}`}
                                                className="round-btn w-8 h-8 border-white/20"><X className="w-3.5 h-3.5" /></button>
                                        </li>
                                    );
                                })}
                            </ul>
                        )}

                        <label className="mt-4 block">
                            <span className="flex justify-between text-[13px]">
                                <span>Adventurousness</span>
                                <span className="text-white/60 tabular-nums">{Math.round(taste.explore * 100)}%</span>
                            </span>
                            <input type="range" min={0} max={1} step={0.05} value={taste.explore}
                                onChange={e => taste.setExplore(Number(e.target.value))}
                                aria-valuetext={`${Math.round(taste.explore * 100)}% adventurous`}
                                className="mt-2 w-full accent-violet" />
                            <span className="flex justify-between text-[11px] text-white/45">
                                <span>Close to what you like</span><span>Further afield</span>
                            </span>
                        </label>

                        {s.shown > 0 && (
                            <p className="mt-3 text-[12px] leading-snug text-white/55">
                                Of {s.shown} films picked for you, you opened {s.opened} and liked {s.liked}.
                            </p>
                        )}

                        <div className="mt-4 flex gap-2">
                            <button type="button" className="pill pill-ultra" onClick={() => { setOpen(false); onForYou(); }}>
                                See your picks
                            </button>
                            {taste.signals.length > 0 && (
                                <button type="button" className="pill pill-ghost" onClick={taste.clear}>Start over</button>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

/**
 * The frame's top bar, on the ultramarine: wordmark, sections, search and
 * your taste. On a phone the search drops to its own line and the sections
 * scroll sideways.
 */
const TopBar = ({ view, onNavigate, query, onQuery, onSearch, onCategory, onMovie, onDiscover, taste, profiles, onForYou }: Props) => (
    // A black band behind the bar, so nothing scrolls into view around its rounded edges.
    <div className="relative md:sticky md:top-4 z-40 bg-ink pb-1.5 md:pb-2.5 md:-mt-2.5 md:pt-2.5 md:-mx-2.5 md:px-2.5">
    <header className="relative rounded-[22px] bg-ultra px-3 py-3 md:px-4
                       grid grid-cols-[auto_1fr_auto] lg:grid-cols-[auto_auto_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-2.5">
        <button type="button" onClick={() => onNavigate('home')} aria-label="CINEMIND, back to browse"
            className="flex items-center gap-2 h-11 pl-1.5 pr-2 rounded-full">
            <Mark className="w-7 h-7 text-white" />
            <span className="text-[15px] font-semibold tracking-[-0.01em]">CINEMIND</span>
        </button>

        <nav aria-label="Sections" className="col-span-3 row-start-3 md:row-start-2 lg:col-span-1 lg:row-start-1 lg:col-start-2 scroller gap-1.5">
            {NAV.map(n => (
                <button key={n.id} type="button" onClick={() => onNavigate(n.id)}
                    aria-current={view === n.id ? 'page' : undefined}
                    className={cn('pill h-10', view === n.id ? 'bg-white text-ultra' : 'bg-white/[0.13] text-white hover:bg-white/[0.2]')}>
                    {n.label}
                </button>
            ))}
        </nav>

        <SearchCommand className="col-span-3 row-start-2 md:col-span-1 md:row-start-1 md:col-start-2 lg:col-start-3 min-w-0" value={query} onChange={onQuery}
            onSearch={onSearch} onCategory={onCategory} onMovie={onMovie} onDiscover={onDiscover} />

        <div className="col-start-3 row-start-1 lg:col-start-4 flex justify-end">
            <TasteTray taste={taste} profiles={profiles} onMovie={onMovie} onForYou={onForYou} />
        </div>
    </header>
    </div>
);

export default TopBar;
