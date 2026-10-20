import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { X } from 'lucide-react';
import { Movie } from '../types';
import { Category } from '../lib/search';
import { Blend } from '../lib/blend';
import { img } from '../lib/api';
import { short } from '../lib/format';
import { cn } from '../lib/utils';
import { Mark } from './Marks';
import SearchCommand from './SearchCommand';

export type View = 'home' | 'results' | 'top50' | 'tv' | 'about';

const NAV: { id: View; label: string }[] = [
    { id: 'home', label: 'Browse' },
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
    blend: Blend;
    onUseBlend: () => void;
}

/** The blend tray: who is in it, and a way out for each. */
const BlendTray = ({ blend, onMovie, onUseBlend }: { blend: Blend; onMovie: (m: Movie) => void; onUseBlend: () => void }) => {
    const [open, setOpen] = useState(false);
    const host = useRef<HTMLDivElement>(null);
    const n = blend.films.length;

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
                aria-label={`Your blend, ${n} ${n === 1 ? 'film' : 'films'}`}
                className="flex items-center gap-2 h-11 pl-1.5 pr-4 rounded-full bg-white/[0.13] hover:bg-white/[0.2] transition-colors">
                <span className="flex -space-x-2.5">
                    {n === 0 && <span className="w-8 h-8 rounded-full border border-dashed border-white/50" />}
                    {blend.films.slice(-3).map(m => (
                        <span key={m.item_id} className="w-8 h-8 rounded-full overflow-hidden ring-2 ring-ultra bg-panel-hi">
                            {m.posterPath && <img src={img(m.posterPath, 'w185')} alt="" className="w-full h-full object-cover" />}
                        </span>
                    ))}
                </span>
                <span className="text-[12px] font-medium uppercase">
                    Blend{n > 0 && <span className="ml-1.5 tabular-nums">{n}</span>}
                </span>
            </button>

            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.14 }}
                        className="absolute right-0 top-full mt-2 z-50 w-[min(22rem,calc(100vw-2rem))] rounded-[20px] bg-black border border-white/12 p-4 shadow-[0_24px_60px_-12px_rgba(0,0,0,.9)]">
                        <p className="text-[15px] font-medium">Your blend</p>
                        <p className="mt-1 text-[13px] leading-snug text-white/60">
                            Films you add are recommended from together, as one taste. Up to twelve, kept in this browser.
                        </p>
                        {n === 0 ? (
                            <p className="mt-4 text-[13px] text-white/80">
                                Nothing in it yet. Use “Add to blend” on any film.
                            </p>
                        ) : (
                            <>
                                <ul className="mt-3 max-h-[16rem] overflow-y-auto -mx-1">
                                    {blend.films.map(m => (
                                        <li key={m.item_id} className="flex items-center gap-3 px-1 py-1.5">
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
                                            <button type="button" onClick={() => blend.toggle(m)} aria-label={`Remove ${m.title} from your blend`}
                                                className="round-btn w-8 h-8 border-white/20"><X className="w-3.5 h-3.5" /></button>
                                        </li>
                                    ))}
                                </ul>
                                <div className="mt-3 flex gap-2">
                                    <button type="button" className="pill pill-ultra" onClick={() => { setOpen(false); onUseBlend(); }}>
                                        Recommend from these
                                    </button>
                                    <button type="button" className="pill pill-ghost" onClick={blend.clear}>Clear</button>
                                </div>
                            </>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

/**
 * The frame's top bar, on the ultramarine: wordmark, sections, search and
 * your blend. On a phone the search drops to its own line and the sections
 * scroll sideways.
 */
const TopBar = ({ view, onNavigate, query, onQuery, onSearch, onCategory, onMovie, blend, onUseBlend }: Props) => (
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
            onSearch={onSearch} onCategory={onCategory} onMovie={onMovie} />

        <div className="col-start-3 row-start-1 lg:col-start-4 flex justify-end">
            <BlendTray blend={blend} onMovie={onMovie} onUseBlend={onUseBlend} />
        </div>
    </header>
    </div>
);

export default TopBar;
