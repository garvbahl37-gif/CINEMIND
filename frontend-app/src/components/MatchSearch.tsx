import { useEffect, useId, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowUpRight, Loader2, Search } from 'lucide-react';
import { Movie } from '../types';
import { fetchSuggestions } from '../lib/search';
import { img } from '../lib/api';
import { names, short } from '../lib/format';
import { cn } from '../lib/utils';
import { Mark } from './Marks';

interface Props {
    onPick: (m: Movie) => void;
    /** `band` is the dashboard's bar; `compact` sits in the stage header. */
    variant?: 'band' | 'compact';
}

/** Films worth trying first. Resolved up front, so a click opens at once. */
const TRY = ['Spirited Away', 'Alien', 'Amelie', 'Heat', 'Old Boy'];

const firstFilm = async (title: string, signal?: AbortSignal) =>
    (await fetchSuggestions(title, signal)).titles.find(m => m.item_id != null);

/**
 * Search for recommendations: type a film, get its 24 matches.
 *
 * Titles only. The top bar's search also answers with categories, which is
 * right for browsing but wrong here — "rom" should offer Roman Holiday, not the
 * Romance genre, when what you want is the films closest to one film.
 */
const MatchSearch = ({ onPick, variant = 'band' }: Props) => {
    const [q, setQ] = useState('');
    const [films, setFilms] = useState<Movie[]>([]);
    const [open, setOpen] = useState(false);
    const [cursor, setCursor] = useState(0);
    const [busy, setBusy] = useState(false);
    const [miss, setMiss] = useState(false);
    const [tries, setTries] = useState<(Movie | undefined)[]>([]);
    const box = useRef<HTMLDivElement>(null);
    const input = useRef<HTMLInputElement>(null);
    const listId = useId();
    const inputId = useId();
    const band = variant === 'band';

    useEffect(() => {
        const t = q.trim();
        setMiss(false);
        if (t.length < 2) { setFilms([]); setBusy(false); return; }
        const ctrl = new AbortController();
        setBusy(true);
        const timer = setTimeout(() => {
            fetchSuggestions(t, ctrl.signal)
                .then(d => {
                    const hits = d.titles.filter(m => m.item_id != null).slice(0, 6);
                    setFilms(hits); setCursor(0); setMiss(hits.length === 0);
                })
                .catch(() => { })
                .finally(() => setBusy(false));
        }, 140);
        return () => { clearTimeout(timer); ctrl.abort(); };
    }, [q]);

    useEffect(() => {
        if (!band) return;
        const ctrl = new AbortController();
        Promise.all(TRY.map(t => firstFilm(t, ctrl.signal).catch(() => undefined))).then(setTries);
        return () => ctrl.abort();
    }, [band]);

    useEffect(() => {
        const away = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
        document.addEventListener('mousedown', away);
        return () => document.removeEventListener('mousedown', away);
    }, []);

    const choose = (m?: Movie) => {
        if (!m) return;
        setOpen(false); setQ(''); setFilms([]);
        input.current?.blur();
        onPick(m);
    };

    const onKey = (e: React.KeyboardEvent) => {
        if (e.key === 'Escape') { setOpen(false); input.current?.blur(); return; }
        if (e.key === 'Enter') { e.preventDefault(); choose(films[cursor] ?? films[0]); return; }
        if (!films.length) return;
        if (e.key === 'ArrowDown') { e.preventDefault(); setOpen(true); setCursor(c => (c + 1) % films.length); }
        if (e.key === 'ArrowUp') { e.preventDefault(); setOpen(true); setCursor(c => (c - 1 + films.length) % films.length); }
    };

    const show = open && (films.length > 0 || miss);
    const target = films[cursor] ?? films[0];

    const list = (
        <AnimatePresence>
            {show && (
                <motion.ul
                    id={listId}
                    role="listbox"
                    aria-label="Films"
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.14 }}
                    className="absolute top-full mt-2 left-0 right-0 z-[70] bg-black border border-white/12 rounded-[22px] py-2
                               shadow-[0_24px_60px_-12px_rgba(0,0,0,.9)] max-h-[26rem] overflow-y-auto"
                >
                    {miss && (
                        <li className="px-5 py-3 text-[14px] text-white/70">
                            No film called “{q.trim()}”. Check the spelling, or try its original title.
                        </li>
                    )}
                    {films.map((m, i) => (
                        <li key={m.item_id} id={`${listId}-${i}`} role="option" aria-selected={cursor === i}
                            onMouseEnter={() => setCursor(i)}
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={() => choose(m)}
                            className={cn('flex items-center gap-3 px-5 py-2 cursor-pointer',
                                cursor === i ? 'bg-white/[0.08]' : 'hover:bg-white/[0.05]')}>
                            <span className="w-10 h-[60px] shrink-0 rounded-[8px] overflow-hidden bg-panel-hi">
                                {m.posterPath && <img src={img(m.posterPath, 'w185')} alt="" loading="lazy" className="w-full h-full object-cover" />}
                            </span>
                            <span className="flex-1 min-w-0">
                                <span className="block text-[15px] font-medium truncate">{short(m.title)}</span>
                                <span className="block text-[12px] text-white/55 truncate">
                                    {[m.year, names(m.directors, 1)].filter(Boolean).join(', ')}
                                </span>
                            </span>
                            {cursor === i && (
                                <span className="hidden sm:inline-flex items-center gap-1 text-[12px] text-white/70 shrink-0">
                                    Enter for its matches
                                </span>
                            )}
                        </li>
                    ))}
                </motion.ul>
            )}
        </AnimatePresence>
    );

    const combo = {
        ref: input,
        id: inputId,
        type: 'text' as const,
        role: 'combobox' as const,
        'aria-expanded': show,
        'aria-controls': listId,
        'aria-autocomplete': 'list' as const,
        'aria-activedescendant': show && films[cursor] ? `${listId}-${cursor}` : undefined,
        value: q,
        onChange: (e: React.ChangeEvent<HTMLInputElement>) => { setQ(e.target.value); setOpen(true); },
        onFocus: () => setOpen(true),
        onKeyDown: onKey,
        autoComplete: 'off',
        spellCheck: false,
    };

    if (!band) {
        return (
            <div ref={box} className="relative w-full">
                <Search aria-hidden="true" className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-white/60 pointer-events-none" />
                <input {...combo} aria-label="Switch to another film" placeholder="Switch to another film…"
                    className="w-full h-10 pl-10 pr-10 rounded-full text-[14px] text-white placeholder:text-white/50 bg-white/[0.08]
                               hover:bg-white/[0.12] outline-none focus-visible:outline-2 focus-visible:outline-white transition-colors" />
                {busy && <Loader2 aria-hidden="true" className="absolute right-3.5 top-1/2 -translate-y-1/2 w-4 h-4 animate-spin text-white/60" />}
                {list}
            </div>
        );
    }

    return (
        <section aria-label="Find matches for a film"
            className="relative z-30 rounded-[24px] bg-panel p-3 md:p-4
                       bg-[radial-gradient(ellipse_55%_140%_at_50%_150%,rgba(187,0,1,.5)_0%,rgba(187,0,1,.12)_45%,rgba(0,0,0,0)_75%)]">
            <div ref={box} className="relative">
                <label htmlFor={inputId}
                    className="absolute left-2 top-1/2 -translate-y-1/2 z-10 flex items-center gap-2 h-12 md:h-14 px-3 md:pl-4 md:pr-5
                               rounded-full bg-black text-white cursor-text">
                    <Mark className="w-6 h-6 md:w-7 md:h-7" />
                    <span className="hidden sm:inline text-[13px] font-medium uppercase tracking-[0.02em]">Find matches</span>
                </label>
                <input {...combo} placeholder="Type a film you love…"
                    className="w-full h-16 md:h-[72px] rounded-full bg-white text-black placeholder:text-black/45
                               pl-[4.25rem] sm:pl-[11.5rem] md:pl-[12.5rem] pr-[4.25rem] sm:pr-[12rem]
                               text-[17px] md:text-[21px] tracking-[-0.01em] outline-none
                               focus-visible:outline-2 focus-visible:outline-white focus-visible:outline-offset-4" />
                <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                    {busy && <Loader2 aria-hidden="true" className="w-5 h-5 animate-spin text-black/50" />}
                    <button type="button" onClick={() => target ? choose(target) : input.current?.focus()}
                        aria-label={target ? `See the 24 matches for ${target.title}` : 'Find matches'}
                        className="pill pill-ultra h-12 md:h-14 px-4 md:px-6 text-[13px]">
                        <span className="hidden sm:inline">24 matches</span>
                        <ArrowUpRight className="w-4 h-4" />
                    </button>
                </div>
                {list}
            </div>

            <div className="mt-3 flex items-center gap-2 overflow-x-auto scroller px-1">
                <span className="text-[13px] text-white/55 shrink-0 mr-0.5">Try</span>
                {TRY.map((t, i) => {
                    const m = tries[i];
                    return (
                        <button key={t} type="button"
                            onClick={async () => choose(m ?? await firstFilm(t))}
                            className="flex items-center gap-2 h-9 pl-1 pr-3.5 rounded-full border border-white/20 hover:border-white/60
                                       text-[13px] shrink-0 transition-colors">
                            <span className="w-7 h-7 rounded-full overflow-hidden bg-panel-hi shrink-0">
                                {m?.posterPath && <img src={img(m.posterPath, 'w185')} alt="" className="w-full h-full object-cover" />}
                            </span>
                            {m ? short(m.title) : t}
                        </button>
                    );
                })}
            </div>
        </section>
    );
};

export default MatchSearch;
