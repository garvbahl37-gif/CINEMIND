import { useState, useEffect, useRef, useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Search, Globe, Tags, User, SlidersHorizontal, Clapperboard, CornerDownLeft } from 'lucide-react';
import { Movie } from '../types';
import { Category, fetchSuggestions } from '../lib/search';
import { cn } from '../lib/utils';

interface Props {
    value: string;
    onChange: (v: string) => void;
    /** Enter, or the magnifier: search the raw text and let the server parse it. */
    onSearch: (q: string) => void;
    /** A category row: apply its filters instead of searching for its letters. */
    onCategory: (c: Category) => void;
    onMovie: (m: Movie) => void;
    /** Genre names for the shortcut rail under the box. */
    quickGenres: string[];
}

const KIND_ICON = {
    genre: Clapperboard,
    language: Globe,
    filter: SlidersHorizontal,
    person: User,
    tag: Tags,
} as const;

const KIND_LABEL = {
    genre: 'Genre', language: 'Language', filter: 'Filter',
    person: 'Director / cast', tag: 'Theme',
} as const;

/**
 * The search box.
 *
 * Two kinds of answer share the dropdown: categories, which apply filters, and
 * titles, which open a film. Typing "rom" should be able to mean *Romance* —
 * matching only against titles is what made a genre impossible to search for.
 */
const SearchCommand = ({ value, onChange, onSearch, onCategory, onMovie, quickGenres }: Props) => {
    const [cats, setCats] = useState<Category[]>([]);
    const [titles, setTitles] = useState<Movie[]>([]);
    const [open, setOpen] = useState(false);
    const [cursor, setCursor] = useState(-1);
    const boxRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // One flat list, so the arrow keys walk both groups without special cases.
    const rows = useMemo(
        () => [...cats.map(c => ({ kind: 'cat' as const, c })),
        ...titles.map(m => ({ kind: 'title' as const, m }))],
        [cats, titles]);

    useEffect(() => {
        const q = value.trim();
        if (q.length < 2) { setCats([]); setTitles([]); return; }
        const ctrl = new AbortController();
        const t = setTimeout(() => {
            fetchSuggestions(q, ctrl.signal)
                .then(d => { setCats(d.categories); setTitles(d.titles.slice(0, 6)); setCursor(-1); })
                .catch(() => { });
        }, 140);
        return () => { clearTimeout(t); ctrl.abort(); };
    }, [value]);

    useEffect(() => {
        const away = (e: MouseEvent) => {
            if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
        };
        document.addEventListener('mousedown', away);
        return () => document.removeEventListener('mousedown', away);
    }, []);

    const choose = (i: number) => {
        const r = rows[i];
        if (!r) return;
        setOpen(false);
        if (r.kind === 'cat') onCategory(r.c); else onMovie(r.m);
    };

    const onKey = (e: React.KeyboardEvent) => {
        if (e.key === 'Escape') { setOpen(false); inputRef.current?.blur(); return; }
        if (e.key === 'Enter') {
            if (open && cursor >= 0) choose(cursor);
            else { setOpen(false); onSearch(value); }
            return;
        }
        if (!rows.length) return;
        if (e.key === 'ArrowDown') { e.preventDefault(); setOpen(true); setCursor(c => (c + 1) % rows.length); }
        if (e.key === 'ArrowUp') { e.preventDefault(); setOpen(true); setCursor(c => (c <= 0 ? rows.length : c) - 1); }
    };

    const show = open && rows.length > 0;

    return (
        <div ref={boxRef} className="w-full max-w-2xl">
          <div className="relative group z-[60]">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
                className="relative"
            >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-primary to-rose-600 rounded-2xl
                                blur opacity-20 group-hover:opacity-40 transition duration-500" />
                <input
                    ref={inputRef}
                    type="text"
                    role="combobox"
                    aria-expanded={show}
                    aria-controls="search-suggestions"
                    aria-label="Search films by title, genre, language, director or theme"
                    placeholder="Search a film, or a category — “sci-fi”, “korean thrillers”, “90s horror”"
                    value={value}
                    onChange={(e) => { onChange(e.target.value); setOpen(true); }}
                    onFocus={() => setOpen(true)}
                    onKeyDown={onKey}
                    className={cn(
                        'relative z-10 w-full h-12 md:h-14 pl-6 pr-12 bg-neutral-900/80 backdrop-blur-xl',
                        'border border-white/10 text-base md:text-lg text-white placeholder:text-neutral-500',
                        'focus:outline-none focus:ring-1 focus:ring-primary/50 shadow-2xl transition-all',
                        'placeholder:truncate',
                        show ? 'rounded-t-2xl rounded-b-none border-b-0' : 'rounded-2xl',
                    )}
                />
                <button
                    type="button"
                    aria-label="Search"
                    onClick={() => (value ? onSearch(value) : inputRef.current?.focus())}
                    className="absolute right-3 top-1/2 -translate-y-1/2 z-20 p-2 rounded-full
                               text-neutral-400 hover:text-primary hover:bg-white/5 transition-colors"
                >
                    <Search className="w-5 h-5" />
                </button>
            </motion.div>

            <AnimatePresence>
                {show && (
                    <motion.div
                        id="search-suggestions"
                        role="listbox"
                        initial={{ opacity: 0, y: -8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        transition={{ duration: 0.16 }}
                        className="absolute top-full left-0 right-0 bg-neutral-900/95 backdrop-blur-xl
                                   border border-t-0 border-white/10 rounded-b-2xl overflow-hidden
                                   shadow-2xl z-50 max-h-[26rem] overflow-y-auto"
                    >
                        {cats.length > 0 && (
                            <div className="px-4 pt-3 pb-1.5 text-[10px] font-bold uppercase
                                            tracking-[0.18em] text-neutral-500">
                                Browse by category
                            </div>
                        )}
                        {cats.map((c, i) => {
                            const Icon = KIND_ICON[c.kind] ?? Tags;
                            return (
                                <button
                                    key={`${c.kind}-${c.label}`}
                                    role="option"
                                    aria-selected={cursor === i}
                                    onMouseEnter={() => setCursor(i)}
                                    onClick={() => choose(i)}
                                    className={cn(
                                        'w-full flex items-center gap-3 px-4 py-3 text-left transition-colors',
                                        cursor === i ? 'bg-primary/15' : 'hover:bg-white/5',
                                    )}
                                >
                                    <span className="w-9 h-9 shrink-0 rounded-lg bg-primary/15 border border-primary/25
                                                     flex items-center justify-center text-primary">
                                        <Icon className="w-4 h-4" />
                                    </span>
                                    <span className="flex-1 min-w-0">
                                        <span className="block text-white font-bold text-sm truncate">{c.label}</span>
                                        <span className="block text-[11px] text-neutral-500 truncate">
                                            {KIND_LABEL[c.kind] ?? 'Filter'} · {c.sublabel}
                                        </span>
                                    </span>
                                    {cursor === i && <CornerDownLeft className="w-3.5 h-3.5 text-neutral-500 shrink-0" />}
                                </button>
                            );
                        })}

                        {titles.length > 0 && (
                            <div className={cn('px-4 pt-3 pb-1.5 text-[10px] font-bold uppercase tracking-[0.18em] text-neutral-500',
                                cats.length > 0 && 'border-t border-white/5 mt-1')}>
                                Films
                            </div>
                        )}
                        {titles.map((m, j) => {
                            const i = cats.length + j;
                            return (
                                <button
                                    key={m.item_id ?? m.tmdbId}
                                    role="option"
                                    aria-selected={cursor === i}
                                    onMouseEnter={() => setCursor(i)}
                                    onClick={() => choose(i)}
                                    className={cn(
                                        'w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors',
                                        cursor === i ? 'bg-primary/15' : 'hover:bg-white/5',
                                    )}
                                >
                                    <span className="w-9 h-[52px] shrink-0 rounded-md overflow-hidden bg-neutral-800">
                                        {m.poster && (
                                            <img src={m.poster} alt="" loading="lazy"
                                                className="w-full h-full object-cover" />
                                        )}
                                    </span>
                                    <span className="flex-1 min-w-0">
                                        <span className="block text-white font-bold text-sm truncate">{m.title}</span>
                                        <span className="block text-[11px] text-neutral-500 truncate">
                                            {m.releaseDate?.split('-')[0] ?? '—'}
                                            {m.genres?.length ? ` · ${m.genres.slice(0, 2).join(', ')}` : ''}
                                        </span>
                                    </span>
                                </button>
                            );
                        })}
                    </motion.div>
                )}
            </AnimatePresence>
          </div>

            {/* Category shortcuts — the fastest path to a genre, with no typing at all. */}
            <div className="relative z-[50] mt-4 flex flex-wrap justify-center gap-2">
                {quickGenres.map((g) => (
                    <button
                        key={g}
                        onClick={() => onCategory({
                            kind: 'genre', label: g, sublabel: '', filters: { genres: [g] },
                        })}
                        className="px-4 py-2.5 sm:py-2 rounded-full border border-white/10 bg-white/[0.04]
                                   text-[11px] font-bold uppercase tracking-[0.12em] text-neutral-400
                                   hover:text-white hover:border-primary/40 hover:bg-primary/10
                                   focus:outline-none focus-visible:ring-1 focus-visible:ring-primary
                                   transition-colors"
                    >
                        {g}
                    </button>
                ))}
            </div>
        </div>
    );
};

export default SearchCommand;
