import { useState, useEffect, useRef, useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Search, Globe, Tags, User, SlidersHorizontal, Clapperboard, CornerDownLeft } from 'lucide-react';
import { Movie } from '../types';
import { Category, fetchSuggestions } from '../lib/search';
import { cn } from '../lib/utils';
import { img } from '../lib/api';
import { short } from '../lib/format';

interface Props {
    value: string;
    onChange: (v: string) => void;
    /** Enter, or the magnifier: search the raw text and let the server parse it. */
    onSearch: (q: string) => void;
    /** A category row: apply its filters instead of searching for its letters. */
    onCategory: (c: Category) => void;
    onMovie: (m: Movie) => void;
    className?: string;
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
const SearchCommand = ({ value, onChange, onSearch, onCategory, onMovie, className }: Props) => {
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
        <div ref={boxRef} className={cn('relative w-full', className)}>
            <div className="relative">
                <input
                    ref={inputRef}
                    type="text"
                    role="combobox"
                    aria-expanded={show}
                    aria-controls="search-suggestions"
                    aria-autocomplete="list"
                    aria-label="Search films by title, genre, language, director or theme"
                    placeholder="Search a film, or “korean thrillers”, “90s horror”…"
                    value={value}
                    onChange={(e) => { onChange(e.target.value); setOpen(true); }}
                    onFocus={() => setOpen(true)}
                    onKeyDown={onKey}
                    className="w-full h-11 pl-5 pr-12 rounded-full bg-white/[0.13] hover:bg-white/[0.17] focus:bg-white/[0.2]
                               text-[14px] text-white placeholder:text-white/65 placeholder:truncate
                               outline-none focus-visible:outline-2 focus-visible:outline-white transition-colors"
                />
                <button
                    type="button"
                    aria-label="Search"
                    onClick={() => (value ? onSearch(value) : inputRef.current?.focus())}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 grid place-items-center w-8 h-8 rounded-full
                               text-white/80 hover:text-white hover:bg-white/10 transition-colors"
                >
                    <Search className="w-4 h-4" />
                </button>
            </div>

            <AnimatePresence>
                {show && (
                    <motion.div
                        id="search-suggestions"
                        role="listbox"
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.14 }}
                        className="absolute top-full mt-2 left-0 right-0 min-w-[min(100%,22rem)] z-50 max-h-[26rem] overflow-y-auto
                                   bg-black border border-white/12 rounded-[20px] shadow-[0_24px_60px_-12px_rgba(0,0,0,.9)] py-2"
                    >
                        {cats.length > 0 && (
                            <div className="px-4 pt-1.5 pb-1 text-[12px] text-white/50">Categories</div>
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
                                        'w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors',
                                        cursor === i ? 'bg-white/[0.08]' : 'hover:bg-white/[0.05]',
                                    )}
                                >
                                    <span className="w-9 h-9 shrink-0 rounded-full bg-ultra grid place-items-center text-white">
                                        <Icon className="w-4 h-4" />
                                    </span>
                                    <span className="flex-1 min-w-0">
                                        <span className="block text-white font-medium text-[14px] truncate">{c.label}</span>
                                        <span className="block text-[12px] text-white/50 truncate">
                                            {KIND_LABEL[c.kind] ?? 'Filter'}, {c.sublabel}
                                        </span>
                                    </span>
                                    {cursor === i && <CornerDownLeft className="w-3.5 h-3.5 text-white/50 shrink-0" />}
                                </button>
                            );
                        })}

                        {titles.length > 0 && (
                            <div className={cn('px-4 pt-2.5 pb-1 text-[12px] text-white/50',
                                cats.length > 0 && 'border-t border-white/10 mt-1.5')}>
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
                                        'w-full flex items-center gap-3 px-4 py-2 text-left transition-colors',
                                        cursor === i ? 'bg-white/[0.08]' : 'hover:bg-white/[0.05]',
                                    )}
                                >
                                    <span className="w-9 h-[54px] shrink-0 rounded-[8px] overflow-hidden bg-panel-hi">
                                        {m.posterPath && (
                                            <img src={img(m.posterPath, 'w185')} alt="" loading="lazy"
                                                className="w-full h-full object-cover" />
                                        )}
                                    </span>
                                    <span className="flex-1 min-w-0">
                                        <span className="block text-white font-medium text-[14px] truncate">{short(m.title)}</span>
                                        <span className="block text-[12px] text-white/50 truncate">
                                            {[m.year, m.genres?.slice(0, 2).join(', ')].filter(Boolean).join(', ')}
                                        </span>
                                    </span>
                                </button>
                            );
                        })}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default SearchCommand;
