import { AnimatePresence, motion } from 'framer-motion';
import { useState } from 'react';
import { X, SlidersHorizontal, ChevronDown } from 'lucide-react';
import { Filters, Facets, Sort, SORTS, EMPTY, isActive } from '../lib/search';
import { cn } from '../lib/utils';

interface Props {
    filters: Filters;
    onChange: (f: Filters) => void;
    /** Counts for the current result set; null until the first response. */
    facets: Facets | null;
    /** The full vocabulary, so the bar keeps its shape while counts move. */
    vocabulary: Facets | null;
}

/** How many genres to show before the rest go behind a disclosure. */
const GENRE_HEAD = 10;

const Chip = ({ label, count, on, disabled, onClick }: {
    label: string; count?: number; on: boolean; disabled?: boolean; onClick: () => void;
}) => (
    <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        aria-pressed={on}
        className={cn(
            'px-2.5 py-2.5 sm:py-1.5 rounded-lg border text-[11px] font-bold uppercase',
            'tracking-[0.06em] leading-none transition-colors focus:outline-none',
            'focus-visible:ring-1 focus-visible:ring-primary',
            on
                ? 'bg-primary text-white border-primary'
                : disabled
                    ? 'bg-transparent text-neutral-700 border-white/5 cursor-not-allowed'
                    : 'bg-white/[0.03] text-neutral-300 border-white/10 hover:text-white hover:border-primary/50 hover:bg-primary/10',
        )}
    >
        {label}
        {/* A selected chip's count is the result total, already in the header. */}
        {count !== undefined && !on && (
            <span className="ml-1.5 tabular-nums font-medium text-neutral-500">{count}</span>
        )}
    </button>
);

const Row = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="flex flex-col sm:flex-row sm:items-baseline gap-1.5 sm:gap-3">
        <span className="w-16 shrink-0 text-[10px] font-bold uppercase tracking-[0.16em] text-neutral-400">
            {title}
        </span>
        <div className="flex flex-wrap items-center gap-1.5">{children}</div>
    </div>
);

const selectCls =
    'bg-white/[0.03] border border-white/10 rounded-lg px-2.5 py-2.5 sm:py-[7px] ' +
    'text-[11px] font-bold uppercase tracking-[0.06em] leading-none text-neutral-300 ' +
    'focus:outline-none focus-visible:ring-1 focus-visible:ring-primary cursor-pointer ' +
    'hover:border-white/20 transition-colors';

/**
 * The filter bar for a result set.
 *
 * Genres combine with AND — picking Action and Comedy means action-comedies,
 * not everything that is either. Counts come from the server for the set you
 * are already looking at, so a chip reading 0 really is a dead end.
 */
const FilterBar = ({ filters, onChange, facets, vocabulary }: Props) => {
    // Nineteen genres, eleven decades and forty languages is most of a phone
    // screen, so the panel starts collapsed there and the films stay first.
    const [open, setOpen] = useState(
        () => typeof window === 'undefined' || window.innerWidth >= 640);
    const [allGenres, setAllGenres] = useState(false);
    const applied = filters.genres.length + (filters.lang ? 1 : 0) + (filters.yearMin ? 1 : 0);

    const counts = new Map((facets?.genres ?? []).map(g => [String(g.value), g.count]));
    const decades = facets?.decades ?? vocabulary?.decades ?? [];
    const languages = facets?.languages ?? vocabulary?.languages ?? [];

    // Selected first, then by how much each would leave: the useful ones lead
    // and the long tail (IMAX, Film-Noir) falls behind the disclosure.
    const genres = (vocabulary?.genres ?? facets?.genres ?? [])
        .map(g => String(g.value))
        .sort((a, b) => {
            const sa = filters.genres.includes(a), sb = filters.genres.includes(b);
            if (sa !== sb) return sa ? -1 : 1;
            return (counts.get(b) ?? 0) - (counts.get(a) ?? 0);
        });
    const shown = allGenres ? genres : genres.slice(0, GENRE_HEAD);
    const hidden = genres.length - shown.length;

    const toggleGenre = (g: string) => onChange({
        ...filters,
        genres: filters.genres.includes(g)
            ? filters.genres.filter(x => x !== g)
            : [...filters.genres, g],
    });

    const setDecade = (d: number) => onChange(
        filters.yearMin === d
            ? { ...filters, yearMin: undefined, yearMax: undefined }
            : { ...filters, yearMin: d, yearMax: d + 9 });

    return (
        <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="bg-neutral-900/50 backdrop-blur-xl border border-white/10 rounded-2xl
                       px-4 sm:px-5 py-3.5 mb-8"
        >
            <div className="flex flex-wrap items-center justify-between gap-3">
                <button
                    type="button"
                    onClick={() => setOpen(o => !o)}
                    aria-expanded={open}
                    aria-controls="filter-groups"
                    className="flex items-center gap-2 text-neutral-300 hover:text-white
                               transition-colors focus:outline-none focus-visible:ring-1
                               focus-visible:ring-primary rounded-lg py-1 pr-1.5"
                >
                    <SlidersHorizontal className="w-3.5 h-3.5 text-primary" />
                    <span className="text-[11px] font-bold uppercase tracking-[0.16em]">Refine</span>
                    {applied > 0 && (
                        <span className="px-1.5 py-0.5 rounded-md bg-primary text-white
                                         text-[10px] font-bold tabular-nums leading-none">{applied}</span>
                    )}
                    <ChevronDown className={cn('w-3.5 h-3.5 text-neutral-500 transition-transform',
                        open && 'rotate-180')} />
                </button>

                <div className="flex items-center gap-1.5">
                    <label htmlFor="sort" className="sr-only">Sort results</label>
                    <select
                        id="sort"
                        value={filters.sort}
                        onChange={(e) => onChange({ ...filters, sort: e.target.value as Sort })}
                        className={selectCls}
                    >
                        {SORTS.map(s => (
                            <option key={s.value} value={s.value} className="bg-neutral-900 normal-case">
                                {s.label}
                            </option>
                        ))}
                    </select>

                    {isActive(filters) && (
                        <button
                            onClick={() => onChange(EMPTY)}
                            className="flex items-center gap-1.5 px-2.5 py-2.5 sm:py-[7px] rounded-lg
                                       border border-white/10 text-[11px] font-bold uppercase
                                       leading-none tracking-[0.06em] text-neutral-400
                                       hover:text-white hover:border-primary/50 transition-colors
                                       focus:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                        >
                            <X className="w-3 h-3" /> Clear
                        </button>
                    )}
                </div>
            </div>

            <AnimatePresence initial={false}>
                {open && (
                    <motion.div
                        id="filter-groups"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                        className="overflow-hidden"
                    >
                        <div className="mt-3.5 pt-3.5 border-t border-white/[0.07] space-y-3">
                            <Row title="Genre">
                                {shown.map(g => {
                                    const on = filters.genres.includes(g);
                                    const n = counts.get(g) ?? 0;
                                    return (
                                        <Chip key={g} label={g} count={facets ? n : undefined} on={on}
                                            disabled={!on && facets != null && n === 0}
                                            onClick={() => toggleGenre(g)} />
                                    );
                                })}
                                {(hidden > 0 || allGenres) && (
                                    <button
                                        type="button"
                                        onClick={() => setAllGenres(a => !a)}
                                        className="px-2.5 py-2.5 sm:py-1.5 text-[11px] font-bold uppercase
                                                   tracking-[0.06em] leading-none text-neutral-500
                                                   hover:text-primary transition-colors focus:outline-none
                                                   focus-visible:ring-1 focus-visible:ring-primary rounded-lg"
                                    >
                                        {allGenres ? 'Fewer' : `+${hidden} more`}
                                    </button>
                                )}
                            </Row>

                            {(decades.length > 0 || languages.length > 1) && (
                                <div className="flex flex-col lg:flex-row lg:items-baseline gap-3 lg:gap-8">
                                    {decades.length > 0 && (
                                        <Row title="Decade">
                                            {decades.map(d => (
                                                <Chip key={d.value} label={d.label ?? `${d.value}s`}
                                                    count={d.count}
                                                    on={filters.yearMin === Number(d.value)}
                                                    onClick={() => setDecade(Number(d.value))} />
                                            ))}
                                        </Row>
                                    )}

                                    {languages.length > 1 && (
                                        <Row title="Language">
                                            <label htmlFor="lang" className="sr-only">Language</label>
                                            <select
                                                id="lang"
                                                value={filters.lang ?? ''}
                                                onChange={(e) => onChange({
                                                    ...filters, lang: e.target.value || undefined,
                                                })}
                                                className={cn(selectCls, filters.lang &&
                                                    'bg-primary border-primary text-white')}
                                            >
                                                <option value="" className="bg-neutral-900 normal-case">
                                                    Any language
                                                </option>
                                                {languages.map(l => (
                                                    <option key={l.value} value={l.value}
                                                        className="bg-neutral-900 normal-case">
                                                        {l.label} ({l.count.toLocaleString()})
                                                    </option>
                                                ))}
                                            </select>
                                        </Row>
                                    )}
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export default FilterBar;
