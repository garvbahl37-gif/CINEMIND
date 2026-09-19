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
    total: number;
}

const Chip = ({ label, count, on, disabled, onClick }: {
    label: string; count?: number; on: boolean; disabled?: boolean; onClick: () => void;
}) => (
    <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        aria-pressed={on}
        className={cn(
            'px-3.5 py-2.5 sm:py-2 rounded-full border text-[11px] font-bold uppercase',
            'tracking-[0.1em] transition-colors focus:outline-none',
            'focus-visible:ring-1 focus-visible:ring-primary',
            on
                ? 'bg-primary text-white border-primary shadow-[0_0_18px_-4px_rgba(220,38,38,0.7)]'
                : disabled
                    ? 'bg-white/[0.02] text-neutral-700 border-white/5 cursor-not-allowed'
                    : 'bg-white/[0.04] text-neutral-400 border-white/10 hover:text-white hover:border-primary/40 hover:bg-primary/10',
        )}
    >
        {label}
        {count !== undefined && (
            <span className={cn('ml-1.5 tabular-nums font-medium',
                on ? 'text-white/70' : 'text-neutral-600')}>
                {count.toLocaleString()}
            </span>
        )}
    </button>
);

const Group = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-4">
        <span className="w-24 shrink-0 pt-2 text-[10px] font-bold uppercase tracking-[0.18em] text-neutral-600">
            {title}
        </span>
        <div className="flex flex-wrap gap-2">{children}</div>
    </div>
);

/**
 * The filter bar for a result set.
 *
 * Genres combine with AND — picking Action and Comedy means action-comedies,
 * not everything that is either. Counts come from the server for the set you
 * are already looking at, so a chip reading 0 really is a dead end.
 */
const FilterBar = ({ filters, onChange, facets, vocabulary, total }: Props) => {
    // Nineteen genres plus decades and languages is most of a phone screen, so
    // the panel starts collapsed there and the results stay the first thing seen.
    const [open, setOpen] = useState(
        () => typeof window === 'undefined' || window.innerWidth >= 640);
    const applied = filters.genres.length + (filters.lang ? 1 : 0) + (filters.yearMin ? 1 : 0);

    const counts = new Map((facets?.genres ?? []).map(g => [String(g.value), g.count]));
    const genres = (vocabulary?.genres ?? facets?.genres ?? []).map(g => String(g.value));
    const decades = facets?.decades ?? vocabulary?.decades ?? [];
    // The picked language leads, so it can't fall outside the visible chips and
    // leave the bar looking as though nothing is selected.
    const langs = facets?.languages ?? vocabulary?.languages ?? [];
    const languages = filters.lang
        ? [...langs.filter(l => l.value === filters.lang),
           ...langs.filter(l => l.value !== filters.lang)]
        : langs;

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

    const langLabel = languages.find(l => l.value === filters.lang)?.label;

    return (
        <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="bg-neutral-900/50 backdrop-blur-xl border border-white/10 rounded-2xl
                       p-4 sm:p-5 mb-8 space-y-4"
        >
            <div className="flex flex-wrap items-center justify-between gap-3">
                <button
                    type="button"
                    onClick={() => setOpen(o => !o)}
                    aria-expanded={open}
                    aria-controls="filter-groups"
                    className="flex items-center gap-2 text-neutral-400 hover:text-white
                               transition-colors focus:outline-none focus-visible:ring-1
                               focus-visible:ring-primary rounded-full py-1 pr-2"
                >
                    <SlidersHorizontal className="w-4 h-4 text-primary" />
                    <span className="text-[11px] font-bold uppercase tracking-[0.18em]">Refine</span>
                    {applied > 0 && (
                        <span className="px-1.5 py-0.5 rounded-full bg-primary text-white
                                         text-[10px] font-bold tabular-nums">{applied}</span>
                    )}
                    <span className="text-[11px] text-neutral-600 tabular-nums">
                        {total.toLocaleString()} {total === 1 ? 'film' : 'films'}
                    </span>
                    <ChevronDown className={cn('w-3.5 h-3.5 transition-transform',
                        open && 'rotate-180')} />
                </button>

                <div className="flex items-center gap-2">
                    <label htmlFor="sort" className="sr-only">Sort results</label>
                    <select
                        id="sort"
                        value={filters.sort}
                        onChange={(e) => onChange({ ...filters, sort: e.target.value as Sort })}
                        className="bg-white/[0.04] border border-white/10 rounded-full px-3.5 py-2
                                   text-[11px] font-bold uppercase tracking-[0.1em] text-neutral-300
                                   focus:outline-none focus:ring-1 focus:ring-primary cursor-pointer"
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
                            className="flex items-center gap-1.5 px-3.5 py-2 rounded-full border
                                       border-white/10 text-[11px] font-bold uppercase tracking-[0.1em]
                                       text-neutral-400 hover:text-white hover:border-primary/40
                                       focus:outline-none focus-visible:ring-1 focus-visible:ring-primary
                                       transition-colors"
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
                className="space-y-3 pt-1 border-t border-white/5 overflow-hidden">
                <div className="pt-3">
                    <Group title="Genre">
                        {genres.map(g => {
                            const on = filters.genres.includes(g);
                            const n = counts.get(g) ?? 0;
                            return (
                                <Chip key={g} label={g} count={facets ? n : undefined} on={on}
                                    disabled={!on && facets != null && n === 0}
                                    onClick={() => toggleGenre(g)} />
                            );
                        })}
                    </Group>
                </div>

                {decades.length > 0 && (
                    <Group title="Decade">
                        {decades.map(d => (
                            <Chip key={d.value} label={d.label ?? `${d.value}s`} count={d.count}
                                on={filters.yearMin === Number(d.value)}
                                onClick={() => setDecade(Number(d.value))} />
                        ))}
                    </Group>
                )}

                {languages.length > 1 && (
                    <Group title="Language">
                        {languages.slice(0, 7).map(l => (
                            <Chip key={l.value} label={l.label ?? String(l.value)} count={l.count}
                                on={filters.lang === l.value}
                                onClick={() => onChange({
                                    ...filters,
                                    lang: filters.lang === l.value ? undefined : String(l.value),
                                })} />
                        ))}
                        {languages.length > 7 && (
                            <>
                                <label htmlFor="lang" className="sr-only">More languages</label>
                                <select
                                    id="lang"
                                    value={languages.slice(0, 7).some(l => l.value === filters.lang)
                                        ? '' : (filters.lang ?? '')}
                                    onChange={(e) => onChange({ ...filters, lang: e.target.value || undefined })}
                                    className="bg-white/[0.04] border border-white/10 rounded-full px-3.5 py-2
                                               text-[11px] font-bold uppercase tracking-[0.1em] text-neutral-400
                                               focus:outline-none focus:ring-1 focus:ring-primary cursor-pointer"
                                >
                                    <option value="" className="bg-neutral-900 normal-case">
                                        {langLabel && !languages.slice(0, 7).some(l => l.value === filters.lang)
                                            ? langLabel : 'More…'}
                                    </option>
                                    {languages.slice(7).map(l => (
                                        <option key={l.value} value={l.value} className="bg-neutral-900 normal-case">
                                            {l.label} ({l.count.toLocaleString()})
                                        </option>
                                    ))}
                                </select>
                            </>
                        )}
                    </Group>
                )}
            </motion.div>
            )}
            </AnimatePresence>
        </motion.div>
    );
};

export default FilterBar;
