import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import { X, Check, ChevronDown } from 'lucide-react';
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

/** Nominal panel width, used to decide which way it should open. */
const PANEL_PX = 240;

interface Option {
    value: string;
    label: string;
    count?: number;
    on: boolean;
    disabled?: boolean;
}

/**
 * One filter, as a menu.
 *
 * A menu rather than a row of chips: nineteen genres and forty languages laid
 * out as chips is a wall of equal-weight buttons that pushes the films off the
 * screen. Behind a trigger, the same options become a scannable list with the
 * counts in a column, and the bar stays one line tall whatever the catalogue
 * grows to.
 */
const Menu = ({ label, summary, active, options, onPick, align = 'left' }: {
    label: string;
    summary: string;
    active: boolean;
    options: Option[];
    onPick: (value: string) => void;
    align?: 'left' | 'right';
}) => {
    const [open, setOpen] = useState(false);
    // The bar wraps, so a trigger's position is not known from its order in it.
    // Flip the panel on open when opening leftwards would run off the screen.
    const [flip, setFlip] = useState(false);
    const host = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!open) return;
        const r = host.current?.getBoundingClientRect();
        if (r) setFlip(r.left + PANEL_PX > window.innerWidth - 16);
        const away = (e: MouseEvent) => {
            if (host.current && !host.current.contains(e.target as Node)) setOpen(false);
        };
        const esc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
        document.addEventListener('mousedown', away);
        document.addEventListener('keydown', esc);
        return () => {
            document.removeEventListener('mousedown', away);
            document.removeEventListener('keydown', esc);
        };
    }, [open]);

    return (
        <div ref={host} className="relative">
            <button
                type="button"
                onClick={() => setOpen(o => !o)}
                aria-expanded={open}
                aria-haspopup="true"
                className={cn(
                    'group flex items-center gap-2.5 h-10 pl-3 pr-2.5 rounded-lg border',
                    'transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-primary',
                    active
                        ? 'bg-primary/12 border-primary/60'
                        : 'bg-white/[0.03] border-white/10 hover:border-white/25',
                )}
            >
                <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-neutral-500">
                    {label}
                </span>
                <span className={cn('text-[12px] font-bold max-w-[13rem] truncate',
                    active ? 'text-white' : 'text-neutral-400')}>
                    {summary}
                </span>
                <ChevronDown className={cn('w-3.5 h-3.5 shrink-0 text-neutral-500 transition-transform',
                    open && 'rotate-180')} />
            </button>

            <AnimatePresence>
                {open && (
                    <motion.div
                        role="menu"
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.14 }}
                        className={cn(
                            'absolute top-full mt-2 z-50 min-w-[15rem] max-h-[19rem] overflow-y-auto',
                            'bg-neutral-950/[0.98] backdrop-blur-xl border border-white/10',
                            'rounded-xl shadow-2xl py-1.5',
                            'max-w-[calc(100vw-2rem)]',
                            align === 'right' || flip ? 'right-0' : 'left-0',
                        )}
                    >
                        {options.map(o => (
                            <button
                                key={o.value}
                                role="menuitemcheckbox"
                                aria-checked={o.on}
                                disabled={o.disabled}
                                onClick={() => onPick(o.value)}
                                className={cn(
                                    'w-full flex items-center gap-2.5 px-3 py-2 text-left transition-colors',
                                    o.disabled ? 'cursor-not-allowed opacity-35' : 'hover:bg-white/[0.06]',
                                )}
                            >
                                <span className={cn(
                                    'w-4 h-4 shrink-0 rounded border flex items-center justify-center',
                                    o.on ? 'bg-primary border-primary' : 'border-white/20',
                                )}>
                                    {o.on && <Check className="w-3 h-3 text-white" strokeWidth={3} />}
                                </span>
                                <span className={cn('flex-1 text-[12.5px] font-medium truncate',
                                    o.on ? 'text-white' : 'text-neutral-300')}>
                                    {o.label}
                                </span>
                                {o.count !== undefined && (
                                    <span className="text-[11px] tabular-nums text-neutral-500 shrink-0">
                                        {o.count.toLocaleString()}
                                    </span>
                                )}
                            </button>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

/** "Comedy, Romance" — but "Comedy +2" once the list outgrows the trigger. */
const summarise = (picked: string[], empty: string) =>
    picked.length === 0 ? empty
        : picked.length <= 2 ? picked.join(', ')
            : `${picked[0]} +${picked.length - 1}`;

/**
 * The filter bar for a result set.
 *
 * Genres combine with AND — picking Action and Comedy means action-comedies,
 * not everything that is either. Counts come from the server for the set you
 * are already looking at, so an option reading 0 really is a dead end.
 */
const FilterBar = ({ filters, onChange, facets, vocabulary }: Props) => {
    const counts = new Map((facets?.genres ?? []).map(g => [String(g.value), g.count]));
    const decades = facets?.decades ?? vocabulary?.decades ?? [];
    const languages = facets?.languages ?? vocabulary?.languages ?? [];

    // Selected first, then by how much each would leave: the useful genres lead
    // and the long tail (IMAX, Film-Noir) settles at the bottom of the list.
    const genres = (vocabulary?.genres ?? facets?.genres ?? [])
        .map(g => String(g.value))
        .sort((a, b) => {
            const sa = filters.genres.includes(a), sb = filters.genres.includes(b);
            if (sa !== sb) return sa ? -1 : 1;
            return (counts.get(b) ?? 0) - (counts.get(a) ?? 0);
        });

    const genreOptions: Option[] = genres.map(g => ({
        value: g,
        label: g,
        count: facets ? counts.get(g) ?? 0 : undefined,
        on: filters.genres.includes(g),
        disabled: !filters.genres.includes(g) && facets != null && (counts.get(g) ?? 0) === 0,
    }));

    const decadeOptions: Option[] = [
        { value: '', label: 'Any decade', on: !filters.yearMin },
        ...decades.map(d => ({
            value: String(d.value),
            label: d.label ?? `${d.value}s`,
            count: d.count,
            on: filters.yearMin === Number(d.value),
        })),
    ];

    const langOptions: Option[] = [
        { value: '', label: 'Any language', on: !filters.lang },
        ...languages.map(l => ({
            value: String(l.value),
            label: l.label ?? String(l.value),
            count: l.count,
            on: filters.lang === l.value,
        })),
    ];

    const sortOptions: Option[] = SORTS.map(s => ({
        value: s.value, label: s.label, on: filters.sort === s.value,
    }));

    const decadeLabel = filters.yearMin ? `${filters.yearMin}s` : 'Any';
    const langLabel = languages.find(l => l.value === filters.lang)?.label
        ?? (filters.lang ? filters.lang.toUpperCase() : 'Any');

    return (
        <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="flex flex-wrap items-center gap-2 mb-8"
        >
            <Menu
                label="Genre"
                summary={summarise(filters.genres, 'Any')}
                active={filters.genres.length > 0}
                options={genreOptions}
                onPick={(g) => onChange({
                    ...filters,
                    genres: filters.genres.includes(g)
                        ? filters.genres.filter(x => x !== g)
                        : [...filters.genres, g],
                })}
            />

            <Menu
                label="Decade"
                summary={decadeLabel}
                active={!!filters.yearMin}
                options={decadeOptions}
                onPick={(v) => onChange(v
                    ? { ...filters, yearMin: Number(v), yearMax: Number(v) + 9 }
                    : { ...filters, yearMin: undefined, yearMax: undefined })}
            />

            <Menu
                label="Language"
                summary={langLabel}
                active={!!filters.lang}
                options={langOptions}
                onPick={(v) => onChange({ ...filters, lang: v || undefined })}
            />

            <div className="hidden sm:block flex-1 min-w-0" />

            <Menu
                label="Sort"
                summary={SORTS.find(s => s.value === filters.sort)?.label ?? 'Best match'}
                active={filters.sort !== 'relevance'}
                options={sortOptions}
                align="right"
                onPick={(v) => onChange({ ...filters, sort: v as Sort })}
            />

            {isActive(filters) && (
                <button
                    onClick={() => onChange(EMPTY)}
                    className="h-10 flex items-center gap-1.5 px-3 rounded-lg border border-white/10
                               text-[11px] font-bold uppercase tracking-[0.08em] text-neutral-400
                               hover:text-white hover:border-primary/50 transition-colors
                               focus:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                >
                    <X className="w-3.5 h-3.5" /> Clear
                </button>
            )}
        </motion.div>
    );
};

export default FilterBar;
