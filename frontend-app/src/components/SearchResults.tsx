import { ArrowLeft } from 'lucide-react';
import { Movie } from '../types';
import PosterCard from './PosterCard';
import FilterBar from './FilterBar';
import { Ground, Look } from './Collections';
import { Facets, Filters, EMPTY } from '../lib/search';
import { cn } from '../lib/utils';

interface SearchResultsProps {
    query: string;
    /** Set when the page has a name of its own — a collection, the Top 50. */
    title?: string;
    look?: Look;
    backdrop?: string;
    note?: string;
    results: Movie[];
    loading: boolean;
    onBack: () => void;
    onSelectMovie: (movie: Movie) => void;
    /** Supplying this turns on the filter bar; the fixed lists leave it out. */
    filters?: Filters;
    onFilters?: (f: Filters) => void;
    facets?: Facets | null;
    vocabulary?: Facets | null;
    total?: number;
    /** Number the cards: the list is a ranking. */
    ranked?: boolean;
}

/** What this result set is, in words, whether it came from text or from the filters. */
const describe = (query: string, f?: Filters, langLabel?: string) => {
    if (query) return query;
    const bits: string[] = [];
    if (langLabel) bits.push(langLabel);
    bits.push(...(f?.genres ?? []));
    if (f?.yearMin) bits.push(`${f.yearMin}s`);
    return bits.length ? bits.join(' ') : 'Everything';
};

const SearchResults = ({
    query, title, look = 'ultra', backdrop, note, results, loading, onBack, onSelectMovie,
    filters, onFilters, facets, vocabulary, total, ranked,
}: SearchResultsProps) => {
    const langLabel = (vocabulary?.languages ?? facets?.languages ?? [])
        .find(l => l.value === filters?.lang)?.label;
    const heading = title ?? describe(query, filters, langLabel);
    const count = total ?? results.length;
    const ink = look === 'paper' ? 'text-black' : 'text-white';

    return (
        <div className="space-y-3">
            {/* The banner wears the same ground as the collection tiles. */}
            <section className="relative overflow-hidden rounded-[24px] min-h-[260px] md:min-h-[320px] p-5 md:p-8 flex flex-col">
                <Ground look={look} backdrop={backdrop} />
                <button type="button" onClick={onBack} aria-label="Back to browse"
                    className={cn('relative round-btn w-10 h-10', look === 'paper' ? 'border-black/30 text-black hover:border-black' : 'bg-black/20')}>
                    <ArrowLeft className="w-4 h-4" />
                </button>
                <div className={cn('relative mt-auto pt-10 flex flex-wrap items-end justify-between gap-4', ink, look === 'paper' && 'mb-[min(12%,90px)]')}>
                    <h1 className="display text-[clamp(44px,8vw,120px)] max-w-[14ch] break-words">
                        {heading}
                    </h1>
                    <p className="text-[14px] pb-1.5 tabular-nums" aria-live="polite">
                        {loading ? 'Searching…' : `${count.toLocaleString()} ${count === 1 ? 'film' : 'films'}`}
                        {!loading && count > results.length && <span className="opacity-70">, showing the top {results.length}</span>}
                        {note && <span className="block opacity-70 mt-0.5">{note}</span>}
                    </p>
                </div>
            </section>

            <section className="bg-panel rounded-[24px] p-4 md:p-7">
                {onFilters && (
                    <FilterBar
                        filters={filters ?? EMPTY}
                        onChange={onFilters}
                        facets={facets ?? null}
                        vocabulary={vocabulary ?? null}
                    />
                )}

                {loading && (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
                        {Array.from({ length: 12 }, (_, i) => (
                            <div key={i} className="aspect-[2/3] rounded-[18px] bg-panel-hi animate-pulse" />
                        ))}
                    </div>
                )}

                {!loading && results.length === 0 && (
                    <div className="py-16 text-center">
                        <h2 className="text-[24px] font-medium">Nothing matches {query ? `“${query}”` : 'those filters'}</h2>
                        <p className="mt-2 text-white/60 max-w-[48ch] mx-auto">
                            Remove a filter, or search a genre, a decade like “90s”, a director or a title.
                        </p>
                        <button type="button" onClick={onBack} className="pill pill-white mt-6">Back to browse</button>
                    </div>
                )}

                {!loading && results.length > 0 && (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
                        {results.map((movie, i) => (
                            <PosterCard
                                key={movie.item_id ?? movie.tmdbId ?? i}
                                movie={movie}
                                onSelect={onSelectMovie}
                                size="lg"
                                badge={ranked ? i + 1 : undefined}
                            />
                        ))}
                    </div>
                )}
            </section>
        </div>
    );
};

export default SearchResults;
