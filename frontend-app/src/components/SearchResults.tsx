import { motion } from 'framer-motion';
import { Movie } from '../types';
import MovieCard from './MovieCard';
import FilterBar from './FilterBar';
import { ArrowLeft, SearchX } from 'lucide-react';
import { Facets, Filters, EMPTY } from '../lib/search';

interface SearchResultsProps {
    query: string;
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
}

/** What this result set is, in words, whether it came from text or from chips. */
const describe = (query: string, f?: Filters, langLabel?: string) => {
    if (query) return query;
    const bits: string[] = [];
    if (langLabel) bits.push(langLabel);
    bits.push(...(f?.genres ?? []));
    if (f?.yearMin) bits.push(`${f.yearMin}s`);
    return bits.length ? bits.join(' ') : 'Everything';
};

const SearchResults = ({
    query, results, loading, onBack, onSelectMovie,
    filters, onFilters, facets, vocabulary, total,
}: SearchResultsProps) => {
    const langLabel = (vocabulary?.languages ?? facets?.languages ?? [])
        .find(l => l.value === filters?.lang)?.label;
    const heading = describe(query, filters, langLabel);
    const count = total ?? results.length;

    // Container animation
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1
            }
        }
    };

    return (
        <div className="min-h-screen pt-32 pb-20 px-4 md:px-12 relative z-10">
            {/* Header */}
            <div className="flex items-center gap-4 mb-8">
                <button
                    onClick={onBack}
                    className="p-2 rounded-full bg-white/5 hover:bg-white/10 transition-colors group"
                >
                    <ArrowLeft className="w-6 h-6 text-neutral-400 group-hover:text-white transition-colors" />
                </button>
                <div className="min-w-0">
                    <h2 className="text-3xl font-display font-bold text-white truncate">
                        {query
                            ? <>Results for <span className="text-primary">"{query}"</span></>
                            : <span className="text-primary">{heading}</span>}
                    </h2>
                    <p className="text-neutral-400 mt-1">
                        {loading ? 'Searching…'
                            : `${count.toLocaleString()} ${count === 1 ? 'film' : 'films'}`}
                        {!loading && count > results.length && ` · showing the top ${results.length}`}
                    </p>
                </div>
            </div>

            {onFilters && (
                <FilterBar
                    filters={filters ?? EMPTY}
                    onChange={onFilters}
                    facets={facets ?? null}
                    vocabulary={vocabulary ?? null}
                />
            )}

            {/* Loading State */}
            {loading && (
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
                    {[...Array(10)].map((_, i) => (
                        <div key={i} className="aspect-[2/3] rounded-xl bg-white/5 animate-pulse" />
                    ))}
                </div>
            )}

            {/* Empty State */}
            {!loading && results.length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 text-center">
                    <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6">
                        <SearchX className="w-10 h-10 text-neutral-500" />
                    </div>
                    <h3 className="text-2xl font-bold text-white mb-2">No matches found</h3>
                    <p className="text-neutral-400 max-w-md">
                        Nothing matches {query ? `"${query}"` : 'those filters'}. The catalogue
                        covers 17,719 films — try removing a filter, or search a genre,
                        a decade like "90s", a director, or a title.
                    </p>
                    <button
                        onClick={onBack}
                        className="mt-8 px-6 py-3 bg-primary hover:bg-red-700 text-white rounded-full font-medium transition-colors"
                    >
                        Back to Browse
                    </button>
                </div>
            )}

            {/* Results Grid */}
            {!loading && results.length > 0 && (
                <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                    className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6"
                >
                    {results.map((movie) => (
                        <MovieCard
                            key={movie.item_id ?? movie.tmdbId}
                            movie={movie}
                            onSelect={onSelectMovie}
                            className="w-full md:w-full"
                        />
                    ))}
                </motion.div>
            )}
        </div>
    );
};

export default SearchResults;
