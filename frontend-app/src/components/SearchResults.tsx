import { motion } from 'framer-motion';
import { Movie } from '../types';
import MovieCard from './MovieCard';
import { ArrowLeft, SearchX } from 'lucide-react';

interface SearchResultsProps {
    query: string;
    results: Movie[];
    loading: boolean;
    onBack: () => void;
    onSelectMovie: (movie: Movie) => void;
}

const SearchResults = ({ query, results, loading, onBack, onSelectMovie }: SearchResultsProps) => {

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
                <div>
                    <h2 className="text-3xl font-display font-bold text-white">
                        Results for <span className="text-primary">"{query}"</span>
                    </h2>
                    <p className="text-neutral-400 mt-1">
                        Found {results.length} movies matching your intent
                    </p>
                </div>
            </div>

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
                        We couldn't find any movies matching "{query}" in our database or by tag.
                        Try searching for a genre like "Action" or "Hindi".
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
                            key={movie.tmdbId}
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
