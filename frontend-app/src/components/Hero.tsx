import { useState, useEffect } from 'react';
import { Info, Search } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Button from './ui/Button';
import Badge from './ui/Badge';
import Input from './ui/Input';
import { Movie } from '../types';
import { TMDB_API_KEY, TMDB_IMAGE_BASE } from '../config';

interface HeroProps {
    featuredMovie: Movie | null;
    onSearch: (query: string) => void;
    suggestions: Movie[];
    onSelectSuggestion: (movie: Movie) => void;
}

const Hero = ({ featuredMovie, onSearch, suggestions, onSelectSuggestion }: HeroProps) => {
    const [imageSrc, setImageSrc] = useState<string | null>(null);

    useEffect(() => {
        if (!featuredMovie) return;

        // Use existing if available
        if (featuredMovie.backdrop || featuredMovie.poster) {
            setImageSrc(featuredMovie.backdrop || featuredMovie.poster || null);
            return;
        }

        // Fetch from TMDB
        const fetchImages = async () => {
            if (!featuredMovie.tmdbId) return;
            try {
                const response = await fetch(
                    `https://api.themoviedb.org/3/movie/${featuredMovie.tmdbId}?api_key=${TMDB_API_KEY}`
                );
                const data = await response.json();
                if (data.backdrop_path) {
                    setImageSrc(`${TMDB_IMAGE_BASE}/original${data.backdrop_path}`);
                } else if (data.poster_path) {
                    setImageSrc(`${TMDB_IMAGE_BASE}/original${data.poster_path}`);
                }
            } catch (error) {
                console.error("Failed to fetch hero image:", error);
            }
        };

        fetchImages();
    }, [featuredMovie]);

    if (!featuredMovie) return null;

    return (
        <div className="relative h-[90vh] w-full overflow-hidden">
            {/* Background Image with slight Parallax */}
            <motion.div
                className="absolute inset-0"
                initial={{ scale: 1.1 }}
                animate={{ scale: 1 }}
                transition={{ duration: 10, ease: "easeOut" }}
            >
                <img
                    src={imageSrc || `https://via.placeholder.com/1920x1080?text=${encodeURIComponent(featuredMovie.title)}`}
                    alt={featuredMovie.title}
                    className="w-full h-full object-cover object-top opacity-50"
                />

                {/* Advanced Gradient Overlays for "Cinema" feel */}
                <div className="absolute inset-0 bg-gradient-to-r from-black via-black/40 to-transparent" />
                <div className="absolute inset-0 bg-gradient-to-t from-black via-black/10 to-transparent" />
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_transparent_0%,_rgba(0,0,0,0.4)_100%)]" />
            </motion.div>

            {/* Content */}
            <div className="relative z-10 h-full max-w-[1800px] mx-auto px-4 md:px-12 flex flex-col justify-center pt-20">
                <motion.div
                    initial={{ opacity: 0, x: -50 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 1, delay: 0.2 }}
                    className="max-w-3xl space-y-8"
                >
                    {/* Tags/Badges */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 }}
                        className="flex items-center gap-4"
                    >
                        <Badge variant="primary" className="px-4 py-1.5 text-sm uppercase tracking-wider font-bold">
                            Featured Premiere
                        </Badge>
                        {featuredMovie.vote_average && (
                            <div className="flex items-center gap-2 text-primary font-bold">
                                <span className="text-3xl font-display">{featuredMovie.vote_average.toFixed(1)}</span>
                                <span className="text-sm text-gray-400 font-medium uppercase tracking-wide">Rating</span>
                            </div>
                        )}
                    </motion.div>

                    {/* Title */}
                    <h1 className="text-5xl md:text-8xl font-display font-black leading-[0.9] text-white drop-shadow-2xl tracking-tighter">
                        {featuredMovie.title}
                    </h1>

                    {/* Overview */}
                    <p className="text-lg md:text-2xl text-gray-300 line-clamp-3 font-light leading-relaxed max-w-2xl text-shadow-sm">
                        {featuredMovie.overview}
                    </p>

                    {/* Centered Search Bar Section */}
                    <div className="w-full max-w-xl relative group z-50">
                        <div className="absolute -inset-1 bg-gradient-to-r from-primary to-rose-600 rounded-full blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
                        <Input
                            placeholder="What are you in the mood for? (e.g. Comedy, Action, Tom Cruise)"
                            onChange={(e) => onSearch(e.target.value)}
                            className="h-16 pl-14 bg-black/60 backdrop-blur-xl border border-white/10 text-xl rounded-full shadow-2xl focus:bg-black/90 focus:border-primary/50 transition-all placeholder:text-gray-500 text-white"
                            icon={<Search className="w-7 h-7 text-gray-400 group-focus-within:text-primary transition-colors mt-0.5" />}
                        />

                        {/* Search Suggestions Dropdown */}
                        <AnimatePresence>
                            {suggestions.length > 0 && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: 10 }}
                                    className="absolute top-full left-0 right-0 mt-2 bg-black/90 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl z-50"
                                >
                                    {suggestions.map((movie) => (
                                        <div
                                            key={movie.tmdbId}
                                            className="flex items-center gap-4 p-3 hover:bg-white/10 cursor-pointer transition-colors"
                                            onClick={() => onSelectSuggestion(movie)}
                                        >
                                            <img
                                                src={movie.poster}
                                                alt={movie.title}
                                                className="w-10 h-14 object-cover rounded-md bg-gray-800"
                                            />
                                            <div>
                                                <h4 className="text-white font-medium text-sm">{movie.title}</h4>
                                                <div className="flex items-center gap-2 text-xs text-gray-400">
                                                    <span>{movie.releaseDate?.split('-')[0] || 'N/A'}</span>
                                                    <span>•</span>
                                                    <span className="text-primary">{movie.vote_average?.toFixed(1)}</span>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Actions - Removed "Watch Trailer" as requested */}
                    <div className="flex items-center gap-6 pt-2">
                        <Button variant="glass" size="lg" icon={Info} className="rounded-full px-10 hover:bg-white/10 hover:border-white/40">
                            More Details
                        </Button>
                    </div>
                </motion.div>
            </div>
        </div>
    );
};

export default Hero;
