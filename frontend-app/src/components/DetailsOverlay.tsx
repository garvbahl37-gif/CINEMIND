import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Star } from 'lucide-react';
import Button from './ui/Button';
import Badge from './ui/Badge';
import { Movie } from '../types';
import { TMDB_API_KEY, TMDB_IMAGE_BASE } from '../config';

interface DetailsOverlayProps {
    movie: Movie | null;
    onClose: () => void;
    similarMovies?: Movie[];
    franchiseMovies?: Movie[];
    onSelectSimilar: (movie: Movie) => void;
}

const DetailsOverlay = ({ movie, onClose, similarMovies, franchiseMovies, onSelectSimilar }: DetailsOverlayProps) => {
    const [cast, setCast] = React.useState<any[]>([]);

    useEffect(() => {
        if (movie) {
            document.body.style.overflow = 'hidden';

            // Fetch Cast
            if (movie.tmdbId) {
                fetch(`https://api.themoviedb.org/3/movie/${movie.tmdbId}/credits?api_key=${TMDB_API_KEY}`)
                    .then(res => res.json())
                    .then(data => {
                        if (data.cast) {
                            setCast(data.cast.slice(0, 5));
                        }
                    })
                    .catch(err => console.error("Failed to fetch cast:", err));
            }

            return () => { document.body.style.overflow = 'unset'; };
        }
    }, [movie]);

    if (!movie) return null;

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-xl overflow-y-auto"
                onClick={onClose}
            >
                <div className="min-h-full flex items-center justify-center p-4 md:p-10">
                    <motion.div
                        className="relative w-full max-w-6xl bg-[#0a0a0a] rounded-[2rem] overflow-hidden shadow-2xl shadow-black ring-1 ring-white/5 flex flex-col pt-12 md:pt-0"
                        onClick={(e) => e.stopPropagation()}
                        initial={{ opacity: 0, y: 50, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 50, scale: 0.98 }}
                        transition={{ type: "spring", damping: 25, stiffness: 200 }}
                    >
                        {/* Close Button */}
                        <div className="absolute top-4 right-4 z-50">
                            <Button
                                variant="glass"
                                size="icon"
                                onClick={onClose}
                                className="rounded-full w-12 h-12 bg-black/50 hover:bg-white/20 border-white/10 backdrop-blur-md"
                            >
                                <X size={24} className="text-white" />
                            </Button>
                        </div>

                        {/* SPLIT LAYOUT CONTAINER */}
                        <div className="flex flex-col md:flex-row gap-8 md:gap-12 p-8 md:p-12 relative z-10">

                            {/* LEFT: POSTER (Fixed Aspect) */}
                            <div className="w-full md:w-[350px] shrink-0 flex flex-col gap-6">
                                <motion.div
                                    className="relative aspect-[2/3] rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)] ring-1 ring-white/10 bg-neutral-900"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.2 }}
                                >
                                    <img
                                        src={movie.poster || `https://via.placeholder.com/500x750?text=${encodeURIComponent(movie.title)}`}
                                        alt={movie.title}
                                        className="w-full h-full object-cover animate-in fade-in duration-700"
                                        loading="eager"
                                    />
                                </motion.div>
                            </div>

                            {/* RIGHT: INFO */}
                            <div className="flex-1 flex flex-col justify-center gap-6 md:pt-8">
                                <motion.h2
                                    className="text-5xl md:text-7xl font-display font-black tracking-tighter text-primary drop-shadow-lg leading-[0.9]"
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.3 }}
                                >
                                    {movie.title}
                                </motion.h2>

                                {/* Description */}
                                <motion.p
                                    className="text-lg md:text-xl text-gray-300 leading-relaxed font-light max-w-2xl"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.4 }}
                                >
                                    {movie.overview}
                                </motion.p>

                                {/* Metadata - Yellow Star Style */}
                                <motion.div
                                    className="flex items-center gap-6 mt-2"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.5 }}
                                >
                                    <div className="flex items-center gap-2 text-2xl font-bold text-yellow-500">
                                        <Star className="w-6 h-6 fill-current" />
                                        <span>{movie.vote_average ? movie.vote_average.toFixed(1) : 'NR'}/10</span>
                                    </div>

                                    <div className="text-gray-400 font-mono text-sm tracking-widest uppercase">
                                        Released: <span className="text-white">{movie.releaseDate || 'Unknown'}</span>
                                    </div>
                                </motion.div>

                                {/* Genres Pucks */}
                                <motion.div
                                    className="flex flex-wrap gap-2 mt-2"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.6 }}
                                >
                                    {movie.genres?.map(g => (
                                        <Badge key={g} variant="outline" className="border-white/10 text-gray-300 px-3 py-1 rounded-full text-xs hover:border-primary/50 hover:text-primary transition-colors cursor-default">
                                            {g}
                                        </Badge>
                                    ))}
                                </motion.div>
                            </div>
                        </div>




                        {/* RECOMMENDATIONS SECTION */}
                        {similarMovies && similarMovies.length > 0 && (
                            <div className="px-8 md:px-12 pt-8 pb-12">
                                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                                    Users who watched <span className="text-primary">{movie.title}</span> also liked
                                </h3>
                                <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-hide px-2 -mx-2">
                                    {similarMovies.slice(0, 25).map(sm => (
                                        <div
                                            key={sm.tmdbId}
                                            className="w-[200px] shrink-0 cursor-pointer group relative"
                                            onClick={() => onSelectSimilar(sm)}
                                        >
                                            <div className="aspect-[2/3] rounded-xl overflow-hidden mb-3 relative bg-neutral-800 ring-0 group-hover:ring-2 ring-primary transition-all duration-300">
                                                <img
                                                    src={sm.poster || `https://via.placeholder.com/300x450?text=${encodeURIComponent(sm.title)}`}
                                                    alt={sm.title}
                                                    className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity duration-300"
                                                />
                                                {/* Gradient Overlay for Text Readability */}
                                                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60 group-hover:opacity-40 transition-opacity" />

                                                {/* Rating Badge */}
                                                <div className="absolute top-2 right-2 z-20 bg-black/60 backdrop-blur-md border border-white/10 px-2 py-1 rounded-md flex items-center gap-1 shadow-lg">
                                                    <Star className="w-3 h-3 text-yellow-500 fill-current" />
                                                    <span className="text-[10px] font-bold text-white">
                                                        {(sm.vote_average || 0).toFixed(1)}
                                                    </span>
                                                </div>
                                            </div>
                                            <h4 className="text-sm font-bold text-white truncate group-hover:text-primary transition-colors pr-2">{sm.title}</h4>
                                            <div className="flex justify-between items-center mt-1">
                                                <p className="text-xs text-gray-500 font-mono">{sm.releaseDate?.split('-')[0]}</p>
                                                <Badge variant="outline" className="text-[10px] px-1 py-0 border-white/10 text-gray-400 group-hover:border-primary/50 group-hover:text-primary transition-colors">Movie</Badge>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* FRANCHISE / COLLECTION SECTION */}
                        {franchiseMovies && franchiseMovies.length > 0 && (
                            <div className="px-8 md:px-12 pt-4 pb-12">
                                <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                                    <span className="text-primary">Franchise</span> Collection
                                </h3>
                                <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-hide px-2 -mx-2">
                                    {franchiseMovies.map(fm => (
                                        <div
                                            key={fm.tmdbId}
                                            className="w-[200px] shrink-0 cursor-pointer group relative"
                                            onClick={() => onSelectSimilar(fm)}
                                        >
                                            <div className="aspect-[2/3] rounded-xl overflow-hidden mb-3 relative bg-neutral-800 ring-0 group-hover:ring-2 ring-primary transition-all duration-300">
                                                <img
                                                    src={fm.poster || `https://via.placeholder.com/300x450?text=${encodeURIComponent(fm.title)}`}
                                                    alt={fm.title}
                                                    className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity duration-300"
                                                />
                                                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60 group-hover:opacity-40 transition-opacity" />

                                                {/* Release Year Badge */}
                                                <div className="absolute top-2 left-2 z-20 bg-black/60 backdrop-blur-md border border-white/10 px-2 py-1 rounded-md">
                                                    <span className="text-[10px] font-bold text-white">
                                                        {fm.releaseDate?.split('-')[0]}
                                                    </span>
                                                </div>

                                                {/* Rating Badge */}
                                                <div className="absolute top-2 right-2 z-20 bg-black/60 backdrop-blur-md border border-white/10 px-2 py-1 rounded-md flex items-center gap-1 shadow-lg">
                                                    <Star className="w-3 h-3 text-yellow-500 fill-current" />
                                                    <span className="text-[10px] font-bold text-white">
                                                        {(fm.vote_average || 0).toFixed(1)}
                                                    </span>
                                                </div>
                                            </div>
                                            <h4 className="text-sm font-bold text-white truncate group-hover:text-primary transition-colors pr-2">{fm.title}</h4>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* CAST SECTION - Centered & Spaced */}
                        <div className="px-8 md:px-12 pb-16 pt-8 border-t border-white/5">
                            <h3 className="text-2xl font-bold text-white mb-8 text-center">Cast</h3>
                            <div className="flex flex-wrap justify-center gap-8 md:gap-10">
                                {cast.length > 0 ? (
                                    cast.map(actor => (
                                        <div key={actor.id} className="w-[140px] flex-shrink-0 group">
                                            <div className="h-[180px] w-full bg-neutral-800 rounded-xl overflow-hidden mb-4 relative shadow-lg">
                                                {actor.profile_path ? (
                                                    <img
                                                        src={`${TMDB_IMAGE_BASE}/w185${actor.profile_path}`}
                                                        alt={actor.name}
                                                        className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500 hover:scale-105"
                                                    />
                                                ) : (
                                                    <div className="w-full h-full flex items-center justify-center text-gray-500 font-bold text-2xl">
                                                        {actor.name[0]}
                                                    </div>
                                                )}
                                            </div>
                                            <p className="text-gray-100 font-medium text-base leading-tight text-center group-hover:text-primary transition-colors">{actor.name}</p>
                                            <p className="text-neutral-400 text-sm text-center mt-1.5 line-clamp-2">{actor.character}</p>
                                        </div>
                                    ))
                                ) : (
                                    <p className="text-gray-500 text-sm">Loading cast...</p>
                                )}
                            </div>
                        </div>

                    </motion.div>
                </div>
            </motion.div>
        </AnimatePresence>
    );
};

export default DetailsOverlay;
