import { useState, useEffect, memo } from 'react';
import { Star } from 'lucide-react';
import { Movie } from '../types';
import { TMDB_API_KEY, TMDB_IMAGE_BASE } from '../config';
import GlassSkeleton from './GlassSkeleton';
import { motion } from 'framer-motion';

interface MovieCardProps {
    movie: Movie;
    onSelect: (movie: Movie) => void;
}

const MovieCard = memo(({ movie, onSelect }: MovieCardProps) => {
    const [imageSrc, setImageSrc] = useState<string | null>(movie.poster || null);
    const [imageLoaded, setImageLoaded] = useState(false);
    const [rating, setRating] = useState<number | null>(movie.vote_average || null);
    const [releaseDate, setReleaseDate] = useState<string | null>(movie.releaseDate || null);

    useEffect(() => {
        const fetchMovieDetails = async () => {
            // If we have everything, don't fetch
            if ((imageSrc && !imageSrc.includes('via.placeholder')) && rating && releaseDate) return;
            if (!movie.tmdbId) return;

            try {
                const response = await fetch(
                    `https://api.themoviedb.org/3/movie/${movie.tmdbId}?api_key=${TMDB_API_KEY}`
                );
                const data = await response.json();

                // Update Image if needed
                if (data.poster_path && (!imageSrc || imageSrc.includes('via.placeholder'))) {
                    setImageSrc(`${TMDB_IMAGE_BASE}/w500${data.poster_path}`);
                }
                // Update Metadata if needed
                if (!rating && data.vote_average) setRating(data.vote_average);
                if (!releaseDate && data.release_date) setReleaseDate(data.release_date);

            } catch (error) {
                // Silent fail
            }
        };

        fetchMovieDetails();
    }, [movie.tmdbId, movie.poster, rating, releaseDate]);

    return (
        <motion.div
            className="group relative flex-shrink-0 w-[150px] md:w-[260px] aspect-[2/3] rounded-xl md:rounded-2xl overflow-hidden cursor-pointer snap-start bg-neutral-900 shadow-xl ring-1 ring-white/10 will-change-transform backface-hidden"
            onClick={() => onSelect({ ...movie, poster: imageSrc || movie.poster, vote_average: rating || movie.vote_average, releaseDate: releaseDate || movie.releaseDate })}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
        >
            {/* Loading Skeleton */}
            {!imageLoaded && (
                <div className="absolute inset-0 z-20">
                    <GlassSkeleton className="w-full h-full" />
                </div>
            )}

            {/* Poster */}
            <img
                src={imageSrc || `https://via.placeholder.com/300x450?text=${encodeURIComponent(movie.title)}`}
                alt={movie.title}
                className={`w-full h-full object-cover transition-opacity duration-500 ${!imageLoaded ? 'opacity-0' : 'opacity-100'}`}
                loading="lazy"
                onLoad={() => setImageLoaded(true)}
            />

            {/* Premium Cinematic Gradient Overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/95 via-black/50 to-transparent opacity-80 group-hover:opacity-100 transition-opacity duration-300" />

            {/* Subtle Inner Glow on Hover */}
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-gradient-to-t from-primary/10 to-transparent pointer-events-none mix-blend-overlay" />

            {/* Content Overlay */}
            <div className="absolute bottom-0 left-0 right-0 p-3 md:p-6 z-10 flex flex-col justify-end h-full">
                <div className="transform translate-y-2 group-hover:translate-y-0 transition-transform duration-200 ease-out">
                    <h3 className="text-white font-display font-bold text-sm md:text-xl leading-tight mb-1 md:mb-3 line-clamp-2 drop-shadow-lg group-hover:text-primary-foreground transition-colors tracking-tight">
                        {movie.title}
                    </h3>

                    <div className="flex items-center justify-between text-[10px] md:text-xs font-semibold text-gray-300 mb-2 md:mb-4 tracking-wide">
                        <div className="flex items-center gap-1.5 px-1.5 md:px-2 py-0.5 md:py-1 rounded-md bg-white/10 backdrop-blur-md border border-white/10">
                            <Star className="w-3 h-3 md:w-3.5 md:h-3.5 text-yellow-500 fill-current" />
                            <span className="text-white">{rating ? rating.toFixed(1) : '—'}</span>
                        </div>
                        <div className="px-1.5 md:px-2 py-0.5 md:py-1 rounded-md bg-black/40 backdrop-blur-sm border border-white/5 text-gray-400">
                            {releaseDate ? releaseDate.split('-')[0] : 'Unknown'}
                        </div>
                    </div>

                    <div className="hidden md:flex flex-wrap gap-2 h-0 opacity-0 group-hover:h-auto group-hover:opacity-100 transition-all duration-300 delay-75">
                        {movie.genres?.slice(0, 3).map((g) => (
                            <span key={g} className="text-[10px] font-medium px-2.5 py-1 rounded-full bg-primary/20 text-primary-foreground border border-primary/20 backdrop-blur-sm">
                                {g}
                            </span>
                        ))}
                    </div>
                </div>
            </div>
        </motion.div>
    );
});

export default MovieCard;
