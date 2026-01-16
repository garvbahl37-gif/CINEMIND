import { useRef } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import MovieCard from './MovieCard';
import Button from './ui/Button';
import { Movie } from '../types';

interface MovieRowProps {
    title: string;
    movies: Movie[];
    onSelectMovie: (movie: Movie) => void;
}

const MovieRow = ({ title, movies, onSelectMovie }: MovieRowProps) => {
    const rowRef = useRef<HTMLDivElement>(null);

    const scroll = (direction: 'left' | 'right') => {
        if (rowRef.current) {
            const { current } = rowRef;
            const scrollAmount = window.innerWidth * 0.7; // Scrolling 70% of screen width
            if (direction === 'left') {
                current.scrollBy({ left: -scrollAmount, behavior: 'smooth' });
            } else {
                current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
            }
        }
    };

    if (!movies || movies.length === 0) return null;

    return (
        <div className="relative py-8 space-y-4 group/row-container">
            {/* Header */}
            <h2 className="px-4 md:px-12 text-2xl font-bold font-display text-white tracking-wide flex items-center gap-2 group/title cursor-pointer w-fit">
                <span className="border-l-4 border-primary pl-3 transition-all duration-300 group-hover/title:border-white group-hover/title:pl-4">
                    {title}
                </span>
                <ChevronRight className="w-5 h-5 opacity-0 -translate-x-2 group-hover/title:opacity-100 group-hover/title:translate-x-0 transition-all duration-300 text-primary" />
            </h2>

            {/* Scroll Container */}
            <div className="relative group/scroll-area">
                {/* Left Chevron */}
                <div className="absolute left-0 top-0 bottom-0 w-16 bg-gradient-to-r from-black/90 to-transparent z-40 opacity-0 group-hover/scroll-area:opacity-100 transition-opacity duration-300 flex items-center justify-start pl-2 pointer-events-none group-hover/scroll-area:pointer-events-auto">
                    <Button
                        variant="glass"
                        size="icon"
                        onClick={() => scroll('left')}
                        className="rounded-full w-10 h-10 border-white/20 hover:border-primary hover:text-primary hover:bg-black/80"
                    >
                        <ChevronLeft className="w-6 h-6" />
                    </Button>
                </div>

                {/* Scrollable Area */}
                <div
                    ref={rowRef}
                    className="flex gap-4 md:gap-6 overflow-x-auto px-4 md:px-12 pb-8 pt-4 scrollbar-hide snap-x scroll-smooth no-scrollbar"
                    style={{
                        // Hide scrollbar for various browsers
                        msOverflowStyle: 'none',
                        scrollbarWidth: 'none'
                    }}
                >
                    {movies.slice(0, 20).map((movie) => (
                        <MovieCard
                            key={movie.tmdbId}
                            movie={movie}
                            onSelect={onSelectMovie}
                        />
                    ))}
                </div>

                {/* Right Chevron */}
                <div className="absolute right-0 top-0 bottom-0 w-16 bg-gradient-to-l from-black/90 to-transparent z-40 opacity-0 group-hover/scroll-area:opacity-100 transition-opacity duration-300 flex items-center justify-end pr-2 pointer-events-none group-hover/scroll-area:pointer-events-auto">
                    <Button
                        variant="glass"
                        size="icon"
                        onClick={() => scroll('right')}
                        className="rounded-full w-10 h-10 border-white/20 hover:border-primary hover:text-primary hover:bg-black/80"
                    >
                        <ChevronRight className="w-6 h-6" />
                    </Button>
                </div>
            </div>
        </div>
    );
};

export default MovieRow;
