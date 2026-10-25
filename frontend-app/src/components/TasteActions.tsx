import { Bookmark, Heart, ThumbsDown } from 'lucide-react';
import { Movie } from '../types';
import { Taste } from '../lib/taste';
import { short } from '../lib/format';
import { cn } from '../lib/utils';

const ON = 'bg-white text-black border-white hover:border-white';

interface Props {
    movie: Movie;
    taste: Taste;
    tabIndex?: number;
    /** Set where the buttons sit inside something clickable, like a hero slide. */
    stop?: boolean;
    className?: string;
}

/**
 * Like, watchlist or pass on a film. Each is a toggle — pressing it again takes
 * it back — and a film holds one of the three at a time, the latest choice.
 * Everything here feeds the "For you" shelf.
 */
const TasteActions = ({ movie, taste, tabIndex, stop, className }: Props) => {
    const liked = taste.has('like', movie);
    const listed = taste.has('watchlist', movie);
    const passed = taste.has('skip', movie);
    const name = short(movie.title);
    const act = (kind: 'like' | 'watchlist' | 'skip') => (e: React.MouseEvent) => {
        if (stop) e.stopPropagation();
        taste.toggle(kind, movie);
    };

    return (
        <div className={cn('flex gap-1.5', className)}>
            <button type="button" tabIndex={tabIndex} aria-pressed={liked} onClick={act('like')}
                className={cn('pill h-10', liked ? 'pill-white' : 'pill-violet')}>
                <Heart className={cn('w-4 h-4', liked && 'fill-current')} /> {liked ? 'Liked' : 'Like'}
            </button>
            <button type="button" tabIndex={tabIndex} aria-pressed={listed} onClick={act('watchlist')}
                aria-label={listed ? `Remove ${name} from your watchlist` : `Add ${name} to your watchlist`}
                title={listed ? 'On your watchlist' : 'Watchlist'}
                className={cn('round-btn w-10 h-10', listed ? ON : 'bg-black/40 backdrop-blur')}>
                <Bookmark className={cn('w-4 h-4', listed && 'fill-current')} />
            </button>
            <button type="button" tabIndex={tabIndex} aria-pressed={passed} onClick={act('skip')}
                aria-label={passed ? `Undo not for me on ${name}` : `${name} is not for me`}
                title={passed ? 'Marked not for me' : 'Not for me'}
                className={cn('round-btn w-10 h-10', passed ? ON : 'bg-black/40 backdrop-blur')}>
                <ThumbsDown className={cn('w-4 h-4', passed && 'fill-current')} />
            </button>
        </div>
    );
};

export default TasteActions;
