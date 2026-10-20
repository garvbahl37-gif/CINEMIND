import { memo, ReactNode, useState } from 'react';
import { Movie } from '../types';
import { img } from '../lib/api';
import { pct, runtime, short, stars } from '../lib/format';
import { cn } from '../lib/utils';
import { Asterisk } from './Marks';

interface Props {
    movie: Movie;
    onSelect?: (m: Movie) => void;
    /** What is struck large in the top corner. */
    figure?: 'rating' | 'match' | 'none';
    /** A small disc in the top-left corner — a rank, usually. */
    badge?: ReactNode;
    size?: 'sm' | 'md' | 'lg';
    className?: string;
    /** Set on the stage, where the card is driven by its parent. */
    tabIndex?: number;
}

const WIDTH = {
    sm: 'w-[112px]',
    md: 'w-[150px] sm:w-[172px] lg:w-[196px]',
    lg: 'w-full',
};

/** The number in the corner, with its superscript unit. */
export const Figure = ({ movie, kind, className }: {
    movie: Movie; kind: 'rating' | 'match'; className?: string;
}) => {
    const v = kind === 'match' ? pct(movie.score) : stars(movie.rating);
    if (v == null) return null;
    return (
        <span className={cn('numeral inline-flex items-start text-white', className)}>
            {v}
            {kind === 'match'
                ? <span className="text-[0.36em] font-normal tracking-normal ml-[0.06em] mt-[0.02em]">%</span>
                : <Asterisk className="w-[0.24em] h-[0.24em] ml-[0.1em] mt-[0.02em]" />}
        </span>
    );
};

/** When a film has no artwork: its title, set in the card. */
const NoArt = ({ title }: { title: string }) => (
    <div className="absolute inset-0 flex items-end p-4 pb-20 bg-[linear-gradient(160deg,#2B1FBB_0%,#120a4f_55%,#000_100%)]">
        <span className="display text-[26px] text-white/90 break-words">{short(title)}</span>
    </div>
);

/**
 * A film, as a card: poster, a big number in the corner, title and pills at the
 * foot. Ratings are the MovieLens average out of five; match strength is the
 * neighbour score as a percentage of the closest match.
 */
const PosterCard = ({ movie, onSelect, figure = 'rating', badge, size = 'md', className, tabIndex }: Props) => {
    const [failed, setFailed] = useState(false);
    const src = img(movie.posterPath, size === 'sm' ? 'w185' : 'w342');
    const label = [
        movie.title,
        movie.year,
        figure === 'match' && movie.score != null ? `${pct(movie.score)}% match` : null,
        figure === 'rating' && movie.rating != null ? `rated ${stars(movie.rating)} of 5` : null,
    ].filter(Boolean).join(', ');

    return (
        <button
            type="button"
            tabIndex={tabIndex}
            onClick={() => onSelect?.(movie)}
            aria-label={label}
            className={cn(
                'group relative shrink-0 snap-start overflow-hidden text-left bg-panel-hi aspect-[2/3]',
                size === 'sm' ? 'rounded-[12px]' : 'rounded-[18px]',
                WIDTH[size], className,
            )}
        >
            {src && !failed
                ? <img src={src} alt="" loading="lazy" decoding="async" onError={() => setFailed(true)}
                    className="absolute inset-0 w-full h-full object-cover" />
                : <NoArt title={movie.title} />}

            {/* Scrims: the top for the number, the foot for the title. */}
            <span aria-hidden="true" className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(0,0,0,.55)_0%,rgba(0,0,0,0)_24%,rgba(0,0,0,0)_44%,rgba(0,0,0,.82)_68%,rgba(0,0,0,.96)_100%)]" />

            {badge != null && (
                <span className={cn(
                    'absolute top-2.5 left-2.5 grid place-items-center rounded-full bg-black/70 text-white font-medium tabular-nums',
                    size === 'sm' ? 'w-5 h-5 text-[10px]' : 'min-w-7 h-7 px-1.5 text-[12px]',
                )}>
                    {badge}
                </span>
            )}

            {figure !== 'none' && (
                <Figure movie={movie} kind={figure}
                    className={cn('absolute top-2.5 right-3',
                        size === 'sm' ? 'text-[22px]' : 'text-[34px] lg:text-[40px]')} />
            )}

            <span className={cn('absolute inset-x-0 bottom-0 flex flex-col', size === 'sm' ? 'p-2' : 'p-3.5')}>
                <span className={cn('font-medium leading-[1.15] line-clamp-2 text-white',
                    size === 'sm' ? 'text-[11px]' : 'text-[15px]')}>
                    {short(movie.title)}
                </span>
                {size !== 'sm' && (
                    <span className="mt-2 flex gap-1">
                        {movie.year && <span className="pill-sm border border-white/30 text-white/90">{movie.year}</span>}
                        {movie.runtime && <span className="pill-sm border border-white/30 text-white/90">{runtime(movie.runtime)}</span>}
                    </span>
                )}
            </span>

            <span aria-hidden="true" className={cn(
                'absolute inset-0 ring-1 ring-inset ring-white/10 group-hover:ring-white/60 transition-[box-shadow]',
                size === 'sm' ? 'rounded-[12px]' : 'rounded-[18px]',
            )} />
        </button>
    );
};

export default memo(PosterCard);
