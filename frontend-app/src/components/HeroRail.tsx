import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUpRight, ChevronLeft, ChevronRight } from 'lucide-react';
import { Movie } from '../types';
import { img } from '../lib/api';
import { names, runtime, short } from '../lib/format';
import { Taste } from '../lib/taste';
import { cn } from '../lib/utils';
import { Figure } from './PosterCard';
import TasteActions from './TasteActions';

interface Props {
    films: Movie[];
    onMatches: (m: Movie) => void;
    taste: Taste;
}

/**
 * The opening rail: one film full-bleed, the next one showing at the edge so
 * it is obvious the rail moves. Scroll-snap does the work, so trackpads,
 * touch and the arrow buttons all land on a whole slide.
 */
const HeroRail = ({ films, onMatches, taste }: Props) => {
    const rail = useRef<HTMLDivElement>(null);
    const [index, setIndex] = useState(0);

    const onScroll = useCallback(() => {
        const el = rail.current;
        const first = el?.firstElementChild as HTMLElement | null;
        if (!el || !first) return;
        setIndex(Math.round(el.scrollLeft / (first.offsetWidth + 12)));
    }, []);

    const go = (i: number) => {
        const el = rail.current;
        const target = el?.children[Math.max(0, Math.min(films.length - 1, i))] as HTMLElement | undefined;
        if (el && target) el.scrollTo({ left: target.offsetLeft - el.offsetLeft, behavior: 'smooth' });
    };

    useEffect(onScroll, [onScroll, films.length]);

    return (
        <section aria-roledescription="carousel" aria-label="Featured films" className="relative">
            <div ref={rail} onScroll={onScroll} className="scroller gap-3">
                {films.map((m, i) => {
                    const active = i === index;
                    return (
                        <article
                            key={m.item_id}
                            aria-roledescription="slide"
                            aria-label={`${i + 1} of ${films.length}: ${m.title}`}
                            onClick={() => !active && go(i)}
                            className={cn(
                                'relative shrink-0 snap-start overflow-hidden rounded-[24px] bg-panel',
                                'w-[calc(100%-44px)] md:w-[calc(100%-132px)]',
                                'h-[min(76vh,640px)] min-h-[440px]',
                                !active && 'cursor-pointer',
                            )}
                        >
                            {m.backdropPath && (
                                <img src={img(m.backdropPath, 'w1280')} alt=""
                                    loading={i < 2 ? 'eager' : 'lazy'} decoding="async"
                                    className="absolute inset-0 w-full h-full object-cover" />
                            )}
                            <div aria-hidden="true" className="absolute inset-0 bg-[linear-gradient(to_top,rgba(0,0,0,.92)_0%,rgba(0,0,0,.55)_32%,rgba(0,0,0,0)_62%),linear-gradient(to_right,rgba(0,0,0,.55)_0%,rgba(0,0,0,0)_55%)]" />

                            {m.directors?.length ? (
                                <p className="absolute top-5 right-5 hidden sm:block px-3.5 py-2 rounded-full bg-black/55 backdrop-blur-md text-[13px] text-white/75">
                                    Directed by <span className="text-white">{names(m.directors, 2)}</span>
                                </p>
                            ) : null}

                            <div className="absolute inset-x-0 bottom-0 p-5 md:p-9 flex flex-col md:flex-row md:items-end gap-6">
                                <div className="flex-1 min-w-0">
                                    <h2 className="text-[clamp(40px,6.4vw,92px)] font-medium leading-[0.9] tracking-[-0.04em] max-w-[13ch] text-balance">
                                        {short(m.title)}
                                    </h2>
                                    {m.overview && (
                                        <p className="mt-4 max-w-[52ch] text-[14px] md:text-[15px] leading-[1.5] text-white/80 line-clamp-3">
                                            {m.overview}
                                        </p>
                                    )}
                                    <div className="mt-5 flex flex-wrap gap-1.5">
                                        {m.genres?.slice(0, 3).map(g => (
                                            <span key={g} className="pill h-8 px-3.5 bg-white text-black normal-case text-[13px]">{g}</span>
                                        ))}
                                        {m.year && <span className="pill h-8 px-3.5 border border-white/40 normal-case text-[13px]">{m.year}</span>}
                                        {m.runtime && <span className="pill h-8 px-3.5 border border-white/40 normal-case text-[13px]">{runtime(m.runtime)}</span>}
                                    </div>
                                </div>

                                {/* On a phone the actions wrap under the rating rather than run off the slide. */}
                                <div className="flex flex-wrap md:flex-nowrap md:flex-col items-end justify-between gap-x-5 gap-y-4 md:shrink-0">
                                    <div className="text-right">
                                        <Figure movie={m} kind="rating" className="text-[64px] md:text-[96px]" />
                                        {m.rating_count ? (
                                            <p className="mt-2 text-[12px] text-white/70 tabular-nums">
                                                {m.rating_count.toLocaleString()} ratings
                                            </p>
                                        ) : null}
                                    </div>
                                    <div className="flex flex-wrap md:justify-end gap-1.5 max-w-full">
                                        <button type="button" tabIndex={active ? 0 : -1}
                                            onClick={(e) => { e.stopPropagation(); onMatches(m); }}
                                            className="pill pill-ultra h-10">
                                            24 matches <ArrowUpRight className="w-4 h-4" />
                                        </button>
                                        <TasteActions movie={m} taste={taste} tabIndex={active ? 0 : -1} stop />
                                    </div>
                                </div>
                            </div>
                        </article>
                    );
                })}
            </div>

            <div className="hidden md:flex absolute top-5 left-6 gap-1.5">
                <button type="button" className="round-btn bg-black/40 backdrop-blur" aria-label="Previous film"
                    disabled={index === 0} onClick={() => go(index - 1)}>
                    <ChevronLeft className="w-4 h-4" />
                </button>
                <button type="button" className="round-btn bg-black/40 backdrop-blur" aria-label="Next film"
                    disabled={index >= films.length - 1} onClick={() => go(index + 1)}>
                    <ChevronRight className="w-4 h-4" />
                </button>
                <span className="self-center ml-2 text-[13px] text-white/75 tabular-nums" aria-live="polite">
                    {index + 1} / {films.length}
                </span>
            </div>
        </section>
    );
};

export default HeroRail;
