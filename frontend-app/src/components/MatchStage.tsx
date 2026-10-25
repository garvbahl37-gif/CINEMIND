import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { ArrowUpRight, ChevronLeft, ChevronRight, X } from 'lucide-react';
import { Movie } from '../types';
import { fetchSimilar, img } from '../lib/api';
import { names, pct, runtime, short, stars } from '../lib/format';
import { Taste } from '../lib/taste';
import { cn } from '../lib/utils';
import { Figure } from './PosterCard';
import { Star } from './Marks';
import MatchSearch from './MatchSearch';
import TasteActions from './TasteActions';

interface Props {
    pick: Movie;
    onClose: () => void;
    taste: Taste;
}

interface Slot {
    m: Movie;
    /** 0 is your pick; 1–24 the neighbours in ranked order. */
    rank: number;
    /** Where it sits: 0 is the centre, negative to the left. */
    p: number;
}

/**
 * Where rank r sits. The closest match goes immediately right of your pick,
 * the second immediately left, and so on outwards — so distance from the
 * centre reads as distance in the model's space.
 */
const position = (rank: number) => (rank % 2 ? 1 : -1) * Math.ceil(rank / 2);

const SCALE = [1, 0.8, 0.68, 0.6, 0.54, 0.5, 0.47];
const DIM = [0, 0.3, 0.52, 0.64, 0.72, 0.78, 0.82];
const scaleAt = (d: number) => SCALE[Math.min(d, SCALE.length - 1)];

/** Horizontal centre of the card d places from focus, in px. */
const offsetAt = (d: number, w: number) => {
    let x = 0;
    for (let i = 1; i <= d; i++) {
        // Neighbours of the focus sit apart; further out they tuck under.
        const gap = i === 1 ? 16 : -w * 0.14;
        x += (w * scaleAt(i - 1)) / 2 + gap + (w * scaleAt(i)) / 2;
    }
    return x;
};

const useCardWidth = () => {
    const calc = () => {
        const vw = window.innerWidth, vh = window.innerHeight;
        return Math.round(Math.max(150, Math.min(vw * (vw < 640 ? 0.46 : 0.19), vh * 0.33, 300)));
    };
    const [w, setW] = useState(calc);
    useEffect(() => {
        const on = () => setW(calc());
        window.addEventListener('resize', on);
        return () => window.removeEventListener('resize', on);
    }, []);
    return w;
};

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const MatchStage = ({ pick, onClose, taste }: Props) => {
    const [trail, setTrail] = useState<Movie[]>([pick]);
    const current = trail[trail.length - 1];
    const [data, setData] = useState<{ source: Movie; matches: Movie[] } | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [attempt, setAttempt] = useState(0);
    const [focus, setFocus] = useState(0);
    const [entered, setEntered] = useState(false);

    const w = useCardWidth();
    const reduce = useReducedMotion();
    const dialog = useRef<HTMLDivElement>(null);
    const details = useRef<HTMLElement>(null);

    // A new pick from outside starts a new trail.
    useEffect(() => { setTrail([pick]); }, [pick]);

    useEffect(() => {
        setData(null); setError(null); setFocus(0); setEntered(false);
        if (current.item_id == null) { setError(`${current.title} is not in the model's index.`); return; }
        const ctrl = new AbortController();
        fetchSimilar(current.item_id, 24, ctrl.signal)
            .then(d => {
                setData(d);
                // Let the fan-out play once, then move without delays.
                window.setTimeout(() => setEntered(true), 1100);
            })
            .catch(e => { if (e.name !== 'AbortError') setError(`Couldn't load matches for ${short(current.title)}.`); });
        return () => ctrl.abort();
    }, [current, attempt]);

    const slots: Slot[] = useMemo(() => [
        { m: data?.source ? { ...current, ...data.source } : current, rank: 0, p: 0 },
        ...(data?.matches ?? []).map((m, i) => ({ m, rank: i + 1, p: position(i + 1) })),
    ], [data, current]);

    const lo = Math.min(...slots.map(s => s.p)), hi = Math.max(...slots.map(s => s.p));
    const focused = slots.find(s => s.p === focus) ?? slots[0];
    const move = useCallback((by: number) => setFocus(f => Math.max(lo, Math.min(hi, f + by))), [lo, hi]);

    const dive = useCallback((m: Movie) => {
        setTrail(t => [...t, m]);
        details.current?.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
    }, [reduce]);

    // Lock the page behind, take focus, and give it back on close.
    useEffect(() => {
        const before = document.activeElement as HTMLElement | null;
        const root = document.documentElement;
        const was = [root.style.overflow, document.body.style.overflow];
        root.style.overflow = 'hidden';
        document.body.style.overflow = 'hidden';
        dialog.current?.focus();
        return () => {
            [root.style.overflow, document.body.style.overflow] = was;
            before?.focus?.();
        };
    }, []);

    useEffect(() => {
        const key = (e: KeyboardEvent) => {
            // Typing in the film search is not steering the stage.
            if ((e.target as HTMLElement | null)?.closest?.('input, textarea')) return;
            if (e.key === 'Escape') { e.preventDefault(); onClose(); }
            else if (e.key === 'ArrowRight') { e.preventDefault(); move(1); }
            else if (e.key === 'ArrowLeft') { e.preventDefault(); move(-1); }
            else if (e.key === 'Home') { e.preventDefault(); setFocus(0); }
        };
        window.addEventListener('keydown', key);
        return () => window.removeEventListener('keydown', key);
    }, [move, onClose]);

    // Swipe and trackpad. A drag beyond a few pixels is a move, not a click.
    const drag = useRef<{ x: number; moved: boolean } | null>(null);
    const wheel = useRef({ acc: 0, t: 0, lock: 0 });
    const onPointerDown = (e: React.PointerEvent) => { drag.current = { x: e.clientX, moved: false }; };
    const onPointerMove = (e: React.PointerEvent) => {
        if (drag.current && Math.abs(e.clientX - drag.current.x) > 8) drag.current.moved = true;
    };
    const onPointerUp = (e: React.PointerEvent) => {
        const d = drag.current;
        if (d?.moved) move(-Math.round((e.clientX - d.x) / (w * 0.55)) || (e.clientX < d.x ? 1 : -1));
        window.setTimeout(() => { drag.current = null; }, 0);
    };
    // Nothing on the stage scrolls, so a wheel or trackpad gesture over it in
    // either direction steps through the matches instead. The cooldown keeps a
    // trackpad fling to a few cards rather than racing to the end.
    const onWheel = (e: React.WheelEvent) => {
        const delta = Math.abs(e.deltaX) >= Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        const now = performance.now(), wh = wheel.current;
        if (now < wh.lock) return;
        if (now - wh.t > 220) wh.acc = 0;
        wh.t = now;
        wh.acc += delta;
        if (Math.abs(wh.acc) > 60) { move(wh.acc > 0 ? 1 : -1); wh.acc = 0; wh.lock = now + 260; }
    };

    const h = Math.round(w * 1.5);
    const isMatch = focused.rank > 0;
    const source = slots[0].m;
    // Tags often repeat a genre ("action" beside Action); show each idea once.
    const sharedGenres = focused.m.shared_genres ?? [];
    const sharedTags = (focused.m.shared_tags ?? [])
        .filter(t => !sharedGenres.some(g => g.toLowerCase() === t.toLowerCase()));

    return (
        <motion.div
            ref={dialog}
            role="dialog"
            aria-modal="true"
            aria-label={`Matches for ${current.title}`}
            tabIndex={-1}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduce ? 0 : 0.25 }}
            // The spread shadow paints the ultramarine frame around the stage,
            // so the page behind never shows at its edges.
            // It never scrolls as a whole: the header, the stage and the ticks hold
            // still, and only the details beneath scroll when they run long.
            className="fixed inset-1.5 md:inset-4 z-[120] rounded-[28px] bg-black overflow-hidden outline-none
                       flex flex-col shadow-[0_0_0_100vmax_#2B1FBB]"
        >
            {/* The glow the cards stand in. */}
            <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 top-0"
                style={{
                    height: h + 420,
                    background: `radial-gradient(ellipse ${Math.round(w * 2.7)}px ${Math.round(h * 0.72)}px at 50% ${Math.round(110 + h / 2)}px,
                        #d40003 0%, #b80002 34%, rgba(130,0,0,.78) 58%, rgba(45,0,0,.4) 80%, rgba(0,0,0,0) 100%)`,
                }} />

            <header className="relative z-20 shrink-0 flex flex-wrap md:flex-nowrap items-center gap-3 px-4 md:px-6 pt-4">
                <button type="button" onClick={onClose} className="round-btn w-10 h-10" aria-label="Close matches">
                    <X className="w-4 h-4" />
                </button>
                <nav aria-label="Films you have followed" className="min-w-0 flex items-baseline gap-x-2 overflow-x-auto scroller">
                    <span className="text-[13px] text-white/60 shrink-0">Matches for</span>
                    {trail.map((m, i) => (
                        <span key={`${m.item_id}-${i}`} className="flex items-baseline gap-2 shrink-0">
                            {i > 0 && <span aria-hidden="true" className="text-white/35">/</span>}
                            {i === trail.length - 1
                                ? <span aria-current="page" className="text-[15px] font-medium">{short(m.title)}</span>
                                : <button type="button" onClick={() => setTrail(t => t.slice(0, i + 1))}
                                    className="text-[15px] text-white/60 hover:text-white underline-offset-4 hover:underline">
                                    {short(m.title)}
                                </button>}
                        </span>
                    ))}
                </nav>
                <div className="w-full md:w-auto md:ml-auto flex items-center gap-5 md:shrink-0">
                    <p className="hidden xl:block text-[13px] text-white/50 shrink-0">Move with ← →, close with Esc</p>
                    <div className="w-full md:w-[19rem]">
                        <MatchSearch variant="compact" onPick={(m) => setTrail([m])} />
                    </div>
                </div>
            </header>

            {/* The stage. */}
            <div
                className="relative z-0 shrink-0 mt-6 select-none touch-none"
                style={{ height: h + 24 }}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={() => { drag.current = null; }}
                onClickCapture={(e) => { if (drag.current?.moved) { e.stopPropagation(); e.preventDefault(); } }}
                onWheel={onWheel}
            >
                {slots.map(({ m, rank, p }) => {
                    const o = p - focus, d = Math.abs(o);
                    const x = Math.sign(o) * offsetAt(d, w);
                    const hidden = d > 6;
                    return (
                        <motion.button
                            type="button"
                            key={`${current.item_id}-${m.item_id}`}
                            tabIndex={-1}
                            aria-label={rank ? `Match ${rank}: ${m.title}, ${pct(m.score)}%` : `Your pick: ${m.title}`}
                            onClick={() => (o !== 0 ? setFocus(p) : rank ? dive(m) : undefined)}
                            initial={reduce ? false : { x: 0, scale: 0.55, opacity: 0 }}
                            animate={{ x, scale: scaleAt(d), opacity: hidden ? 0 : 1 }}
                            transition={reduce ? { duration: 0 } : {
                                type: 'spring', stiffness: 230, damping: 30,
                                delay: entered ? 0 : 0.08 + Math.abs(p) * 0.04,
                            }}
                            style={{
                                width: w, height: h, left: '50%', top: 12, marginLeft: -w / 2,
                                zIndex: 50 - d, pointerEvents: hidden ? 'none' : undefined,
                            }}
                            className={cn('absolute overflow-hidden rounded-[22px] bg-panel-hi text-left shadow-[0_30px_60px_-20px_rgba(0,0,0,.8)]',
                                o === 0 && rank === 0 && 'ring-2 ring-white ring-offset-0')}
                        >
                            {m.posterPath && <img src={img(m.posterPath, 'w500')} alt="" draggable={false}
                                className="absolute inset-0 w-full h-full object-cover" />}
                            <span aria-hidden="true" className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(0,0,0,.55)_0%,rgba(0,0,0,0)_24%,rgba(0,0,0,0)_52%,rgba(0,0,0,.9)_100%)]" />

                            {rank
                                ? <span className="absolute top-4 left-4 grid place-items-center w-9 h-9 rounded-full bg-black/70 text-[14px] font-medium tabular-nums">{rank}</span>
                                : <span className="absolute top-4 left-4 pill h-8 px-3 bg-white text-black">Your pick</span>}
                            {rank
                                ? <Figure movie={m} kind="match" className="absolute top-3.5 right-4 text-[60px]" />
                                : <Figure movie={m} kind="rating" className="absolute top-3.5 right-4 text-[60px]" />}

                            <span className="absolute inset-x-0 bottom-0 p-5">
                                <span className="block text-[22px] font-medium leading-[1.05] tracking-[-0.02em] line-clamp-2">{short(m.title)}</span>
                                <span className="mt-3 flex gap-1.5">
                                    {m.year && <span className="pill-sm h-7 px-3 text-[12px] border border-white/35">{m.year}</span>}
                                    {m.runtime && <span className="pill-sm h-7 px-3 text-[12px] border border-white/35">{runtime(m.runtime)}</span>}
                                </span>
                            </span>

                            {/* Further from focus, deeper into the red. */}
                            <motion.span aria-hidden="true" className="absolute inset-0 bg-[#1a0000] pointer-events-none"
                                initial={false} animate={{ opacity: DIM[Math.min(d, DIM.length - 1)] }}
                                transition={{ duration: reduce ? 0 : 0.3 }} />
                        </motion.button>
                    );
                })}

                <button type="button" onClick={() => move(-1)} disabled={focus <= lo} aria-label="Previous match"
                    className="round-btn max-sm:hidden w-11 h-11 absolute left-3 md:left-6 top-1/2 -translate-y-1/2 z-[60] bg-black/50 backdrop-blur">
                    <ChevronLeft className="w-5 h-5" />
                </button>
                <button type="button" onClick={() => move(1)} disabled={focus >= hi} aria-label="Next match"
                    className="round-btn max-sm:hidden w-11 h-11 absolute right-3 md:right-6 top-1/2 -translate-y-1/2 z-[60] bg-black/50 backdrop-blur">
                    <ChevronRight className="w-5 h-5" />
                </button>
            </div>

            {/* One tick per film, laid out as they sit on the stage. */}
            <div className="relative z-10 shrink-0 mt-5 flex justify-center items-end gap-[5px] h-6" role="group" aria-label="Jump to a match">
                {Array.from({ length: hi - lo + 1 }, (_, i) => lo + i).map(p => {
                    const s = slots.find(x => x.p === p);
                    return (
                        <button key={p} type="button" onClick={() => setFocus(p)}
                            aria-label={s?.rank ? `Match ${s.rank}` : 'Your pick'}
                            className="group h-6 w-2.5 flex items-end justify-center">
                            <span className={cn('block w-[2px] rounded-full transition-[height,background-color]',
                                p === focus ? 'h-6 bg-white' : p === 0 ? 'h-4 bg-white/70' : 'h-2.5 bg-white/35 group-hover:bg-white/70')} />
                        </button>
                    );
                })}
            </div>

            {/* What the focused card is, and why it is here. */}
            <section ref={details} aria-live="polite"
                // The fade at the foot says there is more below when the details run long.
                className="relative z-10 flex-1 min-h-0 overflow-y-auto overscroll-contain
                           [mask-image:linear-gradient(to_bottom,#000_calc(100%-48px),transparent)]
                           grid lg:grid-cols-[1.15fr_1fr] content-start gap-x-12 gap-y-8 px-5 md:px-10 pt-8 pb-12">
                <div className="min-w-0">
                    <p className="text-[13px] text-white/60">
                        {isMatch ? `Match ${focused.rank} of ${data?.matches.length ?? 24}` : 'Your pick'}
                    </p>
                    <h2 className="mt-2 text-[clamp(32px,4.2vw,60px)] font-medium leading-[0.92] tracking-[-0.04em] text-balance">
                        {focused.m.title}
                    </h2>
                    <div className="mt-5 flex flex-wrap gap-1.5">
                        {focused.m.year && <span className="pill h-8 px-3.5 border border-white/35 normal-case text-[13px]">{focused.m.year}</span>}
                        {focused.m.runtime && <span className="pill h-8 px-3.5 border border-white/35 normal-case text-[13px]">{runtime(focused.m.runtime)}</span>}
                        {focused.m.rating != null && (
                            <span className="pill h-8 px-3.5 border border-white/35 normal-case text-[13px]">
                                {stars(focused.m.rating)} <Star className="w-3 h-3" /> from {focused.m.rating_count?.toLocaleString() ?? 'MovieLens'} ratings
                            </span>
                        )}
                    </div>
                    {focused.m.overview && (
                        <p className="mt-5 max-w-[62ch] text-[15px] leading-[1.55] text-white/75">{focused.m.overview}</p>
                    )}
                    <p className="mt-4 text-[14px] text-white/60 max-w-[62ch]">
                        {focused.m.directors?.length ? <>{focused.m.media_type === 'tv' ? 'Created' : 'Directed'} by <span className="text-white/90">{names(focused.m.directors, 2)}</span>. </> : null}
                        {focused.m.cast?.length ? <>With <span className="text-white/90">{names(focused.m.cast, 4)}</span>.</> : null}
                    </p>
                </div>

                <div className="min-w-0">
                    {error ? (
                        <div className="rounded-[20px] border border-white/15 p-6">
                            <p className="text-[16px]">{error}</p>
                            <p className="mt-1 text-[14px] text-white/60">The recommendation API didn't answer. Check your connection and try again.</p>
                            <button type="button" onClick={() => setAttempt(a => a + 1)} className="pill pill-white mt-4">Try again</button>
                        </div>
                    ) : isMatch ? (
                        <>
                            <div className="flex items-end gap-5">
                                <Figure movie={focused.m} kind="match" className="text-[96px] md:text-[120px]" />
                                <p className="pb-2 text-[14px] leading-snug text-white/70 max-w-[24ch]">
                                    match with <span className="text-white">{short(source.title)}</span>. Scores are relative: the closest film is 100.
                                </p>
                            </div>
                            {(sharedGenres.length || sharedTags.length) ? (
                                <>
                                    <h3 className="mt-7 text-[14px] text-white/60">What the two share</h3>
                                    <div className="mt-3 flex flex-wrap gap-1.5">
                                        {sharedGenres.map(g => <span key={g} className="pill h-8 px-3.5 bg-white text-black normal-case text-[13px]">{cap(g)}</span>)}
                                        {sharedTags.map(t => <span key={t} className="pill h-8 px-3.5 border border-white/35 normal-case text-[13px]">{t}</span>)}
                                    </div>
                                </>
                            ) : (
                                <p className="mt-7 text-[14px] text-white/60 max-w-[44ch]">
                                    No genre or theme in common: this one is here because the same people rated both highly.
                                </p>
                            )}
                        </>
                    ) : (
                        <>
                            <p className="text-[15px] text-white/75 max-w-[46ch] leading-[1.55]">
                                {data
                                    ? <>Its 24 closest films fan out on either side, nearest first. Match strength comes from 32 million ratings, reranked by what the films are about.</>
                                    : 'Finding its 24 closest films…'}
                            </p>
                            {focused.m.tags?.length ? (
                                <>
                                    <h3 className="mt-7 text-[14px] text-white/60">Themes</h3>
                                    <div className="mt-3 flex flex-wrap gap-1.5">
                                        {focused.m.tags.map(t => <span key={t} className="pill h-8 px-3.5 border border-white/35 normal-case text-[13px]">{t}</span>)}
                                    </div>
                                </>
                            ) : null}
                        </>
                    )}

                    <div className="mt-8 flex flex-wrap gap-2">
                        {isMatch && (
                            <button type="button" onClick={() => dive(focused.m)} className="pill pill-ultra h-10">
                                Find its matches <ArrowUpRight className="w-4 h-4" />
                            </button>
                        )}
                        <TasteActions movie={focused.m} taste={taste} />
                    </div>
                </div>
            </section>
        </motion.div>
    );
};

export default MatchStage;
