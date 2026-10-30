import { useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { MARK } from './Marks';

interface Props {
    /** Films in the catalogue, once known. */
    films?: number;
    /** The catalogue has answered (or failed): the gate may open. */
    ready: boolean;
    /** Called once the gate is fully open and the loader can unmount. */
    onDone: () => void;
}

/** Long enough for the mark to assemble once; never longer than the data takes past it. */
const MIN_MS = 700;
const EASE = [0.76, 0, 0.18, 1] as const;

/**
 * The opening: the CINEMIND mark assembles on the ultramarine — the centre
 * card rises, its two matches slide out beside it — while twenty-five ticks — a film and its twenty-four matches — run under the
 * wordmark until the catalogue has loaded. Then the black of the app opens
 * out from the centre like a projector gate, to exactly where the shell sits,
 * and the dashboard is already behind it.
 */
const Loader = ({ ready, onDone, films }: Props) => {
    const reduce = useReducedMotion();
    const [minDone, setMinDone] = useState(false);
    const [phase, setPhase] = useState<'loading' | 'opening' | 'fading'>('loading');
    const [tick, setTick] = useState(0);
    const inset = typeof window !== 'undefined' && window.innerWidth >= 768 ? 16 : 6;

    useEffect(() => {
        const t = window.setTimeout(() => setMinDone(true), reduce ? 300 : MIN_MS);
        return () => window.clearTimeout(t);
    }, [reduce]);

    // The ticks run outwards from the centre, the way matches sit on the stage.
    useEffect(() => {
        if (phase !== 'loading' || reduce) return;
        const t = window.setInterval(() => setTick(n => n + 1), 70);
        return () => window.clearInterval(t);
    }, [phase, reduce]);

    useEffect(() => {
        if (ready && minDone && phase === 'loading') setPhase(reduce ? 'fading' : 'opening');
    }, [ready, minDone, phase, reduce]);

    const lit = (i: number) => {
        if (phase !== 'loading' || reduce) return true;
        const d = Math.abs(i - 12);                 // distance from the centre tick
        return d <= tick % 16;
    };

    return (
        <motion.div
            role="status"
            aria-live="polite"
            aria-label={ready ? 'Catalogue loaded' : 'Loading the catalogue'}
            className="fixed inset-0 z-[200] bg-ultra overflow-hidden"
            initial={{ opacity: 1 }}
            animate={{ opacity: phase === 'fading' ? 0 : 1 }}
            transition={{ duration: reduce ? 0.2 : 0.25, ease: 'easeOut' }}
            onAnimationComplete={() => { if (phase === 'fading') onDone(); }}
        >
            {/* The gate: the app's black, opening from a sliver to the shell's own edges. */}
            <motion.div
                aria-hidden="true"
                className="absolute inset-0 bg-ink"
                initial={{ clipPath: 'inset(50% 50% 50% 50% round 28px)' }}
                animate={{
                    clipPath: phase === 'loading'
                        ? 'inset(50% 50% 50% 50% round 28px)'
                        : `inset(${inset}px ${inset}px ${inset}px ${inset}px round 28px)`,
                }}
                // Waits for the wordmark to clear before it opens.
                transition={{ duration: reduce ? 0 : 0.55, delay: reduce ? 0 : 0.12, ease: EASE }}
                onAnimationComplete={() => { if (phase === 'opening') setPhase('fading'); }}
            />

            <motion.div
                className="absolute inset-0 grid place-items-center px-6"
                animate={{ opacity: phase === 'loading' ? 1 : 0, scale: phase === 'loading' ? 1 : 0.96 }}
                transition={{ duration: reduce ? 0 : 0.2, ease: 'easeIn' }}
            >
                <div className="flex flex-col items-center text-center">
                    {/* The mark assembles the way the stage does: the side cards slide out from behind the centre. */}
                    <svg viewBox="0 0 24 24" aria-hidden="true" className="w-24 h-24 md:w-28 md:h-28 text-white" fill="currentColor">
                        {([MARK.left, MARK.right] as number[]).map((x, i) => (
                            <motion.rect key={x} y={MARK.side.y} width={MARK.side.w} height={MARK.side.h} rx={MARK.side.rx}
                                initial={reduce ? false : { x: 9, opacity: 0 }}
                                animate={{ x, opacity: 0.45 }}
                                transition={{ delay: 0.22 + i * 0.05, type: 'spring', stiffness: 320, damping: 26 }} />
                        ))}
                        <motion.rect x={MARK.centre.x} y={MARK.centre.y} width={MARK.centre.w} height={MARK.centre.h} rx={MARK.centre.rx}
                            style={{ originX: '50%', originY: '50%' }}
                            initial={reduce ? false : { scaleY: 0, opacity: 0 }}
                            animate={{ scaleY: 1, opacity: 1 }}
                            transition={{ delay: 0.05, duration: 0.35, ease: EASE }} />
                    </svg>

                    <h1 className="mt-8 overflow-hidden display text-[clamp(56px,13vw,168px)] leading-[0.82]">
                        <motion.span className="block"
                            initial={reduce ? false : { y: '102%' }}
                            animate={{ y: 0 }}
                            transition={{ delay: 0.12, duration: 0.55, ease: EASE }}>
                            Cinemind
                        </motion.span>
                    </h1>

                    <div className="mt-9 flex items-end gap-[6px] h-5" aria-hidden="true">
                        {Array.from({ length: 25 }, (_, i) => (
                            <span key={i} className={i === 12 ? 'w-[2px] h-5 bg-white' : 'w-[2px] rounded-full transition-[height,background-color] duration-150'}
                                style={i === 12 ? undefined : {
                                    height: lit(i) ? 12 : 6,
                                    backgroundColor: lit(i) ? 'rgba(255,255,255,.9)' : 'rgba(255,255,255,.3)',
                                }} />
                        ))}
                    </div>

                    <motion.p className="mt-5 text-[14px] text-white/75 tabular-nums"
                        initial={reduce ? false : { opacity: 0 }} animate={{ opacity: 1 }}
                        transition={{ delay: 0.35, duration: 0.35 }}>
                        {ready && films ? `${films.toLocaleString()} films, 24 matches each` : 'Loading the catalogue'}
                    </motion.p>
                </div>
            </motion.div>
        </motion.div>
    );
};

export default Loader;
