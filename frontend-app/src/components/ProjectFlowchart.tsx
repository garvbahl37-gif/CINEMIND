import { motion, useInView } from 'framer-motion';
import { Database, Blocks, Network, Layers, ArrowDownWideNarrow, Zap } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../config';

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * The recommendation pipeline, running.
 *
 * The stages reflow into a grid rather than scrolling sideways: the previous
 * version kept all six on one rail and auto-scrolled it, which clipped the
 * first and last node at every width below xl. Progress now reads off a single
 * bar above the grid, so nothing has to fit on one line to make sense.
 */

const STAGES = [
    { icon: Database, label: '32M Ratings', detail: 'MovieLens, 200k viewers' },
    { icon: Blocks, label: 'Co-occurrence', detail: 'item x item, damped' },
    { icon: Network, label: 'Two-Tower', detail: '64-d embeddings' },
    { icon: Layers, label: 'Candidates', detail: 'union of both signals' },
    { icon: ArrowDownWideNarrow, label: 'Rerank', detail: 'genre, tags, confidence' },
    { icon: Zap, label: 'Serve', detail: 'precomputed lookup' },
];

const STEP_MS = 1400;

const ProjectFlowchart = () => {
    const [active, setActive] = useState(0);
    const [latency, setLatency] = useState<number | null>(null);
    const hostRef = useRef<HTMLDivElement>(null);
    const inView = useInView(hostRef, { margin: '-10%' });

    const reduced = typeof window !== 'undefined'
        && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    useEffect(() => {
        if (!inView || reduced) { if (reduced) setActive(STAGES.length - 1); return; }
        const t = setInterval(() => setActive(p => (p + 1) % STAGES.length), STEP_MS);
        return () => clearInterval(t);
    }, [inView, reduced]);

    // Show what the API actually costs rather than a hardcoded figure.
    useEffect(() => {
        let alive = true;
        const t0 = performance.now();
        fetch(`${API_BASE}/api/similar/0?k=6`)
            .then(r => r.json())
            .then(() => { if (alive) setLatency(Math.round(performance.now() - t0)); })
            .catch(() => { });
        return () => { alive = false; };
    }, []);

    return (
        <div ref={hostRef}
            className="w-full bg-black/40 border border-white/10 rounded-3xl p-5 sm:p-7 md:p-8
                       backdrop-blur-sm relative overflow-hidden">
            <div className="absolute inset-0 opacity-[0.07] pointer-events-none"
                style={{
                    backgroundImage:
                        'linear-gradient(rgba(255,255,255,.14) 1px, transparent 1px),' +
                        'linear-gradient(90deg, rgba(255,255,255,.14) 1px, transparent 1px)',
                    backgroundSize: '44px 44px',
                }} />

            <div className="relative z-10">
                <div className="flex items-center justify-between gap-4 mb-5">
                    <h3 className="text-base md:text-lg font-bold text-white/85">
                        Recommendation pipeline
                    </h3>
                    <div className="flex items-center gap-2 text-[11px] text-primary font-bold
                                    bg-primary/10 px-3 py-1.5 rounded-full border border-primary/25">
                        <span className="relative flex h-1.5 w-1.5">
                            <span className="animate-ping absolute inline-flex h-full w-full
                                             rounded-full bg-primary opacity-75" />
                            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-primary" />
                        </span>
                        Live
                    </div>
                </div>

                {/* Six segments, one per stage: the handoff that the connectors used to show. */}
                <div className="flex gap-1.5 mb-6" aria-hidden>
                    {STAGES.map((s, i) => (
                        <div key={s.label}
                            className="relative h-[3px] flex-1 rounded-full overflow-hidden bg-white/[.07]">
                            <motion.div
                                className="absolute inset-y-0 left-0 rounded-full bg-primary"
                                animate={{ width: active > i ? '100%' : '0%' }}
                                transition={{ duration: .4, ease: EASE }}
                            />
                            {active === i && !reduced && (
                                <motion.div
                                    className="absolute inset-y-0 left-0 rounded-full bg-primary"
                                    initial={{ width: '0%' }}
                                    animate={{ width: '100%' }}
                                    transition={{ duration: STEP_MS / 1000, ease: 'linear' }}
                                />
                            )}
                        </div>
                    ))}
                </div>

                <ol className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-2.5 sm:gap-3">
                    {STAGES.map((s, i) => {
                        const isActive = active === i;
                        const done = active > i;
                        const Icon = s.icon;
                        return (
                            <motion.li
                                key={s.label}
                                className="relative rounded-2xl border px-3 py-4 sm:py-5 flex flex-col
                                           items-center gap-3 text-center min-w-0"
                                animate={{
                                    borderColor: isActive ? 'rgb(220 38 38)'
                                        : done ? 'rgba(220,38,38,.34)' : 'rgba(255,255,255,.10)',
                                    backgroundColor: isActive ? 'rgba(220,38,38,.12)'
                                        : done ? 'rgba(220,38,38,.05)' : 'rgba(23,23,23,.45)',
                                    boxShadow: isActive
                                        ? '0 0 26px rgba(220,38,38,.28)' : '0 0 0 rgba(0,0,0,0)',
                                }}
                                transition={{ duration: .45, ease: EASE }}
                            >
                                <motion.div
                                    className="p-2.5 sm:p-3 rounded-xl shrink-0"
                                    animate={{
                                        backgroundColor: isActive ? 'rgb(220 38 38)'
                                            : done ? 'rgba(220,38,38,.35)' : 'rgba(255,255,255,.06)',
                                        color: isActive || done ? '#fff' : 'rgb(156 163 175)',
                                    }}
                                    transition={{ duration: .45, ease: EASE }}
                                >
                                    <Icon size={20} />
                                </motion.div>

                                <div className="min-w-0 w-full">
                                    <h4 className={`text-[13px] sm:text-sm font-bold mb-0.5 transition-colors
                                                    ${isActive ? 'text-white' : 'text-gray-300'}`}>
                                        {s.label}
                                    </h4>
                                    <p className="text-[10.5px] text-gray-400 leading-snug">
                                        {s.detail}
                                    </p>
                                </div>

                                <span className="absolute top-2 left-2.5 text-[10px] font-mono
                                                 text-white/25 tabular-nums">
                                    {String(i + 1).padStart(2, '0')}
                                </span>
                            </motion.li>
                        );
                    })}
                </ol>
            </div>

            <div className="mt-6 flex flex-wrap justify-between items-center gap-3 text-[11px]
                            text-gray-500 border-t border-white/5 pt-4">
                <span>17,719 films indexed</span>
                <span>
                    {latency === null
                        ? 'measuring round trip…'
                        : <>live round trip <span className="text-primary font-bold">{latency} ms</span></>}
                </span>
            </div>
        </div>
    );
};

export default ProjectFlowchart;
