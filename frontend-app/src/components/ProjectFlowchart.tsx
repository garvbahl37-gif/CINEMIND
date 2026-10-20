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
        <div ref={hostRef} className="relative">
            {/* One segment per stage: the handoff from each to the next. */}
            <div className="flex gap-1.5 mb-4" aria-hidden>
                {STAGES.map((s, i) => (
                    <div key={s.label} className="relative h-[3px] flex-1 rounded-full overflow-hidden bg-white/[.1]">
                        <motion.div
                            className="absolute inset-y-0 left-0 rounded-full bg-white"
                            animate={{ width: active > i ? '100%' : '0%' }}
                            transition={{ duration: .4, ease: EASE }}
                        />
                        {active === i && !reduced && (
                            <motion.div
                                className="absolute inset-y-0 left-0 rounded-full bg-white"
                                initial={{ width: '0%' }}
                                animate={{ width: '100%' }}
                                transition={{ duration: STEP_MS / 1000, ease: 'linear' }}
                            />
                        )}
                    </div>
                ))}
            </div>

            <ol className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-2">
                {STAGES.map((s, i) => {
                    const isActive = active === i;
                    const done = active > i;
                    const Icon = s.icon;
                    return (
                        <li key={s.label}
                            className={`relative rounded-[18px] p-4 min-h-[148px] flex flex-col transition-colors duration-300
                                ${isActive ? 'bg-white text-black' : done ? 'bg-ultra text-white' : 'bg-panel-hi text-white'}`}>
                            <div className="flex items-start justify-between">
                                <span className="numeral text-[34px]">{i + 1}</span>
                                <Icon size={18} className={isActive ? 'text-black/70' : 'text-white/60'} />
                            </div>
                            <h4 className="mt-auto text-[16px] font-medium leading-tight">{s.label}</h4>
                            <p className={`mt-1 text-[12.5px] leading-snug ${isActive ? 'text-black/65' : 'text-white/60'}`}>
                                {s.detail}
                            </p>
                        </li>
                    );
                })}
            </ol>

            <div className="mt-4 flex flex-wrap justify-between items-center gap-3 text-[13px] text-white/55">
                <span>Every ranking is computed offline, so a request is an array lookup rather than a search.</span>
                <span className="pill h-8 px-3.5 border border-white/25 normal-case text-[13px] text-white tabular-nums">
                    {latency === null ? 'Measuring a live request…' : <>Live request: {latency} ms round trip</>}
                </span>
            </div>
        </div>
    );
};

export default ProjectFlowchart;
