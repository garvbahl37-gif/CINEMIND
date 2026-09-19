import { useEffect, useRef, useState } from 'react';
import { motion, useInView } from 'framer-motion';

const EASE = [0.16, 1, 0.3, 1] as const;

const STAGES = [
  { key: 'ratings',  label: 'Ratings',    sub: '32,000,204',  detail: 'MovieLens' },
  { key: 'cooc',     label: 'Co-occur',   sub: 'item × item',  detail: 'damped' },
  { key: 'tower',    label: 'Two-tower',  sub: '64 dimensions', detail: 'InfoNCE' },
  { key: 'rerank',   label: 'Rerank',     sub: 'genre + tags', detail: 'IDF' },
  { key: 'serve',    label: 'Serve',      sub: '~1 ms',        detail: 'precomputed' },
];

/**
 * The recommendation pipeline, running. A pulse of light travels the rail and
 * lights each stage as it passes, then loops — the shape of the system in one
 * glance. It only animates while on screen, and holds still for anyone who has
 * asked for reduced motion.
 */
export default function Pipeline() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { margin: '-15%' });
  const [active, setActive] = useState(-1);
  const reduced = typeof window !== 'undefined'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  useEffect(() => {
    if (!inView) return;
    if (reduced) { setActive(STAGES.length - 1); return; }
    let i = 0;
    setActive(0);
    const t = setInterval(() => {
      i = (i + 1) % (STAGES.length + 1);
      setActive(i === STAGES.length ? -1 : i);
    }, 900);
    return () => clearInterval(t);
  }, [inView, reduced]);

  return (
    <div ref={ref} className="relative my-14">
      {/* the rail */}
      <div className="relative mx-auto hidden h-[2px] w-full md:block"
           style={{ background: 'rgba(255,255,255,.09)' }}>
        {!reduced && inView && (
          <motion.div
            className="absolute inset-y-0 w-28"
            style={{ background:
              'linear-gradient(90deg, transparent, var(--lamp), transparent)',
              filter: 'drop-shadow(0 0 8px var(--lamp))' }}
            animate={{ left: ['-7rem', '100%'] }}
            transition={{ duration: 4.5, ease: 'linear', repeat: Infinity }}
          />
        )}
      </div>

      <ol className="mt-0 grid gap-4 md:-mt-[9px] md:grid-cols-5">
        {STAGES.map((s, i) => {
          const on = active === i;
          return (
            <li key={s.key} className="relative flex gap-4 md:block">
              {/* node on the rail */}
              <div className="flex shrink-0 justify-start md:justify-center">
                <motion.span
                  className="block rounded-full"
                  animate={{
                    width: on ? 18 : 12, height: on ? 18 : 12,
                    background: on ? 'var(--lamp)' : 'var(--ink-lift)',
                    boxShadow: on
                      ? '0 0 0 4px rgba(232,53,74,.2), 0 0 24px var(--lamp)'
                      : '0 0 0 1px rgba(255,255,255,.12)',
                  }}
                  transition={{ duration: .45, ease: EASE }}
                />
              </div>

              <motion.div className="pb-2 md:mt-5 md:text-center"
                animate={{ opacity: on ? 1 : .52, y: on ? -2 : 0 }}
                transition={{ duration: .45, ease: EASE }}>
                <div className="text-[.9rem] font-semibold">{s.label}</div>
                <div className="machine mt-1" style={{ letterSpacing: '.02em' }}>{s.sub}</div>
                <div className="mt-0.5 text-[.72rem]" style={{ color: 'var(--halide-dim)' }}>
                  {s.detail}
                </div>
              </motion.div>

              {/* vertical connector on narrow screens */}
              {i < STAGES.length - 1 && (
                <span className="absolute left-[5px] top-5 h-full w-px md:hidden"
                      style={{ background: 'rgba(255,255,255,.09)' }} />
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
