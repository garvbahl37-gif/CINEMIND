import { useRef, useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import FilmCard from './FilmCard';
import type { Film } from '../types';

const EASE = [0.16, 1, 0.3, 1] as const;

export default function Shelf({
  title, films, onSelect, count,
}: { title: string; films: Film[]; onSelect: (f: Film) => void; count?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ left: false, right: true });

  const measure = () => {
    const el = ref.current;
    if (!el) return;
    setEdges({
      left: el.scrollLeft > 8,
      right: el.scrollLeft + el.clientWidth < el.scrollWidth - 8,
    });
  };
  useEffect(measure, [films]);

  const nudge = (d: -1 | 1) =>
    ref.current?.scrollBy({ left: d * ref.current.clientWidth * .8, behavior: 'smooth' });

  if (!films.length) return null;

  return (
    <section className="relative py-8">
      <motion.div className="mb-5 flex items-baseline justify-between gap-4"
        style={{ paddingInline: 'var(--gut)' }}
        initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-80px' }} transition={{ duration: .6, ease: EASE }}>
        <h2 className="flex items-baseline gap-3" style={{ fontSize: 'var(--t-lg)' }}>
          {title}
          {count !== undefined && (
            <span className="text-[.75rem] font-normal"
                  style={{ fontFamily: 'var(--sans)', color: 'var(--halide-dim)' }}>
              {count}
            </span>
          )}
        </h2>
        <div className="hidden gap-2 md:flex">
          {([-1, 1] as const).map((d) => (
            <motion.button key={d} onClick={() => nudge(d)}
              disabled={d === -1 ? !edges.left : !edges.right}
              whileHover={{ scale: 1.08 }} whileTap={{ scale: .92 }}
              aria-label={d === -1 ? 'Scroll left' : 'Scroll right'}
              className="grid h-9 w-9 place-items-center rounded-full transition-colors
                         disabled:opacity-20"
              style={{ border: '1px solid rgba(255,255,255,.12)',
                       background: 'rgba(255,255,255,.04)' }}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                <path d={d === -1 ? 'M15 18l-6-6 6-6' : 'M9 18l6-6-6-6'} />
              </svg>
            </motion.button>
          ))}
        </div>
      </motion.div>

      <div className="relative">
        <div ref={ref} onScroll={measure} className="shelf flex gap-5"
             style={{ paddingInline: 'var(--gut)' }}>
          {films.map((f, i) => (
            <FilmCard key={f.item_id} film={f} onSelect={onSelect} width={182} index={i} />
          ))}
        </div>
        {/* the strip runs off the edge of the frame rather than stopping dead */}
        <div className="pointer-events-none absolute inset-y-0 right-0 w-24"
             style={{ background: 'linear-gradient(to left, var(--ink), transparent)' }} />
      </div>
    </section>
  );
}
