import { useRef, useState, useEffect } from 'react';
import FilmCard from './FilmCard';
import type { Film } from '../types';

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

  const nudge = (dir: -1 | 1) =>
    ref.current?.scrollBy({ left: dir * ref.current.clientWidth * 0.8, behavior: 'smooth' });

  if (!films.length) return null;

  return (
    <section className="py-7">
      <div className="mb-4 flex items-baseline justify-between gap-4"
           style={{ paddingInline: 'var(--gut)' }}>
        <h2 style={{ fontSize: 'var(--t-lg)' }}>
          {title}
          {count !== undefined && (
            <span className="ml-3 align-middle text-[0.8rem] font-normal"
                  style={{ fontFamily: 'var(--sans)', color: 'var(--halide-dim)' }}>
              {count} films
            </span>
          )}
        </h2>
        <div className="hidden gap-1.5 md:flex">
          {([-1, 1] as const).map((d) => (
            <button key={d} onClick={() => nudge(d)}
                    disabled={d === -1 ? !edges.left : !edges.right}
                    aria-label={d === -1 ? 'Scroll left' : 'Scroll right'}
                    className="grid h-8 w-8 place-items-center border transition-colors
                               disabled:opacity-25"
                    style={{ borderColor: 'var(--ink-edge)', borderRadius: 'var(--frame)' }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2.2">
                <path d={d === -1 ? 'M15 18l-6-6 6-6' : 'M9 18l6-6-6-6'} />
              </svg>
            </button>
          ))}
        </div>
      </div>
      <div ref={ref} onScroll={measure} className="shelf flex gap-4"
           style={{ paddingInline: 'var(--gut)' }}>
        {films.map((f) => <FilmCard key={f.item_id} film={f} onSelect={onSelect} width={176} />)}
      </div>
    </section>
  );
}
