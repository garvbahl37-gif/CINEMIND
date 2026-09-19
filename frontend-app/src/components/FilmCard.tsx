import { useState } from 'react';
import { motion } from 'framer-motion';
import { poster } from '../config';
import type { Film } from '../types';

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * A film as a lit frame: the poster carries the weight, the chrome stays quiet
 * until you reach for it, then the frame blooms in lamp gold.
 */
export default function FilmCard({
  film, onSelect, width, index = 0,
}: { film: Film; onSelect: (f: Film) => void; width?: number; index?: number }) {
  const [broken, setBroken] = useState(false);
  const src = poster(film.poster_path, 'w342');

  return (
    <motion.button
      onClick={() => onSelect(film)}
      className="group text-left"
      style={width ? { width, flexShrink: 0 } : { width: '100%' }}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: .6, ease: EASE, delay: Math.min(index, 8) * .045 }}
      whileHover={{ y: -8 }}
      whileTap={{ scale: .97 }}
      aria-label={`${film.title}${film.year ? `, ${film.year}` : ''}`}
    >
      <div className="frame transition-shadow duration-300 group-hover:shadow-[var(--bloom)]"
           style={{ aspectRatio: '2 / 3' }}>
        {src && !broken ? (
          <img src={src} alt="" loading="lazy" decoding="async"
               onError={() => setBroken(true)}
               className="transition-transform duration-[600ms] group-hover:scale-[1.07]"
               style={{ transitionTimingFunction: 'cubic-bezier(.16,1,.3,1)' }} />
        ) : (
          <div className="flex h-full items-center justify-center p-3 text-center"
               style={{ fontFamily: 'var(--serif)', fontSize: 15, color: 'var(--halide-mid)' }}>
            {film.title}
          </div>
        )}

        {/* light wash from the bottom, so the title always has something to sit on */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-1/2 opacity-0
                        transition-opacity duration-300 group-hover:opacity-100"
             style={{ background: 'linear-gradient(to top, rgba(5,6,9,.94), transparent)' }} />

        {film.rating_avg && (
          <div className="pointer-events-none absolute bottom-2.5 left-2.5 flex items-center gap-1
                          opacity-0 transition-opacity duration-300 group-hover:opacity-100">
            <span className="text-[.78rem] font-semibold" style={{ color: 'var(--lamp-hi)' }}>
              {film.rating_avg.toFixed(1)}
            </span>
            <span className="text-[.68rem]" style={{ color: 'var(--halide-dim)' }}>
              / 5
            </span>
          </div>
        )}
      </div>

      <div className="mt-3 leading-tight">
        <div className="truncate text-[.9rem] font-medium transition-colors
                        group-hover:text-[var(--lamp-hi)]">
          {film.title}
        </div>
        <div className="mt-1 text-[.75rem]" style={{ color: 'var(--halide-dim)' }}>
          {film.year ?? '—'}
        </div>
      </div>
    </motion.button>
  );
}
