import { useState } from 'react';
import { poster } from '../config';
import type { Film } from '../types';

/**
 * A film as a frame in a strip. Rectangular, hairline-bordered, no card shadow —
 * the poster is the content, the chrome stays out of its way.
 */
export default function FilmCard({
  film, onSelect, width,
}: { film: Film; onSelect: (f: Film) => void; width?: number }) {
  const [broken, setBroken] = useState(false);
  const src = poster(film.poster_path, 'w342');

  return (
    <button
      onClick={() => onSelect(film)}
      className="group text-left"
      style={width ? { width, flexShrink: 0 } : { width: '100%' }}
      aria-label={`${film.title}${film.year ? `, ${film.year}` : ''}`}
    >
      <div className="frame transition-transform duration-200 group-hover:-translate-y-1"
           style={{ aspectRatio: '2 / 3' }}>
        {src && !broken ? (
          <img src={src} alt="" loading="lazy" decoding="async"
               onError={() => setBroken(true)} />
        ) : (
          <div className="flex h-full items-center justify-center p-3 text-center"
               style={{ fontFamily: 'var(--serif)', fontSize: 15, color: 'var(--halide-mid)' }}>
            {film.title}
          </div>
        )}
        <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity
                        duration-200 group-hover:opacity-100"
             style={{ boxShadow: 'inset 0 0 0 1px var(--lamp)' }} />
      </div>
      <div className="mt-2.5 leading-tight">
        <div className="truncate text-[0.9rem] font-medium transition-colors
                        group-hover:text-[var(--lamp)]">
          {film.title}
        </div>
        <div className="mt-0.5 text-[0.75rem]" style={{ color: 'var(--halide-dim)' }}>
          {film.year ?? '—'}
        </div>
      </div>
    </button>
  );
}
