import { useEffect, useState } from 'react';
import { backdrop, poster } from '../config';
import Perforation from './Perforation';
import type { Film } from '../types';
import { api } from '../api';

/**
 * The hero is the engine doing its job, not a wordmark.
 * A film sits on the left; its true nearest neighbours are listed on the right
 * with the match strength the model actually assigned. One orchestrated reveal
 * on load — the perforations fill in sequence — and nothing else moves by itself.
 */
export default function ProximityHero({
  seeds, onSelect,
}: { seeds: Film[]; onSelect: (f: Film) => void }) {
  const [idx, setIdx] = useState(0);
  const [near, setNear] = useState<Film[]>([]);
  const [shown, setShown] = useState(0);

  const film = seeds[idx];

  useEffect(() => {
    if (!film) return;
    const ac = new AbortController();
    setShown(0);
    api.similar(film.item_id, 5, ac.signal)
      .then((d) => setNear(d.similar_items))
      .catch(() => {});
    return () => ac.abort();
  }, [film?.item_id]);

  useEffect(() => {
    if (!near.length) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) { setShown(near.length); return; }
    let n = 0;
    const t = setInterval(() => {
      n += 1; setShown(n);
      if (n >= near.length) clearInterval(t);
    }, 110);
    return () => clearInterval(t);
  }, [near]);

  if (!film) return null;
  const bd = backdrop(film.backdrop_path, 'original');

  return (
    <section className="relative pt-28" style={{ paddingInline: 'var(--gut)' }}>
      {bd && (
        <div className="absolute inset-0 -z-10">
          <img src={bd} alt="" className="h-full w-full object-cover opacity-[0.30]" />
          <div className="absolute inset-0"
               style={{ background:
                 'linear-gradient(180deg, rgba(16,21,38,.72) 0%, rgba(16,21,38,.60) 45%, var(--ink) 97%)' }} />
        </div>
      )}

      <div className="mx-auto grid max-w-[1180px] items-center gap-x-14 gap-y-10
                      pb-10 lg:grid-cols-[minmax(0,1fr)_400px]">
        {/* the film */}
        <div>
          <p className="machine mb-5">
            17,719 films embedded in 64 dimensions
          </p>
          <h1 style={{ fontSize: 'clamp(2.6rem, 6.5vw, var(--t-4xl))' }}>
            {film.title}
          </h1>
          <div className="mt-5 flex flex-wrap items-center gap-2">
            {film.year && (
              <span className="border px-2.5 py-1 text-[0.75rem]"
                    style={{ borderColor: 'var(--ink-edge)', color: 'var(--halide-mid)',
                             borderRadius: 'var(--frame)' }}>{film.year}</span>
            )}
            {film.genres.slice(0, 3).map((g) => (
              <span key={g} className="border px-2.5 py-1 text-[0.75rem]"
                    style={{ borderColor: 'var(--ink-edge)', color: 'var(--halide-mid)',
                             borderRadius: 'var(--frame)' }}>{g}</span>
            ))}
            {film.rating_avg && (
              <span className="px-1 text-[0.75rem]" style={{ color: 'var(--lamp)' }}>
                {film.rating_avg.toFixed(1)} from {film.rating_count.toLocaleString()} viewers
              </span>
            )}
          </div>

          {film.overview && (
            <p className="mt-6 max-w-[54ch] text-[0.95rem]"
               style={{ color: 'var(--halide-mid)', lineHeight: 1.7 }}>
              {film.overview.length > 240
                ? `${film.overview.slice(0, 240).trimEnd()}…`
                : film.overview}
            </p>
          )}

          <div className="mt-8 flex flex-wrap gap-3">
            <button onClick={() => onSelect(film)}
                    className="px-5 py-2.5 text-[0.875rem] font-semibold transition-opacity
                               hover:opacity-90"
                    style={{ background: 'var(--lamp)', color: 'var(--ink-deep)',
                             borderRadius: 'var(--frame)' }}>
              Explore this film
            </button>
            <button onClick={() => setIdx((i) => (i + 1) % seeds.length)}
                    className="border px-5 py-2.5 text-[0.875rem] transition-colors
                               hover:border-[var(--halide-mid)]"
                    style={{ borderColor: 'var(--ink-edge)', borderRadius: 'var(--frame)' }}>
              Show another
            </button>
          </div>
        </div>

        {/* what the model puts next to it */}
        <div className="border p-5"
             style={{ borderColor: 'var(--ink-edge)', borderRadius: 'var(--frame)',
                      background: 'rgba(10,14,27,.55)', backdropFilter: 'blur(8px)' }}>
          <p className="machine mb-4">Nearest films in the model's space</p>
          <ul className="space-y-3.5">
            {near.map((f, i) => (
              <li key={f.item_id}
                  style={{ opacity: i < shown ? 1 : 0.12, transition: 'opacity .4s ease' }}>
                <button onClick={() => onSelect(f)}
                        className="group flex w-full items-center gap-3 text-left">
                  <div className="frame h-[54px] w-[36px] shrink-0">
                    {poster(f.poster_path, 'w185')
                      ? <img src={poster(f.poster_path, 'w185')!} alt="" loading="lazy" />
                      : null}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[0.875rem] font-medium transition-colors
                                    group-hover:text-[var(--lamp)]">{f.title}</div>
                    <div className="mt-1">
                      <Perforation score={f.score ?? 0}
                                   label={f.shared_genres?.length
                                     ? f.shared_genres.slice(0, 2).join(', ')
                                     : String(f.year ?? '')} />
                    </div>
                  </div>
                </button>
              </li>
            ))}
            {!near.length && Array.from({ length: 5 }, (_, i) => (
              <li key={i} className="h-[54px] animate-pulse"
                  style={{ background: 'var(--ink-raise)', borderRadius: 'var(--frame)' }} />
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
