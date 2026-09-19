import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { backdrop, poster } from '../config';
import MatchScore from './MatchScore';
import { api } from '../api';
import type { Film } from '../types';

/**
 * The detail view. Its job is to answer "why am I being shown these?" —
 * every recommendation carries its match strength and the genres/tags it
 * actually shares with the film you opened.
 */
export default function FilmSheet({
  film, onClose, onSelect,
}: { film: Film; onClose: () => void; onSelect: (f: Film) => void }) {
  const [near, setNear] = useState<Film[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [full, setFull] = useState<Film>(film);

  useEffect(() => {
    const ac = new AbortController();
    setNear([]); setFull(film);
    api.similar(film.item_id, 12, ac.signal)
      .then((d) => { setNear(d.similar_items); setFull(d.source); setLatency(d.latency_ms); })
      .catch(() => {});
    return () => ac.abort();
  }, [film.item_id]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', k);
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', k); document.body.style.overflow = ''; };
  }, [onClose]);

  const bd = backdrop(full.backdrop_path, 'original');
  const ps = poster(full.poster_path, 'w500');

  return (
    <motion.div className="fixed inset-0 z-[100] overflow-y-auto"
         style={{ background: 'rgba(5,6,9,.82)', backdropFilter: 'blur(10px)' }}
         initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
         transition={{ duration: .35 }}
         onClick={onClose} role="dialog" aria-modal="true" aria-label={full.title}>
      <motion.div className="glass mx-auto my-8 max-w-[1080px] overflow-hidden"
           style={{ boxShadow: 'var(--lift-3)' }}
           initial={{ opacity: 0, y: 40, scale: .97 }}
           animate={{ opacity: 1, y: 0, scale: 1 }}
           exit={{ opacity: 0, y: 24, scale: .98 }}
           transition={{ type: 'spring', damping: 26, stiffness: 240 }}
           onClick={(e) => e.stopPropagation()}>

        <div className="relative">
          {bd && (
            <div className="relative h-[220px] overflow-hidden sm:h-[300px]">
              <img src={bd} alt="" className="kenburns h-full w-full object-cover" style={{ opacity: .55 }} />
              <div className="absolute inset-0" style={{ background:
                'linear-gradient(180deg, rgba(8,9,15,.28) 0%, rgba(8,9,15,.74) 48%, rgba(16,18,28,.99) 96%)' }} />
            </div>
          )}
          <motion.button onClick={onClose} aria-label="Close"
                  whileHover={{ scale: 1.1, rotate: 90 }} whileTap={{ scale: .9 }}
                  transition={{ duration: .25, ease: [0.16, 1, 0.3, 1] }}
                  className="absolute right-4 top-4 grid h-10 w-10 place-items-center rounded-full"
                  style={{ border: '1px solid rgba(255,255,255,.14)',
                           background: 'rgba(5,6,9,.7)', backdropFilter: 'blur(10px)' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="2"><path d="M18 6L6 18M6 6l12 12" /></svg>
          </motion.button>
        </div>

        <div className="px-6 pb-10 sm:px-10" style={{ marginTop: bd ? '-64px' : '2rem' }}>
          <div className="flex flex-col gap-7 sm:flex-row">
            {ps && (
              <div className="frame relative z-10 w-[150px] shrink-0"
                   style={{ aspectRatio: '2/3' }}>
                <img src={ps} alt="" />
              </div>
            )}
            <div className="min-w-0 flex-1" style={{ paddingTop: bd ? '4.5rem' : '0.5rem' }}>
              <h2 style={{ fontSize: 'clamp(1.9rem, 4vw, var(--t-2xl))' }}>{full.title}</h2>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                {full.year && <Chip>{full.year}</Chip>}
                {full.runtime ? <Chip>{full.runtime} min</Chip> : null}
                {full.genres.map((g) => <Chip key={g}>{g}</Chip>)}
              </div>
              {full.rating_avg && (
                <p className="mt-4 text-[0.85rem]" style={{ color: 'var(--halide-mid)' }}>
                  <span style={{ color: 'var(--lamp-hi)', fontWeight: 600 }}>
                    {full.rating_avg.toFixed(2)}
                  </span>{' '}
                  average from {full.rating_count.toLocaleString()} MovieLens viewers
                </p>
              )}
              {full.overview && (
                <p className="mt-4 max-w-[62ch] text-[0.95rem]"
                   style={{ color: 'var(--halide-mid)', lineHeight: 1.7 }}>{full.overview}</p>
              )}
              {full.tags?.length > 0 && (
                <div className="mt-5 flex flex-wrap gap-1.5">
                  {full.tags.map((t) => (
                    <span key={t} className="px-2 py-0.5 text-[0.7rem]"
                          style={{ background: 'rgba(255,255,255,.05)', color: 'var(--halide-mid)',
                                   borderRadius: 999 }}>{t}</span>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="mt-12">
            <div className="mb-5 flex items-baseline justify-between gap-4">
              <h3 style={{ fontSize: 'var(--t-md)' }}>If you liked this</h3>
              {latency !== null && (
                <span className="machine">ranked in {latency.toFixed(1)} ms</span>
              )}
            </div>

            {near.length > 0 ? (
              <ul className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
                {near.map((f) => (
                  <li key={f.item_id}>
                    <button onClick={() => onSelect(f)}
                            className="group flex w-full items-center gap-3.5 rounded-[12px] p-2.5 text-left
                                       transition-colors hover:bg-white/[.05]">
                      <div className="frame h-[66px] w-[44px] shrink-0">
                        {poster(f.poster_path, 'w185')
                          ? <img src={poster(f.poster_path, 'w185')!} alt="" loading="lazy" />
                          : null}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-[0.9rem] font-medium transition-colors
                                        group-hover:text-[var(--lamp-hi)]">{f.title}</div>
                        <div className="mt-0.5 truncate text-[0.75rem]"
                             style={{ color: 'var(--halide-dim)' }}>
                          {f.year}
                          {f.shared_genres?.length
                            ? `  ·  shares ${f.shared_genres.slice(0, 2).join(', ')}`
                            : ''}
                        </div>
                        <div className="mt-1.5"><MatchScore score={f.score ?? 0} /></div>
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2">
                {Array.from({ length: 6 }, (_, i) => (
                  <div key={i} className="h-[76px] animate-pulse"
                       style={{ background: 'rgba(255,255,255,.035)', borderRadius: 12 }} />
                ))}
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="px-3 py-1 text-[.75rem]"
          style={{ border: '1px solid rgba(255,255,255,.12)', borderRadius: 999,
                   color: 'var(--halide-mid)', background: 'rgba(255,255,255,.035)' }}>{children}</span>
  );
}
