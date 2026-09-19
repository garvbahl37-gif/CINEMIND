import { useEffect, useState } from 'react';
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion';
import { backdrop, poster } from '../config';
import Perforation from './Perforation';
import type { Film } from '../types';
import { api } from '../api';
import { clip } from '../lib';

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * The hero is the engine working, not a wordmark: a film, and the films the
 * model actually places next to it, with the match strength it assigned.
 * The still pushes in slowly and drifts on scroll; the copy is struck in one
 * orchestrated sequence rather than a shower of separate fades.
 */
export default function ProximityHero({
  seeds, onSelect,
}: { seeds: Film[]; onSelect: (f: Film) => void }) {
  const [idx, setIdx] = useState(0);
  const [near, setNear] = useState<Film[]>([]);
  const film = seeds[idx];

  const { scrollY } = useScroll();
  const bgY = useTransform(scrollY, [0, 900], [0, 190]);
  const fade = useTransform(scrollY, [0, 620], [1, 0]);

  useEffect(() => {
    if (!film) return;
    const ac = new AbortController();
    setNear([]);
    api.similar(film.item_id, 5, ac.signal)
      .then((d) => setNear(d.similar_items)).catch(() => {});
    return () => ac.abort();
  }, [film?.item_id]);

  if (!film) return null;
  const bd = backdrop(film.backdrop_path, 'original');

  return (
    <section className="relative isolate overflow-hidden pt-28"
             style={{ paddingInline: 'var(--gut)' }}>
      {/* the still */}
      <AnimatePresence mode="wait">
        <motion.div key={film.item_id} className="absolute inset-0 -z-10"
                    style={{ y: bgY, opacity: fade }}
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: .9, ease: EASE }}>
          {bd && <img src={bd} alt="" className="kenburns h-full w-full object-cover"
                      style={{ opacity: .46 }} />}
          <div className="absolute inset-0" style={{ background:
            'linear-gradient(180deg, rgba(8,9,15,.58) 0%, rgba(8,9,15,.50) 34%, rgba(8,9,15,.92) 78%, var(--ink) 100%)' }} />
          <div className="absolute inset-0" style={{ background:
            'linear-gradient(95deg, var(--ink) 4%, rgba(8,9,15,.72) 38%, transparent 76%)' }} />
        </motion.div>
      </AnimatePresence>

      <div className="mx-auto grid max-w-[1220px] items-center gap-x-16 gap-y-10
                      pb-14 lg:grid-cols-[minmax(0,1fr)_396px]">
        <AnimatePresence mode="wait">
          <motion.div key={film.item_id}
                      initial="out" animate="in" exit="out"
                      variants={{ in: { transition: { staggerChildren: .07, delayChildren: .1 } },
                                  out: { transition: { staggerChildren: .02 } } }}>
            {[
              <p className="machine mb-5" key="k">17,719 films · 64 dimensions · 32M ratings</p>,

              <h1 key="t" style={{ fontSize: 'clamp(2.7rem, 7vw, var(--t-4xl))',
                                   textShadow: '0 4px 40px rgba(0,0,0,.7)' }}>
                {film.title}
              </h1>,

              <div className="mt-6 flex flex-wrap items-center gap-2" key="m">
                {film.year && <Chip>{film.year}</Chip>}
                {film.genres.slice(0, 3).map((g) => <Chip key={g}>{g}</Chip>)}
                {film.rating_avg && (
                  <span className="pl-1 text-[.8rem] font-medium" style={{ color: 'var(--lamp-hi)' }}>
                    {film.rating_avg.toFixed(1)}
                    <span style={{ color: 'var(--halide-dim)', fontWeight: 400 }}>
                      {' '}/ 5 · {film.rating_count.toLocaleString()} viewers
                    </span>
                  </span>
                )}
              </div>,

              film.overview ? (
                <p key="o" className="mt-6 max-w-[52ch] text-[.97rem]"
                   style={{ color: 'var(--halide-mid)', lineHeight: 1.75 }}>
                  {clip(film.overview, 215)}
                </p>
              ) : null,

              <div className="mt-9 flex flex-wrap gap-3" key="c">
                <motion.button onClick={() => onSelect(film)}
                  whileHover={{ scale: 1.03 }} whileTap={{ scale: .97 }}
                  transition={{ duration: .2, ease: EASE }}
                  className="px-6 py-3 text-[.875rem] font-semibold"
                  style={{ background: 'linear-gradient(135deg, var(--lamp-warm), var(--lamp))',
                           color: '#1B1405', borderRadius: 'var(--r-sm)',
                           boxShadow: '0 6px 26px var(--lamp-glow)' }}>
                  Explore this film
                </motion.button>
                <motion.button onClick={() => setIdx((i) => (i + 1) % seeds.length)}
                  whileHover={{ scale: 1.03 }} whileTap={{ scale: .97 }}
                  transition={{ duration: .2, ease: EASE }}
                  className="glass px-6 py-3 text-[.875rem]">
                  Next reel
                </motion.button>
              </div>,
            ].filter(Boolean).map((child, i) => (
              <motion.div key={i}
                variants={{ out: { opacity: 0, y: 26, filter: 'blur(6px)' },
                            in: { opacity: 1, y: 0, filter: 'blur(0px)',
                                  transition: { duration: .75, ease: EASE } } }}>
                {child}
              </motion.div>
            ))}
          </motion.div>
        </AnimatePresence>

        {/* what the model puts next to it */}
        <motion.div className="glass p-5"
                    initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: .8, delay: .3, ease: EASE }}>
          <div className="mb-4 flex items-center gap-2">
            <span className="block h-1.5 w-1.5 rounded-full"
                  style={{ background: 'var(--emulsion)',
                           boxShadow: '0 0 10px var(--emulsion)' }} />
            <p className="machine">Nearest in the model's space</p>
          </div>
          <ul className="space-y-2">
            <AnimatePresence mode="popLayout">
              {near.map((f, i) => (
                <motion.li key={f.item_id}
                  initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ delay: .12 * i, duration: .55, ease: EASE }}>
                  <motion.button onClick={() => onSelect(f)}
                    whileHover={{ x: 5 }} whileTap={{ scale: .98 }}
                    className="group flex w-full items-center gap-3 rounded-[10px] p-2 text-left
                               transition-colors hover:bg-white/[.045]">
                    <div className="frame h-[56px] w-[38px] shrink-0">
                      {poster(f.poster_path, 'w185')
                        && <img src={poster(f.poster_path, 'w185')!} alt="" loading="lazy" />}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[.875rem] font-medium transition-colors
                                      group-hover:text-[var(--lamp-hi)]">{f.title}</div>
                      <div className="mt-1.5">
                        <Perforation score={f.score ?? 0}
                          label={f.shared_genres?.length
                            ? f.shared_genres.slice(0, 2).join(' · ') : String(f.year ?? '')} />
                      </div>
                    </div>
                  </motion.button>
                </motion.li>
              ))}
            </AnimatePresence>
            {!near.length && Array.from({ length: 5 }, (_, i) => (
              <li key={i} className="h-[72px] animate-pulse rounded-[10px]"
                  style={{ background: 'rgba(255,255,255,.035)' }} />
            ))}
          </ul>
        </motion.div>
      </div>
    </section>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="px-3 py-1 text-[.75rem]"
          style={{ border: '1px solid rgba(255,255,255,.12)', borderRadius: 999,
                   color: 'var(--halide-mid)', background: 'rgba(255,255,255,.035)',
                   backdropFilter: 'blur(8px)' }}>{children}</span>
  );
}
