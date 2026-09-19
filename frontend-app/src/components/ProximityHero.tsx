import { useEffect, useState } from 'react';
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion';
import { backdrop, poster } from '../config';
import MatchScore from './MatchScore';
import HeroSearch from './HeroSearch';
import type { Film } from '../types';
import { api } from '../api';
import { clip } from '../lib';

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * The hero is the engine working, not a wordmark: a film, its poster as a
 * physical object, and the films the model actually places next to it with the
 * match strength it assigned.
 *
 * The still pushes in slowly and drifts on scroll; the year sits behind the
 * title as a huge ghosted numeral to give the type something to stand on.
 */
export default function ProximityHero({
  seeds, onSelect, onSearch,
}: { seeds: Film[]; onSelect: (f: Film) => void; onSearch: (q: string) => void }) {
  const [idx, setIdx] = useState(0);
  const [near, setNear] = useState<Film[]>([]);
  const film = seeds[idx];

  const { scrollY } = useScroll();
  const bgY = useTransform(scrollY, [0, 900], [0, 200]);
  const fade = useTransform(scrollY, [0, 640], [1, 0]);

  useEffect(() => {
    if (!film) return;
    const ac = new AbortController();
    setNear([]);
    api.similar(film.item_id, 4, ac.signal)
      .then((d) => setNear(d.similar_items)).catch(() => {});
    return () => ac.abort();
  }, [film?.item_id]);

  if (!film) return null;
  const bd = backdrop(film.backdrop_path, 'original');
  const ps = poster(film.poster_path, 'w500');

  return (
    <section className="relative z-20 overflow-visible pt-24">
      {/* the still */}
      <AnimatePresence mode="wait">
        <motion.div key={film.item_id} className="absolute inset-0 -z-10 overflow-hidden"
                    style={{ y: bgY, opacity: fade }}
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 1, ease: EASE }}>
          {bd && <img src={bd} alt="" className="kenburns h-full w-full object-cover"
                      style={{ opacity: .58 }} />}
          <div className="absolute inset-0" style={{ background:
            'linear-gradient(180deg, rgba(10,7,9,.72) 0%, rgba(10,7,9,.42) 30%, rgba(10,7,9,.90) 76%, var(--ink) 100%)' }} />
          <div className="absolute inset-0" style={{ background:
            'linear-gradient(100deg, var(--ink) 2%, rgba(10,7,9,.86) 34%, rgba(10,7,9,.18) 70%, transparent 100%)' }} />
          {/* scarlet wash from the lamp side */}
          <div className="absolute inset-0" style={{ background:
            'radial-gradient(ellipse 60% 80% at 8% 40%, rgba(232,53,74,.16), transparent 70%)' }} />
        </motion.div>
      </AnimatePresence>

      {/* letterbox bar — the frame this is projected into */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px"
           style={{ background: 'linear-gradient(90deg, transparent, rgba(232,53,74,.35), transparent)' }} />

      <div className="mx-auto grid max-w-[1320px] items-start gap-x-14 gap-y-12
                      pb-16 lg:grid-cols-[minmax(0,1.15fr)_minmax(320px,.85fr)]"
           style={{ paddingInline: 'var(--gut)' }}>

        {/* ---- the film ---- */}
        <div className="relative">
          {/* the year, set enormous behind the title */}
          <AnimatePresence mode="wait">
            {film.year && (
              <motion.span key={film.item_id} aria-hidden="true"
                className="pointer-events-none absolute -left-2 -top-10 select-none"
                initial={{ opacity: 0, scale: .94 }}
                animate={{ opacity: .045, scale: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 1.1, ease: EASE }}
                style={{ fontFamily: 'var(--display)', fontWeight: 900, fontStretch: '125%',
                         fontSize: 'clamp(9rem, 19vw, 17rem)', lineHeight: .78,
                         letterSpacing: '-.06em', color: 'var(--halide)' }}>
                {film.year}
              </motion.span>
            )}
          </AnimatePresence>

          <AnimatePresence mode="wait">
            <motion.div key={film.item_id} className="relative"
                        initial="out" animate="in" exit="out"
                        variants={{ in: { transition: { staggerChildren: .075, delayChildren: .12 } },
                                    out: { transition: { staggerChildren: .02 } } }}>
              {[
                <div className="mb-6 flex items-center gap-4" key="k">
                  <span className="flex items-center gap-2">
                    <motion.span className="block h-2 w-2 rounded-full"
                      style={{ background: 'var(--lamp)' }}
                      animate={{ opacity: [1, .35, 1], scale: [1, .86, 1] }}
                      transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }} />
                    <span className="text-[.7rem] font-bold tracking-[.22em]"
                          style={{ color: 'var(--lamp-hi)' }}>NOW SHOWING</span>
                  </span>
                  <span className="h-px flex-1 max-w-[120px]"
                        style={{ background: 'rgba(255,255,255,.14)' }} />
                  <span className="text-[.7rem] font-semibold tabular-nums"
                        style={{ color: 'var(--halide-dim)' }}>
                    {String(idx + 1).padStart(2, '0')} / {String(seeds.length).padStart(2, '0')}
                  </span>
                </div>,

                <h1 key="t" style={{
                  fontSize: 'clamp(2.9rem, 7.6vw, 6.6rem)',
                  textShadow: '0 6px 50px rgba(0,0,0,.85)',
                }}>
                  {film.title}
                </h1>,

                <div className="mt-7 flex flex-wrap items-center gap-x-5 gap-y-3" key="m">
                  {film.rating_avg && (
                    <span className="flex items-baseline gap-1.5">
                      <span style={{ fontFamily: 'var(--display)', fontWeight: 800,
                                     fontStretch: '112%', fontSize: '1.9rem', lineHeight: 1,
                                     letterSpacing: '-.04em', color: 'var(--lamp-hi)',
                                     textShadow: '0 0 26px var(--lamp-glow)' }}>
                        {film.rating_avg.toFixed(1)}
                      </span>
                      <span className="text-[.78rem]" style={{ color: 'var(--halide-dim)' }}>
                        / 5 · {film.rating_count.toLocaleString()} viewers
                      </span>
                    </span>
                  )}
                  <span className="hidden h-6 w-px sm:block"
                        style={{ background: 'rgba(255,255,255,.14)' }} />
                  <span className="flex flex-wrap gap-2">
                    {film.genres.slice(0, 3).map((g) => (
                      <span key={g} className="px-3 py-1 text-[.72rem] font-medium"
                            style={{ border: '1px solid rgba(255,255,255,.14)', borderRadius: 999,
                                     color: 'var(--halide-mid)',
                                     background: 'rgba(255,255,255,.04)' }}>{g}</span>
                    ))}
                  </span>
                </div>,

                film.overview ? (
                  <p key="o" className="mt-6 max-w-[50ch] text-[.97rem]"
                     style={{ color: 'var(--halide-mid)', lineHeight: 1.78 }}>
                    {clip(film.overview, 190)}
                  </p>
                ) : null,

                <div className="mt-8" key="s">
                  <HeroSearch onSelect={onSelect} onSubmit={onSearch} />
                </div>,

                <div className="mt-7 flex flex-wrap items-center gap-3" key="c">
                  <motion.button onClick={() => onSelect(film)}
                    whileHover={{ scale: 1.035 }} whileTap={{ scale: .965 }}
                    transition={{ duration: .2, ease: EASE }}
                    className="px-7 py-3.5 text-[.875rem] font-bold"
                    style={{ background: 'linear-gradient(135deg, var(--lamp-warm), var(--lamp))',
                             color: '#2A0209', borderRadius: 999,
                             boxShadow: '0 8px 34px var(--lamp-glow)' }}>
                    Explore this film
                  </motion.button>
                  <motion.button onClick={() => setIdx((i) => (i + 1) % seeds.length)}
                    whileHover={{ scale: 1.035 }} whileTap={{ scale: .965 }}
                    transition={{ duration: .2, ease: EASE }}
                    className="flex items-center gap-2 px-6 py-3.5 text-[.875rem] font-medium"
                    style={{ border: '1px solid rgba(255,255,255,.14)', borderRadius: 999,
                             background: 'rgba(20,15,18,.6)',
                             backdropFilter: 'blur(18px)' }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                      <path d="M4 12a8 8 0 1 1 2.3 5.7" /><path d="M4 20v-5h5" />
                    </svg>
                    Next reel
                  </motion.button>
                </div>,
              ].filter(Boolean).map((child, i, all) => (
                <motion.div key={i}
                  style={{ position: 'relative', zIndex: all.length - i }}
                  variants={{ out: { opacity: 0, y: 30, filter: 'blur(7px)' },
                              in: { opacity: 1, y: 0, filter: 'blur(0px)',
                                    transition: { duration: .8, ease: EASE } } }}>
                  {child}
                </motion.div>
              ))}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* ---- the poster, and what the model puts beside it ---- */}
        <div className="flex flex-col gap-7">
          <AnimatePresence mode="wait">
            <motion.button key={film.item_id}
              onClick={() => onSelect(film)}
              className="group relative mx-auto w-[min(76vw,286px)] lg:mx-0"
              initial={{ opacity: 0, y: 34, rotateY: -14 }}
              animate={{ opacity: 1, y: 0, rotateY: 0 }}
              exit={{ opacity: 0, y: -18, rotateY: 10 }}
              transition={{ duration: .9, ease: EASE }}
              whileHover={{ y: -8, rotateZ: -1.2 }}
              style={{ perspective: 1000 }}
              aria-label={`Open ${film.title}`}>
              <div className="frame"
                   style={{ aspectRatio: '2/3',
                            boxShadow: '0 30px 80px rgba(0,0,0,.8), 0 0 0 1px rgba(255,255,255,.09), 0 0 60px rgba(232,53,74,.14)' }}>
                {ps && <img src={ps} alt="" className="transition-transform duration-[700ms]
                                                       group-hover:scale-[1.05]" />}
                <div className="pointer-events-none absolute inset-0"
                     style={{ background:
                       'linear-gradient(150deg, rgba(255,255,255,.14) 0%, transparent 42%)' }} />
              </div>
              {/* the sprocket strip down the edge, as on a print */}
              <div className="pointer-events-none absolute -left-[13px] top-3 hidden flex-col
                              gap-2.5 lg:flex">
                {Array.from({ length: 9 }, (_, i) => (
                  <span key={i} className="block h-[9px] w-[6px] rounded-[1.5px]"
                        style={{ background: 'rgba(255,255,255,.10)' }} />
                ))}
              </div>
            </motion.button>
          </AnimatePresence>

          <motion.div className="glass p-4"
                      initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: .8, delay: .35, ease: EASE }}>
            <div className="mb-3 flex items-center gap-2">
              <span className="block h-1.5 w-1.5 rounded-full"
                    style={{ background: 'var(--emulsion)',
                             boxShadow: '0 0 10px var(--emulsion)' }} />
              <p className="machine">Nearest in the model's space</p>
            </div>
            <ul className="space-y-1">
              <AnimatePresence mode="popLayout">
                {near.map((f, i) => (
                  <motion.li key={f.item_id}
                    initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ delay: .1 * i, duration: .5, ease: EASE }}>
                    <motion.button onClick={() => onSelect(f)}
                      whileHover={{ x: 4 }} whileTap={{ scale: .98 }}
                      className="group flex w-full items-center gap-3 rounded-[10px] p-2 text-left
                                 transition-colors hover:bg-white/[.05]">
                      <div className="frame h-[52px] w-[35px] shrink-0">
                        {poster(f.poster_path, 'w185')
                          && <img src={poster(f.poster_path, 'w185')!} alt="" loading="lazy" />}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-[.85rem] font-medium transition-colors
                                        group-hover:text-[var(--lamp-hi)]">{f.title}</div>
                        <div className="mt-1">
                          <MatchScore score={f.score ?? 0}
                            note={f.shared_genres?.length
                              ? f.shared_genres.slice(0, 2).join(', ') : String(f.year ?? '')} />
                        </div>
                      </div>
                    </motion.button>
                  </motion.li>
                ))}
              </AnimatePresence>
              {!near.length && Array.from({ length: 4 }, (_, i) => (
                <li key={i} className="h-[68px] animate-pulse rounded-[10px]"
                    style={{ background: 'rgba(255,255,255,.035)' }} />
              ))}
            </ul>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
