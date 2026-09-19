import { useState } from 'react';
import { motion } from 'framer-motion';
import { poster } from '../config';
import type { Film } from '../types';

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * A recommendation as a poster with its match struck across the artwork.
 * The rank badge earns its place here — these really are ordered, strongest
 * first — and the strength of the match is the one thing worth reading at
 * a glance, so it gets the weight.
 */
export default function RecoCard({
  film, rank, onSelect, index,
}: { film: Film; rank: number; onSelect: (f: Film) => void; index: number }) {
  const [broken, setBroken] = useState(false);
  const src = poster(film.poster_path, 'w342');
  const pct = Math.round(Math.min(Math.max(film.score ?? 0, 0), 1) * 100);
  const strong = pct >= 90;

  return (
    <motion.button
      onClick={() => onSelect(film)}
      className="group w-full text-left"
      initial={{ opacity: 0, y: 22 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: .5, ease: EASE, delay: Math.min(index, 10) * .05 }}
      whileHover={{ y: -6 }}
      whileTap={{ scale: .97 }}
      aria-label={`${film.title}, ${pct} percent match`}
    >
      <div className="frame transition-shadow duration-300 group-hover:shadow-[var(--bloom)]"
           style={{ aspectRatio: '2 / 3' }}>
        {src && !broken ? (
          <img src={src} alt="" loading="lazy" decoding="async"
               onError={() => setBroken(true)}
               className="transition-transform duration-[600ms] group-hover:scale-[1.08]"
               style={{ transitionTimingFunction: 'cubic-bezier(.16,1,.3,1)' }} />
        ) : (
          <div className="flex h-full items-center justify-center p-3 text-center text-[.85rem]"
               style={{ color: 'var(--halide-mid)' }}>{film.title}</div>
        )}

        {/* rank, top-left */}
        <span className="absolute left-0 top-0 grid h-7 w-7 place-items-center text-[.7rem]
                         font-bold tabular-nums"
              style={{ background: 'rgba(6,4,5,.82)', color: 'var(--halide-mid)',
                       borderBottomRightRadius: 10, backdropFilter: 'blur(8px)' }}>
          {rank}
        </span>

        {/* the match, struck across the foot of the poster */}
        <div className="absolute inset-x-0 bottom-0 px-2.5 pb-2.5 pt-12"
             style={{ background:
               'linear-gradient(to top, rgba(6,4,5,.985) 0%, rgba(6,4,5,.93) 42%, rgba(6,4,5,.6) 72%, transparent 100%)' }}>
          <div className="flex items-baseline gap-1.5">
            <span className="tabular-nums"
                  style={{ fontFamily: 'var(--display)', fontWeight: 800,
                           fontStretch: '112%', letterSpacing: '-.04em',
                           fontSize: '1.45rem', lineHeight: 1,
                           color: strong ? 'var(--lamp-hi)' : 'var(--halide)',
                           textShadow: strong ? '0 0 20px var(--lamp-glow)' : 'none' }}>
              {pct}
            </span>
            <span className="text-[.7rem] font-medium"
                  style={{ color: strong ? 'var(--lamp-hi)' : 'var(--halide-mid)' }}>%</span>
            <span className="ml-auto text-[.65rem]" style={{ color: 'var(--halide-dim)' }}>
              match
            </span>
          </div>
          {/* strength as a hairline rule under the number */}
          <div className="mt-1.5 h-[3px] w-full overflow-hidden rounded-full"
               style={{ background: 'rgba(255,255,255,.14)' }}>
            <motion.div className="h-full rounded-full"
              initial={{ width: 0 }} animate={{ width: `${pct}%` }}
              transition={{ duration: .8, ease: EASE, delay: .2 + Math.min(index, 10) * .05 }}
              style={{ background: strong
                ? 'linear-gradient(90deg, var(--lamp-deep), var(--lamp-hi))'
                : 'rgba(255,255,255,.4)' }} />
          </div>
        </div>
      </div>

      <div className="mt-3">
        <div className="truncate text-[.9rem] font-semibold transition-colors
                        group-hover:text-[var(--lamp-hi)]">{film.title}</div>
        <div className="mt-1 truncate text-[.72rem]" style={{ color: 'var(--halide-dim)' }}>
          {film.year}
          {film.shared_genres?.length
            ? `  ·  ${film.shared_genres.slice(0, 2).join(', ')}` : ''}
        </div>
      </div>
    </motion.button>
  );
}
