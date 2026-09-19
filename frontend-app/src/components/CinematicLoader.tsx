import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * An Academy leader countdown — the strip of film spliced before a reel so the
 * projectionist can find the frame. Rotating sweep hand, crosshair registration
 * marks, numbers ticking down, gate flicker and grain. It ends with an iris
 * wipe, the oldest transition in cinema.
 *
 * `ready` reports whether the catalogue has arrived. The countdown always runs
 * its full length so the reveal lands on a beat rather than stuttering out.
 */
export default function CinematicLoader({ ready, onDone }: { ready: boolean; onDone: () => void }) {
  const [count, setCount] = useState(3);
  const [phase, setPhase] = useState<'count' | 'iris' | 'gone'>('count');
  const done = useRef(onDone);
  done.current = onDone;

  const reduced = typeof window !== 'undefined'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Beat the countdown down. It parks on 1 until the catalogue lands, so the
  // reveal always falls on a beat instead of stuttering out mid-number.
  useEffect(() => {
    if (reduced || phase !== 'count') return;
    const t = setInterval(() => {
      setCount((c) => (c <= 1 ? 1 : c - 1));
    }, 620);
    return () => clearInterval(t);
  }, [reduced, phase]);

  // Once we are parked on 1 and the data has arrived, run the iris wipe.
  useEffect(() => {
    if (phase !== 'count' || !ready) return;
    if (!reduced && count > 1) return;
    const hold = setTimeout(() => setPhase('iris'), reduced ? 0 : 520);
    return () => clearTimeout(hold);
  }, [phase, ready, count, reduced]);

  // The wipe owns its own effect, so nothing else can cancel its timer.
  useEffect(() => {
    if (phase !== 'iris') return;
    const t = setTimeout(() => { setPhase('gone'); done.current(); }, reduced ? 0 : 880);
    return () => clearTimeout(t);
  }, [phase, reduced]);

  return (
    <AnimatePresence>
      {phase !== 'gone' && (
        <motion.div
          className="fixed inset-0 z-[9999] grid place-items-center overflow-hidden"
          style={{ background: 'var(--ink-deep)' }}
          initial={{ opacity: 1 }}
          animate={{
            opacity: 1,
            // projector gate flicker
            filter: ['brightness(1)', 'brightness(1.1)', 'brightness(.94)', 'brightness(1)'],
          }}
          transition={{ filter: { duration: .28, repeat: Infinity, repeatType: 'mirror' } }}
          exit={{ opacity: 0, transition: { duration: .5 } }}
        >
          {/* iris wipe: a hole opens from the centre and swallows the screen */}
          <motion.div
            className="absolute inset-0 z-20"
            style={{ background: 'var(--ink-deep)' }}
            animate={phase === 'iris'
              ? { clipPath: ['circle(75% at 50% 50%)', 'circle(0% at 50% 50%)'] }
              : { clipPath: 'circle(75% at 50% 50%)' }}
            transition={{ duration: .85, ease: [0.16, 1, 0.3, 1] }}
          />

          {/* leader artwork */}
          <div className="relative z-10 grid place-items-center">
            <svg viewBox="0 0 400 400" className="h-[min(74vw,340px)] w-[min(74vw,340px)]">
              {/* registration rings */}
              {[190, 150, 104].map((r, i) => (
                <circle key={r} cx="200" cy="200" r={r} fill="none"
                        stroke="var(--halide-dim)" strokeWidth={i === 0 ? 1.5 : 1}
                        opacity={i === 0 ? .55 : .3} />
              ))}
              {/* crosshair */}
              <path d="M200 4 V396 M4 200 H396" stroke="var(--halide-dim)"
                    strokeWidth="1" opacity=".3" />
              {/* quadrant ticks */}
              {Array.from({ length: 12 }, (_, i) => {
                const a = (i * 30 * Math.PI) / 180;
                return (
                  <line key={i}
                        x1={200 + Math.cos(a) * 190} y1={200 + Math.sin(a) * 190}
                        x2={200 + Math.cos(a) * 172} y2={200 + Math.sin(a) * 172}
                        stroke="var(--lamp)" strokeWidth="2" opacity=".5" />
                );
              })}
              {/* the sweep hand, one full revolution per number */}
              <motion.line
                x1="200" y1="200" x2="200" y2="14"
                stroke="var(--lamp)" strokeWidth="3" strokeLinecap="round"
                style={{ transformOrigin: '200px 200px', filter: 'drop-shadow(0 0 8px var(--lamp-glow))' }}
                animate={{ rotate: 360 }}
                transition={{ duration: .62, ease: 'linear', repeat: Infinity }}
              />
              {/* the trail the sweep leaves behind it */}
              <defs>
                <radialGradient id="cm-sweep" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="var(--lamp)" stopOpacity=".26" />
                  <stop offset="100%" stopColor="var(--lamp)" stopOpacity="0" />
                </radialGradient>
              </defs>
              <motion.path
                d="M200 200 L200 14 A186 186 0 0 0 33 97 Z"
                fill="url(#cm-sweep)"
                style={{ transformOrigin: '200px 200px' }}
                animate={{ rotate: 360 }}
                transition={{ duration: .62, ease: 'linear', repeat: Infinity }}
              />
              {/* the beat drains this ring, so waiting reads as progress */}
              <motion.circle
                cx="200" cy="200" r="150" fill="none"
                stroke="var(--lamp)" strokeWidth="2.5" strokeLinecap="round"
                pathLength={1} strokeDasharray="1 1"
                style={{ transformOrigin: '200px 200px', rotate: -90,
                         filter: 'drop-shadow(0 0 6px var(--lamp-glow))' }}
                animate={{ strokeDashoffset: [1, 0] }}
                transition={{ duration: .62, ease: 'linear', repeat: Infinity }}
              />
              <circle cx="200" cy="200" r="5" fill="var(--lamp)" />
            </svg>

            {/* the number, re-struck each beat */}
            <div className="pointer-events-none absolute grid place-items-center">
              <AnimatePresence mode="popLayout">
                <motion.span
                  key={count}
                  initial={{ scale: 1.5, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: .8, opacity: 0 }}
                  transition={{ duration: .3, ease: [0.16, 1, 0.3, 1] }}
                  style={{
                    fontFamily: 'var(--serif)', fontWeight: 400,
                    fontSize: 'min(34vw, 176px)', lineHeight: 1,
                    color: 'var(--halide)',
                    textShadow: '0 0 50px rgba(232,181,75,.4), 0 0 14px rgba(0,0,0,.6)',
                    fontVariantNumeric: 'lining-nums',
                  }}
                >
                  {Math.max(count, 1)}
                </motion.span>
              </AnimatePresence>
            </div>
          </div>

          <div className="absolute bottom-[14%] left-0 right-0 text-center">
            <motion.p
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: .25, duration: .6 }}
              style={{ fontFamily: 'var(--serif)', fontSize: '1.7rem', fontWeight: 400,
                       letterSpacing: '-.02em' }}
            >
              Cine<span style={{ color: 'var(--lamp)' }}>mind</span>
            </motion.p>
            <motion.p className="machine mt-2"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: .5 }}>
              {ready ? 'Threading the projector' : 'Loading 17,719 films'}
            </motion.p>
          </div>

          {/* frame edge markers, as on a leader */}
          <div className="pointer-events-none absolute inset-6 border"
               style={{ borderColor: 'rgba(255,255,255,.06)', borderRadius: 4 }} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
