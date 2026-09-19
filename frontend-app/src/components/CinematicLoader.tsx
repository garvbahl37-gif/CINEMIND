import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * An Academy leader — the strip of film spliced before a reel so the
 * projectionist can find the frame. Sweep hand, draining beat ring,
 * registration crosshair, gate flicker, numbers ticking down.
 *
 * It ends on a true iris wipe: the leader is clipped away from the centre
 * outward, revealing the page beneath it. Nothing cross-fades, so the two
 * layers are never visible on top of each other.
 */
export default function CinematicLoader({
  ready, onDone,
}: { ready: boolean; onDone: () => void }) {
  const [count, setCount] = useState(3);
  const [phase, setPhase] = useState<'count' | 'iris' | 'gone'>('count');
  const done = useRef(onDone);
  done.current = onDone;

  const reduced = typeof window !== 'undefined'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Beat down. Parks on 1 until the catalogue lands, so the reveal always
  // falls on a beat rather than stuttering out mid-number.
  useEffect(() => {
    if (reduced || phase !== 'count') return;
    const t = setInterval(() => setCount((c) => (c <= 1 ? 1 : c - 1)), 640);
    return () => clearInterval(t);
  }, [reduced, phase]);

  useEffect(() => {
    if (phase !== 'count' || !ready) return;
    if (!reduced && count > 1) return;
    const hold = setTimeout(() => setPhase('iris'), reduced ? 0 : 540);
    return () => clearTimeout(hold);
  }, [phase, ready, count, reduced]);

  // The wipe owns its own effect so nothing else can cancel its timer.
  useEffect(() => {
    if (phase !== 'iris') return;
    const t = setTimeout(() => { setPhase('gone'); done.current(); }, reduced ? 0 : 1000);
    return () => clearTimeout(t);
  }, [phase, reduced]);

  const irising = phase === 'iris';

  return (
    <AnimatePresence>
      {phase !== 'gone' && (
        <motion.div
          className="fixed inset-0 z-[9999] grid place-items-center overflow-hidden"
          style={{ background: 'var(--ink-deep)', willChange: 'clip-path' }}
          initial={{ clipPath: 'circle(140% at 50% 50%)' }}
          animate={{ clipPath: irising ? 'circle(0% at 50% 50%)' : 'circle(140% at 50% 50%)' }}
          transition={{ duration: reduced ? 0 : .95, ease: EASE }}
        >
          {/* gate flicker — the lamp never sits perfectly still */}
          <motion.div className="absolute inset-0"
            style={{ background: 'var(--ink-deep)' }}
            animate={{ opacity: [1, .97, 1, .99, 1] }}
            transition={{ duration: .5, repeat: Infinity }} />

          {/* leader artwork, struck out just before the iris closes */}
          <motion.div className="relative z-10 grid place-items-center"
            animate={{ opacity: irising ? 0 : 1, scale: irising ? 1.08 : 1 }}
            transition={{ duration: .35, ease: EASE }}>
            <svg viewBox="0 0 400 400" className="h-[min(74vw,340px)] w-[min(74vw,340px)]">
              <defs>
                <radialGradient id="cm-sweep" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="var(--lamp)" stopOpacity=".30" />
                  <stop offset="100%" stopColor="var(--lamp)" stopOpacity="0" />
                </radialGradient>
              </defs>

              {[190, 150, 104].map((r, i) => (
                <circle key={r} cx="200" cy="200" r={r} fill="none"
                        stroke="var(--halide-dim)" strokeWidth={i === 0 ? 1.5 : 1}
                        opacity={i === 0 ? .5 : .26} />
              ))}
              <path d="M200 4 V396 M4 200 H396" stroke="var(--halide-dim)"
                    strokeWidth="1" opacity=".26" />
              {Array.from({ length: 12 }, (_, i) => {
                const a = (i * 30 * Math.PI) / 180;
                return (
                  <line key={i}
                        x1={200 + Math.cos(a) * 190} y1={200 + Math.sin(a) * 190}
                        x2={200 + Math.cos(a) * 172} y2={200 + Math.sin(a) * 172}
                        stroke="var(--lamp)" strokeWidth="2" opacity=".55" />
                );
              })}

              <motion.path d="M200 200 L200 14 A186 186 0 0 0 33 97 Z" fill="url(#cm-sweep)"
                style={{ transformOrigin: '200px 200px' }}
                animate={{ rotate: 360 }}
                transition={{ duration: .64, ease: 'linear', repeat: Infinity }} />

              <motion.line x1="200" y1="200" x2="200" y2="14"
                stroke="var(--lamp)" strokeWidth="3" strokeLinecap="round"
                style={{ transformOrigin: '200px 200px',
                         filter: 'drop-shadow(0 0 10px var(--lamp))' }}
                animate={{ rotate: 360 }}
                transition={{ duration: .64, ease: 'linear', repeat: Infinity }} />

              <motion.circle cx="200" cy="200" r="150" fill="none"
                stroke="var(--lamp)" strokeWidth="2.5" strokeLinecap="round"
                pathLength={1} strokeDasharray="1 1"
                style={{ transformOrigin: '200px 200px', rotate: -90,
                         filter: 'drop-shadow(0 0 7px var(--lamp-glow))' }}
                animate={{ strokeDashoffset: [1, 0] }}
                transition={{ duration: .64, ease: 'linear', repeat: Infinity }} />

              <circle cx="200" cy="200" r="5" fill="var(--lamp)" />
            </svg>

            <div className="pointer-events-none absolute grid place-items-center">
              <AnimatePresence mode="popLayout">
                <motion.span key={count}
                  initial={{ scale: 1.45, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: .82, opacity: 0 }}
                  transition={{ duration: .3, ease: EASE }}
                  style={{ fontFamily: 'var(--display)', fontWeight: 800,
                           fontSize: 'min(30vw, 158px)', lineHeight: 1,
                           letterSpacing: '-.04em', color: 'var(--halide)',
                           textShadow: '0 0 60px rgba(232,53,74,.5), 0 0 16px rgba(0,0,0,.8)' }}>
                  {Math.max(count, 1)}
                </motion.span>
              </AnimatePresence>
            </div>
          </motion.div>

          <motion.div className="absolute bottom-[13%] left-0 right-0 text-center"
            animate={{ opacity: irising ? 0 : 1 }} transition={{ duration: .3 }}>
            <p style={{ fontFamily: 'var(--display)', fontSize: '1.45rem', fontWeight: 800,
                        letterSpacing: '-.03em' }}>
              CINE<span style={{ color: 'var(--lamp-hi)' }}>MIND</span>
            </p>
            <p className="machine mt-2">
              {ready ? 'Threading the projector' : 'Loading 17,719 films'}
            </p>
          </motion.div>

          <motion.div className="pointer-events-none absolute inset-6"
            style={{ border: '1px solid rgba(255,255,255,.06)', borderRadius: 4 }}
            animate={{ opacity: irising ? 0 : 1 }} transition={{ duration: .3 }} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
