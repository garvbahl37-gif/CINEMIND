import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const VIEWS = [
  { id: 'home', label: 'Collection' },
  { id: 'top50', label: 'Top 50' },
  { id: 'tv', label: 'Series' },
  { id: 'about', label: 'How it works' },
];

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * A floating glass rail rather than a full-width bar: the wordmark sits left,
 * the sections centre on the page, and the lit pill behind the active section
 * slides between them as one shared element.
 */
export default function Navbar({
  view, onNavigate, onSearchFocus,
}: { view: string; onNavigate: (v: string) => void; onSearchFocus: () => void }) {
  const [lifted, setLifted] = useState(false);

  useEffect(() => {
    const f = () => setLifted(window.scrollY > 12);
    f();
    window.addEventListener('scroll', f, { passive: true });
    return () => window.removeEventListener('scroll', f);
  }, []);

  return (
    <motion.header
      className="fixed inset-x-0 top-0 z-50"
      initial={{ y: -28, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: .7, delay: .15, ease: EASE }}
    >
      <nav className="mx-auto flex h-20 max-w-[1320px] items-center gap-4"
           style={{ paddingInline: 'var(--gut)' }}>

        <motion.button onClick={() => onNavigate('home')}
          whileHover={{ scale: 1.04 }} whileTap={{ scale: .96 }}
          className="shrink-0"
          style={{ fontFamily: 'var(--serif)', fontSize: '1.6rem', fontWeight: 400,
                   letterSpacing: '-.02em' }}>
          Cine<span style={{ color: 'var(--lamp-hi)',
                             textShadow: '0 0 22px var(--lamp-glow)' }}>mind</span>
        </motion.button>

        {/* centred rail — absolutely placed so it stays centred on the page,
            not on whatever space is left over beside the wordmark */}
        <div className="pointer-events-none absolute left-1/2 hidden -translate-x-1/2 md:block">
          <motion.div
            className="pointer-events-auto flex items-center gap-1 rounded-full p-1.5"
            animate={{
              background: lifted ? 'rgba(20,15,18,.72)' : 'rgba(20,15,18,.42)',
              borderColor: lifted ? 'rgba(255,255,255,.10)' : 'rgba(255,255,255,.06)',
            }}
            transition={{ duration: .4 }}
            style={{
              border: '1px solid rgba(255,255,255,.08)',
              backdropFilter: 'blur(22px) saturate(160%)',
              WebkitBackdropFilter: 'blur(22px) saturate(160%)',
              boxShadow: 'var(--lift-2), inset 0 1px 0 rgba(255,255,255,.07)',
            }}
          >
            {VIEWS.map((v) => (
              <button key={v.id} onClick={() => onNavigate(v.id)}
                      className="relative rounded-full px-4 py-2 text-[.82rem] transition-colors"
                      style={{ color: view === v.id ? 'var(--halide)' : 'var(--halide-mid)' }}>
                {view === v.id && (
                  <motion.span layoutId="nav-lit" className="absolute inset-0 rounded-full"
                    transition={{ type: 'spring', damping: 30, stiffness: 320 }}
                    style={{
                      background: 'linear-gradient(140deg, rgba(232,53,74,.26), rgba(232,53,74,.10))',
                      border: '1px solid rgba(232,53,74,.34)',
                      boxShadow: '0 4px 18px var(--lamp-glow)',
                    }} />
                )}
                <span className="relative z-10">{v.label}</span>
              </button>
            ))}
          </motion.div>
        </div>

        {/* mobile: the same sections, scrollable */}
        <div className="shelf flex flex-1 items-center gap-1 md:hidden">
          {VIEWS.map((v) => (
            <button key={v.id} onClick={() => onNavigate(v.id)}
                    className="relative shrink-0 rounded-full px-3 py-1.5 text-[.78rem]"
                    style={{ color: view === v.id ? 'var(--halide)' : 'var(--halide-dim)',
                             background: view === v.id ? 'rgba(232,53,74,.18)' : 'transparent',
                             border: `1px solid ${view === v.id
                               ? 'rgba(232,53,74,.34)' : 'transparent'}` }}>
              {v.label}
            </button>
          ))}
        </div>

        <motion.button onClick={onSearchFocus}
          whileHover={{ scale: 1.04 }} whileTap={{ scale: .96 }}
          aria-label="Search films"
          className="ml-auto flex shrink-0 items-center gap-2 rounded-full px-4 py-2.5 text-[.82rem]
                     transition-colors hover:text-[var(--halide)]"
          style={{ border: '1px solid rgba(255,255,255,.10)',
                   background: 'rgba(20,15,18,.6)',
                   backdropFilter: 'blur(22px) saturate(160%)',
                   WebkitBackdropFilter: 'blur(22px) saturate(160%)',
                   boxShadow: 'var(--lift-1), inset 0 1px 0 rgba(255,255,255,.06)',
                   color: 'var(--halide-mid)' }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round">
            <circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" />
          </svg>
          <span className="hidden sm:inline">Search</span>
          <kbd className="ml-0.5 hidden rounded px-1.5 py-0.5 text-[.65rem] sm:inline"
               style={{ background: 'rgba(255,255,255,.08)', color: 'var(--halide-dim)' }}>/</kbd>
        </motion.button>
      </nav>
    </motion.header>
  );
}
