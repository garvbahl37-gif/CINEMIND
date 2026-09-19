import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const VIEWS = [
  { id: 'home', label: 'Collection' },
  { id: 'top50', label: 'Top 50' },
  { id: 'tv', label: 'Series' },
  { id: 'about', label: 'How it works' },
];

export default function Navbar({
  view, onNavigate, onSearchFocus,
}: { view: string; onNavigate: (v: string) => void; onSearchFocus: () => void }) {
  const [lifted, setLifted] = useState(false);
  useEffect(() => {
    const f = () => setLifted(window.scrollY > 12);
    window.addEventListener('scroll', f, { passive: true });
    return () => window.removeEventListener('scroll', f);
  }, []);

  return (
    <header
      className="fixed inset-x-0 top-0 z-50 transition-colors duration-300"
      style={{
        background: lifted ? 'rgba(5,6,9,.72)' : 'transparent',
        backdropFilter: lifted ? 'blur(22px) saturate(150%)' : 'none',
        WebkitBackdropFilter: lifted ? 'blur(22px) saturate(150%)' : 'none',
        borderBottom: `1px solid ${lifted ? 'rgba(255,255,255,.07)' : 'transparent'}`,
      }}
    >
      <nav className="flex h-16 items-center gap-5 sm:gap-8"
           style={{ paddingInline: 'var(--gut)' }}>
        <motion.button onClick={() => onNavigate('home')} className="shrink-0"
                whileHover={{ scale: 1.04 }} whileTap={{ scale: .96 }}
                style={{ fontFamily: 'var(--serif)', fontSize: '1.55rem', fontWeight: 400,
                         letterSpacing: '-0.02em' }}>
          Cine<span style={{ color: 'var(--lamp)',
                             textShadow: '0 0 18px var(--lamp-glow)' }}>mind</span>
        </motion.button>

        <div className="shelf flex flex-1 items-center gap-5 sm:gap-7">
          {VIEWS.map((v) => (
            <button key={v.id} onClick={() => onNavigate(v.id)}
                    className="relative shrink-0 py-1 text-[0.8rem] transition-colors sm:text-[0.875rem]"
                    style={{ color: view === v.id ? 'var(--halide)' : 'var(--halide-dim)' }}>
              {v.label}
              {view === v.id && (
                <motion.span layoutId="nav-lit" className="absolute inset-x-0 -bottom-1 h-[2px]"
                      style={{ background: 'var(--lamp)', borderRadius: 2,
                               boxShadow: '0 0 12px var(--lamp-glow)' }} />
              )}
            </button>
          ))}
        </div>

        <motion.button onClick={onSearchFocus}
                whileHover={{ scale: 1.04 }} whileTap={{ scale: .96 }}
                className="ml-auto flex shrink-0 items-center gap-2 rounded-full px-4 py-2 text-[.8rem]
                           transition-colors hover:text-[var(--halide)]"
                style={{ border: '1px solid rgba(255,255,255,.12)',
                         background: 'rgba(255,255,255,.04)',
                         backdropFilter: 'blur(10px)', color: 'var(--halide-mid)' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2"><circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" /></svg>
          <span className="hidden sm:inline">Search</span>
          <kbd className="ml-1 hidden rounded px-1.5 py-0.5 text-[.65rem] sm:inline"
               style={{ background: 'rgba(255,255,255,.07)', color: 'var(--halide-dim)' }}>/</kbd>
        </motion.button>
      </nav>
    </header>
  );
}
