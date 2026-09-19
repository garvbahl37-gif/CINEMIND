import { useEffect, useState } from 'react';

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
        background: lifted ? 'rgba(10,14,27,.92)' : 'transparent',
        backdropFilter: lifted ? 'blur(14px)' : 'none',
        borderBottom: `1px solid ${lifted ? 'var(--ink-edge)' : 'transparent'}`,
      }}
    >
      <nav className="flex h-16 items-center gap-5 sm:gap-8"
           style={{ paddingInline: 'var(--gut)' }}>
        <button onClick={() => onNavigate('home')} className="shrink-0"
                style={{ fontFamily: 'var(--serif)', fontSize: '1.35rem', fontWeight: 900,
                         letterSpacing: '-0.02em' }}>
          Cine<span style={{ color: 'var(--lamp)' }}>mind</span>
        </button>

        <div className="shelf flex flex-1 items-center gap-5 sm:gap-7">
          {VIEWS.map((v) => (
            <button key={v.id} onClick={() => onNavigate(v.id)}
                    className="relative shrink-0 py-1 text-[0.8rem] transition-colors sm:text-[0.875rem]"
                    style={{ color: view === v.id ? 'var(--halide)' : 'var(--halide-dim)' }}>
              {v.label}
              {view === v.id && (
                <span className="absolute inset-x-0 -bottom-0.5 h-px"
                      style={{ background: 'var(--lamp)' }} />
              )}
            </button>
          ))}
        </div>

        <button onClick={onSearchFocus}
                className="ml-auto flex shrink-0 items-center gap-2 border px-3 py-1.5 text-[0.8rem]
                           transition-colors hover:border-[var(--lamp)]"
                style={{ borderColor: 'var(--ink-edge)', borderRadius: 'var(--frame)',
                         color: 'var(--halide-mid)' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2"><circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" /></svg>
          Search
        </button>
      </nav>
    </header>
  );
}
