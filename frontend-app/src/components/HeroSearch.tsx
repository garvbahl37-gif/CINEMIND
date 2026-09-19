import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { poster } from '../config';
import { api } from '../api';
import type { Film } from '../types';

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * The search bar people actually land on. Suggestions appear as they type —
 * debounced, arrow-key navigable, and closed by Escape or a click outside.
 */
export default function HeroSearch({
  onSelect, onSubmit,
}: { onSelect: (f: Film) => void; onSubmit: (q: string) => void }) {
  const [q, setQ] = useState('');
  const [hits, setHits] = useState<Film[]>([]);
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (q.trim().length < 2) { setHits([]); setOpen(false); return; }
    const ac = new AbortController();
    const t = setTimeout(() => {
      api.suggest(q, ac.signal)
        .then((d) => { setHits(d.results); setCursor(0); setOpen(d.results.length > 0); })
        .catch(() => {});
    }, 140);
    return () => { clearTimeout(t); ac.abort(); };
  }, [q]);

  useEffect(() => {
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', away);
    return () => document.removeEventListener('mousedown', away);
  }, []);

  const keys = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') { setOpen(false); return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); setOpen(true); setCursor((c) => Math.min(c + 1, hits.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
    else if (e.key === 'Enter') {
      if (open && hits[cursor]) { onSelect(hits[cursor]); setOpen(false); setQ(''); }
      else if (q.trim()) { onSubmit(q.trim()); setOpen(false); }
    }
  };

  return (
    <div ref={box} className="relative z-30 w-full max-w-[520px]">
      <div className="relative">
        <svg className="pointer-events-none absolute left-5 top-1/2 -translate-y-1/2"
             width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             strokeWidth="2" strokeLinecap="round" style={{ color: 'var(--halide-dim)' }}>
          <circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" />
        </svg>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={keys}
          onFocus={() => hits.length && setOpen(true)}
          placeholder="Search a title, a director, a decade…"
          aria-label="Search films"
          aria-expanded={open}
          role="combobox"
          aria-controls="hero-suggestions"
          className="w-full bg-transparent py-4 pl-[3.25rem] pr-4 text-[.95rem] outline-none
                     placeholder:text-[var(--halide-dim)]"
          style={{
            border: '1px solid rgba(255,255,255,.12)',
            background: 'rgba(20,15,18,.78)',
            backdropFilter: 'blur(22px) saturate(160%)',
            WebkitBackdropFilter: 'blur(22px) saturate(160%)',
            borderRadius: open ? 'var(--r) var(--r) 0 0' : 999,
            boxShadow: 'var(--lift-2), inset 0 1px 0 rgba(255,255,255,.06)',
            transition: 'border-radius .2s ease',
          }}
        />
      </div>

      <AnimatePresence>
        {open && (
          <motion.ul id="hero-suggestions" role="listbox"
            initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }} transition={{ duration: .2, ease: EASE }}
            className="absolute inset-x-0 top-full max-h-[54vh] overflow-y-auto py-1.5"
            style={{
              border: '1px solid rgba(255,255,255,.12)', borderTop: 'none',
              background: 'rgba(12,8,10,.995)',
              backdropFilter: 'blur(22px) saturate(160%)',
              WebkitBackdropFilter: 'blur(22px) saturate(160%)',
              borderRadius: '0 0 var(--r) var(--r)',
              boxShadow: 'var(--lift-3)',
            }}>
            {hits.map((f, i) => (
              <li key={f.item_id} role="option" aria-selected={i === cursor}>
                <button
                  onMouseEnter={() => setCursor(i)}
                  onClick={() => { onSelect(f); setOpen(false); setQ(''); }}
                  className="flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors"
                  style={{ background: i === cursor ? 'rgba(232,53,74,.14)' : 'transparent' }}>
                  <div className="frame h-[50px] w-[34px] shrink-0">
                    {poster(f.poster_path, 'w185')
                      && <img src={poster(f.poster_path, 'w185')!} alt="" loading="lazy" />}
                  </div>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[.875rem]">{f.title}</span>
                    <span className="block truncate text-[.72rem]"
                          style={{ color: 'var(--halide-dim)' }}>
                      {[f.year, f.genres.slice(0, 2).join(', ')].filter(Boolean).join('  ·  ')}
                    </span>
                  </span>
                </button>
              </li>
            ))}
            <li>
              <button onClick={() => { onSubmit(q.trim()); setOpen(false); }}
                      className="w-full px-4 py-2.5 text-left text-[.78rem]"
                      style={{ borderTop: '1px solid rgba(255,255,255,.08)',
                               color: 'var(--halide-dim)' }}>
                See all results for “{q.trim()}”
              </button>
            </li>
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
