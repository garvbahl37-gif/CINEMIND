import { useEffect, useRef, useState } from 'react';
import { poster } from '../config';
import { api } from '../api';
import type { Film } from '../types';

export default function SearchPanel({
  open, onClose, onSelect, onSubmit,
}: {
  open: boolean; onClose: () => void;
  onSelect: (f: Film) => void; onSubmit: (q: string) => void;
}) {
  const [q, setQ] = useState('');
  const [hits, setHits] = useState<Film[]>([]);
  const [cursor, setCursor] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => { if (open) setTimeout(() => input.current?.focus(), 40); }, [open]);

  useEffect(() => {
    if (q.trim().length < 2) { setHits([]); return; }
    const ac = new AbortController();
    const t = setTimeout(() => {
      api.suggest(q, ac.signal).then((d) => { setHits(d.results); setCursor(0); }).catch(() => {});
    }, 130);
    return () => { clearTimeout(t); ac.abort(); };
  }, [q]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', k);
    return () => document.removeEventListener('keydown', k);
  }, [onClose]);

  if (!open) return null;

  const keys = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setCursor((c) => Math.min(c + 1, hits.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
    else if (e.key === 'Enter') {
      if (hits[cursor]) { onSelect(hits[cursor]); onClose(); }
      else if (q.trim()) { onSubmit(q.trim()); onClose(); }
    }
  };

  return (
    <div className="fixed inset-0 z-[110] flex justify-center px-4 pt-[12vh]"
         style={{ background: 'rgba(6,9,18,.8)', backdropFilter: 'blur(5px)' }}
         onClick={onClose}>
      <div className="h-fit w-full max-w-[620px] border"
           style={{ background: 'var(--ink)', borderColor: 'var(--ink-edge)',
                    borderRadius: 'var(--frame)' }}
           onClick={(e) => e.stopPropagation()}>
        <input
          ref={input} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={keys}
          placeholder="A title, a genre, a decade — try “90s sci-fi”"
          aria-label="Search films"
          className="w-full bg-transparent px-5 py-4 text-[1.05rem] outline-none"
          style={{ borderBottom: hits.length ? '1px solid var(--ink-edge)' : 'none' }}
        />
        {hits.length > 0 && (
          <ul className="max-h-[52vh] overflow-y-auto py-1.5">
            {hits.map((f, i) => (
              <li key={f.item_id}>
                <button
                  onMouseEnter={() => setCursor(i)}
                  onClick={() => { onSelect(f); onClose(); }}
                  className="flex w-full items-center gap-3 px-4 py-2 text-left"
                  style={{ background: i === cursor ? 'var(--ink-raise)' : 'transparent' }}>
                  <div className="frame h-[48px] w-[32px] shrink-0">
                    {poster(f.poster_path, 'w185')
                      ? <img src={poster(f.poster_path, 'w185')!} alt="" loading="lazy" />
                      : null}
                  </div>
                  <span className="min-w-0 flex-1 truncate text-[0.9rem]">{f.title}</span>
                  <span className="shrink-0 text-[0.75rem]" style={{ color: 'var(--halide-dim)' }}>
                    {f.year}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
        {q.trim().length >= 2 && (
          <button onClick={() => { onSubmit(q.trim()); onClose(); }}
                  className="w-full px-5 py-3 text-left text-[0.82rem]"
                  style={{ borderTop: '1px solid var(--ink-edge)', color: 'var(--halide-dim)' }}>
            See all results for “{q.trim()}”
          </button>
        )}
      </div>
    </div>
  );
}
