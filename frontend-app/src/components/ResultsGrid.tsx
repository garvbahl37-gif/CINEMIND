import { motion } from 'framer-motion';
import FilmCard from './FilmCard';
import type { Film } from '../types';

export default function ResultsGrid({
  heading, note, films, loading, onSelect, onBack,
}: {
  heading: string; note?: string; films: Film[]; loading: boolean;
  onSelect: (f: Film) => void; onBack: () => void;
}) {
  return (
    <div className="pt-28 pb-20" style={{ paddingInline: 'var(--gut)' }}>
      <motion.button onClick={onBack} className="mb-7 flex items-center gap-2 text-[.82rem]"
              whileHover={{ x: -4 }} transition={{ duration: .2 }}
              style={{ color: 'var(--halide-dim)' }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             strokeWidth="2" strokeLinecap="round"><path d="M15 18l-6-6 6-6" /></svg>
        Back to the collection
      </motion.button>
      <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                 transition={{ duration: .6, ease: [0.16, 1, 0.3, 1] }}
                 style={{ fontSize: 'var(--t-2xl)' }}>{heading}</motion.h1>
      {note && <motion.p className="machine mt-3" initial={{ opacity: 0 }}
                         animate={{ opacity: 1 }} transition={{ delay: .2 }}>{note}</motion.p>}

      {loading ? (
        <div className="mt-10 grid gap-5"
             style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}>
          {Array.from({ length: 18 }, (_, i) => (
            <div key={i} className="animate-pulse" style={{ aspectRatio: '2/3',
                 background: 'rgba(255,255,255,.035)', borderRadius: 'var(--r-sm)' }} />
          ))}
        </div>
      ) : films.length ? (
        <div className="mt-10 grid gap-x-5 gap-y-8"
             style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}>
          {films.map((f, i) => (
            <FilmCard key={f.item_id} film={f} onSelect={onSelect} index={i} />
          ))}
        </div>
      ) : (
        <div className="mt-16 max-w-[46ch]">
          <p style={{ fontFamily: 'var(--serif)', fontSize: 'var(--t-lg)' }}>
            Nothing in the collection matches that.
          </p>
          <p className="mt-3 text-[0.9rem]" style={{ color: 'var(--halide-dim)' }}>
            The catalogue covers 17,719 films from MovieLens. Try a broader term,
            a genre, or a decade such as “80s horror”.
          </p>
        </div>
      )}
    </div>
  );
}
