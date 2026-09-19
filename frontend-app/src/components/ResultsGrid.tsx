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
      <button onClick={onBack} className="mb-6 text-[0.82rem]"
              style={{ color: 'var(--halide-dim)' }}>
        Back to the collection
      </button>
      <h1 style={{ fontSize: 'var(--t-2xl)' }}>{heading}</h1>
      {note && <p className="machine mt-3">{note}</p>}

      {loading ? (
        <div className="mt-10 grid gap-5"
             style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}>
          {Array.from({ length: 18 }, (_, i) => (
            <div key={i} className="animate-pulse" style={{ aspectRatio: '2/3',
                 background: 'var(--ink-raise)', borderRadius: 'var(--frame)' }} />
          ))}
        </div>
      ) : films.length ? (
        <div className="mt-10 grid gap-x-5 gap-y-8"
             style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}>
          {films.map((f) => (
            <FilmCard key={f.item_id} film={f} onSelect={onSelect} />
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
