import { useCallback, useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import ProximityHero from './components/ProximityHero';
import Shelf from './components/Shelf';
import FilmSheet from './components/FilmSheet';
import SearchPanel from './components/SearchPanel';
import ResultsGrid from './components/ResultsGrid';
import HowItWorks from './components/HowItWorks';
import { api } from './api';
import type { BrowsePayload, Film } from './types';

type View = 'home' | 'top50' | 'tv' | 'about' | 'results';

export default function App() {
  const [view, setView] = useState<View>('home');
  const [data, setData] = useState<BrowsePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<Film | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Film[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchMs, setSearchMs] = useState<number | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    api.browse(ac.signal)
      .then(setData)
      .catch((e) => { if (e.name !== 'AbortError') setError(String(e.message ?? e)); });
    return () => ac.abort();
  }, []);

  // "/" opens search from anywhere, the way a catalogue tool should behave.
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault(); setSearchOpen(true);
      }
    };
    document.addEventListener('keydown', k);
    return () => document.removeEventListener('keydown', k);
  }, []);

  const runSearch = useCallback(async (q: string) => {
    setQuery(q); setView('results'); setSearching(true); setResults([]);
    try {
      const d = await api.search(q);
      setResults(d.results); setSearchMs(d.latency_ms);
    } catch { setResults([]); }
    finally { setSearching(false); }
  }, []);

  const navigate = (v: string) => { setView(v as View); window.scrollTo({ top: 0 }); };

  if (error) {
    return (
      <main className="grid min-h-screen place-items-center px-6 text-center">
        <div className="max-w-[44ch]">
          <h1 style={{ fontSize: 'var(--t-xl)' }}>The catalogue didn’t load.</h1>
          <p className="mt-4 text-[0.9rem]" style={{ color: 'var(--halide-mid)' }}>
            The API returned: {error}
          </p>
          <button onClick={() => location.reload()}
                  className="mt-7 px-5 py-2.5 text-[0.875rem] font-semibold"
                  style={{ background: 'var(--lamp)', color: 'var(--ink-deep)',
                           borderRadius: 'var(--frame)' }}>
            Try again
          </button>
        </div>
      </main>
    );
  }

  return (
    <>
      <Navbar view={view} onNavigate={navigate} onSearchFocus={() => setSearchOpen(true)} />

      {view === 'home' && (
        <main>
          {data ? (
            <ProximityHero seeds={data.hero} onSelect={setOpen} />
          ) : (
            <div className="min-h-[70vh] pt-40" style={{ paddingInline: 'var(--gut)' }}>
              <div className="mx-auto max-w-[1180px]">
                <div className="h-4 w-56 animate-pulse"
                     style={{ background: 'var(--ink-raise)' }} />
                <div className="mt-6 h-20 w-full max-w-[620px] animate-pulse"
                     style={{ background: 'var(--ink-raise)' }} />
              </div>
            </div>
          )}

          <div className="pb-24 pt-2">
            {data?.rows.map((r) => (
              <Shelf key={r.genre} title={r.genre} films={r.items}
                     onSelect={setOpen} count={r.items.length} />
            ))}
          </div>
        </main>
      )}

      {view === 'top50' && (
        <ResultsGrid heading="The top 50"
                     note="Ranked by Bayesian average, so a 4.6 from forty voters cannot outrank a 4.4 from ninety thousand"
                     films={data?.top50 ?? []} loading={!data}
                     onSelect={setOpen} onBack={() => navigate('home')} />
      )}

      {view === 'tv' && (
        <ResultsGrid heading="Series" films={data?.tv ?? []} loading={!data}
                     onSelect={setOpen} onBack={() => navigate('home')} />
      )}

      {view === 'results' && (
        <ResultsGrid heading={`“${query}”`}
                     note={searchMs !== null && !searching
                       ? `${results.length} matches in ${searchMs.toFixed(1)} ms` : undefined}
                     films={results} loading={searching}
                     onSelect={setOpen} onBack={() => navigate('home')} />
      )}

      {view === 'about' && <HowItWorks />}

      {open && (
        <FilmSheet film={open} onClose={() => setOpen(null)} onSelect={setOpen} />
      )}

      <SearchPanel open={searchOpen} onClose={() => setSearchOpen(false)}
                   onSelect={setOpen} onSubmit={runSearch} />

      <footer className="border-t py-10 text-[0.78rem]"
              style={{ borderColor: 'var(--ink-edge)', color: 'var(--halide-dim)',
                       paddingInline: 'var(--gut)' }}>
        Built on the MovieLens 32M dataset. Film metadata and artwork from TMDB.
      </footer>
    </>
  );
}
