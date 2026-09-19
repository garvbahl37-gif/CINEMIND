import { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import Navbar from './components/Navbar';
import ProximityHero from './components/ProximityHero';
import Shelf from './components/Shelf';
import FilmSheet from './components/FilmSheet';
import SearchPanel from './components/SearchPanel';
import ResultsGrid from './components/ResultsGrid';
import HowItWorks from './components/HowItWorks';
import CinematicLoader from './components/CinematicLoader';
import Atmosphere from './components/Atmosphere';
import { api } from './api';
import type { BrowsePayload, Film } from './types';

type View = 'home' | 'top50' | 'tv' | 'about' | 'results';
const EASE = [0.16, 1, 0.3, 1] as const;

export default function App() {
  const [view, setView] = useState<View>('home');
  const [data, setData] = useState<BrowsePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [booted, setBooted] = useState(false);
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

  return (
    <>
      <Atmosphere />
      <CinematicLoader ready={!!data || !!error} onDone={() => setBooted(true)} />

      {error ? (
        <main className="grid min-h-screen place-items-center px-6 text-center">
          <div className="glass max-w-[46ch] p-10">
            <h1 style={{ fontSize: 'var(--t-xl)' }}>The projector didn’t start.</h1>
            <p className="mt-4 text-[.9rem]" style={{ color: 'var(--halide-mid)' }}>
              The catalogue API returned: {error}
            </p>
            <motion.button onClick={() => location.reload()}
              whileHover={{ scale: 1.04 }} whileTap={{ scale: .96 }}
              className="mt-7 px-6 py-3 text-[.875rem] font-semibold"
              style={{ background: 'linear-gradient(135deg, var(--lamp-warm), var(--lamp))',
                       color: '#1B1405', borderRadius: 'var(--r-sm)' }}>
              Try again
            </motion.button>
          </div>
        </main>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: booted ? 1 : 0 }}
          transition={{ duration: .8, ease: EASE }}
        >
          <Navbar view={view} onNavigate={navigate} onSearchFocus={() => setSearchOpen(true)} />

          <AnimatePresence mode="wait">
            <motion.main key={view}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: .45, ease: EASE }}>

              {view === 'home' && data && (
                <>
                  <ProximityHero seeds={data.hero} onSelect={setOpen} />
                  <div className="pb-24">
                    {data.rows.map((r) => (
                      <Shelf key={r.genre} title={r.genre} films={r.items}
                             onSelect={setOpen} count={r.items.length} />
                    ))}
                  </div>
                </>
              )}

              {view === 'top50' && (
                <ResultsGrid heading="The Top 50"
                  note="Ranked by Bayesian average — a 4.6 from forty voters cannot outrank a 4.4 from ninety thousand"
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
            </motion.main>
          </AnimatePresence>

          <AnimatePresence>
            {open && (
              <FilmSheet film={open} onClose={() => setOpen(null)} onSelect={setOpen} />
            )}
          </AnimatePresence>

          <SearchPanel open={searchOpen} onClose={() => setSearchOpen(false)}
                       onSelect={setOpen} onSubmit={runSearch} />

          <footer className="py-12 text-[.78rem]"
                  style={{ borderTop: '1px solid rgba(255,255,255,.06)',
                           color: 'var(--halide-dim)', paddingInline: 'var(--gut)' }}>
            Built on the MovieLens 32M dataset. Artwork and synopses from TMDB.
          </footer>
        </motion.div>
      )}
    </>
  );
}
