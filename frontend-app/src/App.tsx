import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import { Movie } from './types';
import { Browse, fetchBlend, fetchBrowse } from './lib/api';
import {
  Category, Facets, Filters, EMPTY, isActive, fromCategory, searchFilms, fetchCategories,
} from './lib/search';
import { useBlend } from './lib/blend';
import TopBar, { View } from './components/TopBar';
import HeroRail from './components/HeroRail';
import Shelf, { Toggle } from './components/Shelf';
import PosterCard from './components/PosterCard';
import Collections, { Collection, Look, filtersOf, title as collectionTitle } from './components/Collections';
import SearchResults from './components/SearchResults';
import MatchStage from './components/MatchStage';
import MatchSearch from './components/MatchSearch';
import Loader from './components/Loader';
import AboutPage from './components/AboutPage';
import { ChatInterface } from './components/ChatInterface';

/** What the results banner wears: a collection's name and ground, or nothing. */
interface Banner { title?: string; look?: Look; backdrop?: string }

const RowSkeleton = () => (
  <>
    {Array.from({ length: 8 }, (_, i) => (
      <div key={i} className="shrink-0 w-[150px] sm:w-[172px] lg:w-[196px] aspect-[2/3] rounded-[18px] bg-panel-hi animate-pulse" />
    ))}
  </>
);

function App() {
  const [view, setView] = useState<View>('home');
  const [booted, setBooted] = useState(false);

  // The whole home page arrives in one prebuilt response.
  const [browse, setBrowse] = useState<Browse | null>(null);
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  // Search: the text in the box, and the query the results on screen came from.
  const [query, setQuery] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [banner, setBanner] = useState<Banner>({});
  const [filters, setFilters] = useState<Filters>(EMPTY);
  const [facets, setFacets] = useState<Facets | null>(null);
  const [vocabulary, setVocabulary] = useState<Facets | null>(null);
  const [results, setResults] = useState<Movie[]>([]);
  const [total, setTotal] = useState(0);
  const [searching, setSearching] = useState(false);
  const latest = useRef(0);

  // The film on the match stage, if any.
  const [pick, setPick] = useState<Movie | null>(null);

  const blend = useBlend();
  const [fromBlend, setFromBlend] = useState(false);
  const [blendRecs, setBlendRecs] = useState<Movie[] | null>(null);

  useEffect(() => {
    setBrowseError(null);
    fetchBrowse()
      .then(setBrowse)
      .catch(e => setBrowseError(e instanceof Error ? e.message : String(e)));
  }, [attempt]);

  // Genre / decade / language vocabulary, fetched once.
  useEffect(() => { fetchCategories().then(setVocabulary).catch(() => { }); }, []);

  const seeds = blend.films.map(f => f.item_id).join(',');
  useEffect(() => {
    if (!seeds) setFromBlend(false);
  }, [seeds]);
  useEffect(() => {
    if (!fromBlend || !seeds) { setBlendRecs(null); return; }
    const ctrl = new AbortController();
    setBlendRecs(null);
    fetchBlend(seeds.split(',').map(Number), 24, ctrl.signal)
      .then(setBlendRecs)
      .catch(e => { if (e.name !== 'AbortError') setBlendRecs([]); });
    return () => ctrl.abort();
  }, [fromBlend, seeds]);

  // Largest genres first, once their sizes are known.
  const genreRows = useMemo(() => {
    const size = new Map((vocabulary?.genres ?? []).map(g => [String(g.value), g.count]));
    return [...(browse?.rows ?? [])].sort((a, b) => (size.get(b.genre) ?? 0) - (size.get(a.genre) ?? 0));
  }, [browse, vocabulary]);
  const genreCount = (g: string) => vocabulary?.genres.find(x => x.value === g)?.count;

  const navigate = (v: View) => {
    setView(v);
    window.scrollTo({ top: 0 });
    if (v === 'home') { setQuery(''); setFilters(EMPTY); }
  };

  /**
   * Run a search.
   *
   * Passing `null` for the filters lets the query speak for itself — the server
   * parses "korean thrillers" into Korean + Thriller and tells us what it
   * applied, which the filter bar then shows as selected. Passing an object
   * means the filters were chosen, and they win over anything the text implied.
   */
  const runSearch = async (q: string, f: Filters | null, meta: Banner = {}) => {
    // Clearing the last filter with nothing typed leaves no request to make —
    // go back to browsing rather than showing an empty result page.
    if (!q.trim() && f && !isActive(f)) { navigate('home'); setResults([]); return; }
    const id = ++latest.current;
    setSearching(true);
    if (view !== 'results') window.scrollTo({ top: 0 });
    setView('results');
    setSearchQuery(q);
    setBanner(meta);
    if (f) setFilters(f);

    try {
      const d = await searchFilms(q, f, { limit: 60, facets: true });
      if (id !== latest.current) return;   // a newer search has already answered
      setResults(d.results);
      setFacets(d.facets);
      setTotal(d.total);
      if (!f) setFilters(d.filters);       // adopt whatever the query itself meant
    } catch (e) {
      if (id !== latest.current) return;
      console.error('Search failed', e);
      setResults([]); setFacets(null); setTotal(0);
    } finally {
      if (id === latest.current) setSearching(false);
    }
  };

  /** A category suggestion or a genre shelf: its filters replace the current set. */
  const runCategory = (c: Category) => {
    if (c.filters.q) { runSearch(c.filters.q, EMPTY); return; }
    setQuery('');
    runSearch('', fromCategory(c, EMPTY));
  };

  const openCollection = (c: Collection, backdrop?: string) =>
    runSearch(c.q ?? '', filtersOf(c), { title: collectionTitle(c), look: c.look, backdrop });

  const recs = fromBlend ? blendRecs : browse?.top50.slice(0, 24) ?? null;

  return (
    <div className="min-h-screen p-1.5 md:p-4">
      <div className="min-h-[calc(100vh-12px)] md:min-h-[calc(100vh-32px)] rounded-[28px] bg-ink p-1.5 md:p-2.5 overflow-clip">
        <TopBar
          view={view}
          onNavigate={navigate}
          query={query}
          onQuery={setQuery}
          onSearch={(q) => runSearch(q, null)}
          onCategory={runCategory}
          onMovie={setPick}
          blend={blend}
          onUseBlend={() => { navigate('home'); setFromBlend(true); }}
        />

        <main className="space-y-1.5 md:space-y-2.5">
          {view === 'home' && (
            browseError ? (
              <section className="bg-panel rounded-[24px] p-6 md:p-10">
                <h1 className="display text-[40px] md:text-[64px]">Can't reach the catalogue</h1>
                <p className="mt-4 max-w-[56ch] text-white/70">
                  The recommendation API didn't answer ({browseError}). If you're running locally, start it
                  with uvicorn on port 8000, or point the dev server at the deployed one.
                </p>
                <button type="button" className="pill pill-white mt-6" onClick={() => setAttempt(a => a + 1)}>Try again</button>
              </section>
            ) : (
              <>
                <MatchSearch onPick={setPick} />

                {browse
                  ? <HeroRail films={browse.hero} onMatches={setPick} blend={blend} />
                  : <div className="h-[min(76vh,640px)] min-h-[440px] rounded-[24px] bg-panel animate-pulse" aria-label="Loading" />}

                <Shelf
                  title={fromBlend ? 'From your blend' : 'Recommended'}
                  aside={fromBlend
                    ? `${blend.films.length} ${blend.films.length === 1 ? 'film' : 'films'} as one taste`
                    : 'Highest rated across 32M ratings'}
                  controls={<Toggle label="My blend" on={fromBlend} onChange={setFromBlend}
                    disabled={!blend.films.length} hint="Add films to your blend first" />}
                  onShowAll={fromBlend ? undefined : () => navigate('top50')}
                >
                  {recs ? recs.map((m, i) => (
                    <PosterCard key={m.item_id ?? i} movie={m} onSelect={setPick}
                      figure={fromBlend ? 'match' : 'rating'} />
                  )) : <RowSkeleton />}
                </Shelf>

                <Collections onOpen={openCollection} />

                {genreRows.map(r => {
                  const n = genreCount(r.genre);
                  return (
                    <Shelf key={r.genre} title={r.genre} aside={n ? `${n.toLocaleString()} films` : undefined}
                      onShowAll={() => runCategory({ kind: 'genre', label: r.genre, sublabel: '', filters: { genres: [r.genre] } })}
                      className="[content-visibility:auto] [contain-intrinsic-size:auto_420px]">
                      {r.items.map(m => <PosterCard key={m.item_id} movie={m} onSelect={setPick} />)}
                    </Shelf>
                  );
                })}
              </>
            )
          )}

          {view === 'results' && (
            <SearchResults
              query={searchQuery}
              title={banner.title}
              look={banner.look}
              backdrop={banner.backdrop}
              results={results}
              loading={searching}
              onBack={() => navigate('home')}
              onSelectMovie={setPick}
              filters={filters}
              // Changing filters keeps the collection's colours but not its
              // name, which would no longer describe what is shown.
              onFilters={(f) => runSearch(searchQuery, f, { look: banner.look, backdrop: banner.backdrop })}
              facets={facets}
              vocabulary={vocabulary}
              total={total}
            />
          )}

          {view === 'top50' && (
            <SearchResults
              query=""
              title="Top 50"
              look="blood"
              note="Bayesian average, so a few perfect scores can't outrank The Godfather"
              results={browse?.top50 ?? []}
              loading={!browse && !browseError}
              onBack={() => navigate('home')}
              onSelectMovie={setPick}
              ranked
            />
          )}

          {view === 'tv' && (
            <SearchResults
              query=""
              title="Series"
              look="violet"
              results={browse?.tv ?? []}
              loading={!browse && !browseError}
              onBack={() => navigate('home')}
              onSelectMovie={setPick}
            />
          )}

          {view === 'about' && (
            <AboutPage onSelect={setPick} />
          )}
        </main>

        <footer className="px-4 md:px-6 py-8 flex flex-wrap gap-x-8 gap-y-2 text-[12px] text-white/50">
          <span>Ratings and tags from MovieLens, GroupLens Research.</span>
          <span>Artwork, synopses and credits from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.</span>
        </footer>
      </div>

      {/* The frame stays put while the page scrolls inside it. */}
      <div aria-hidden="true" className="pointer-events-none fixed inset-1.5 md:inset-4 z-[45] rounded-[28px] shadow-[0_0_0_100vmax_#2B1FBB]" />

      <AnimatePresence>
        {pick && <MatchStage key="stage" pick={pick} onClose={() => setPick(null)} blend={blend} />}
      </AnimatePresence>
      <ChatInterface onSelect={setPick} />
      {!booted && <Loader ready={!!browse || !!browseError} onDone={() => setBooted(true)} />}
    </div>
  );
}

export default App;
