import { useState, useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import MovieRow from './components/MovieRow'
import DetailsOverlay from './components/DetailsOverlay'
import { Movie } from './types'
import GlassLoader from './components/GlassLoader'
import AnimatedBackground from './components/AnimatedBackground'

// ============================================
// API Configuration
// ============================================
import { API_BASE, TMDB_API_KEY, TMDB_IMAGE_BASE } from './config';

/** Accepts either a bare TMDB path or an already-absolute URL. */
const posterUrl = (p?: string | null, size = 'w342') =>
  !p ? null : p.startsWith('http') ? p : `${TMDB_IMAGE_BASE}/${size}${p}`;


// Components
import Navbar from './components/Navbar';
import AboutPage from './components/AboutPage';
import SearchResults from './components/SearchResults';
import SearchCommand from './components/SearchCommand';
import { ChatInterface } from './components/ChatInterface';
import {
  Category, Facets, Filters, EMPTY, isActive, fromCategory, searchFilms, fetchCategories,
} from './lib/search';

function App() {
  // Navigation State
  const [currentView, setCurrentView] = useState('home');

  // State
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null)
  const [movieDetails, setMovieDetails] = useState<Movie | null>(null)
  const [recommendations, setRecommendations] = useState<Movie[]>([])
  const [franchiseMovies, setFranchiseMovies] = useState<Movie[]>([])
  const [top50Movies, setTop50Movies] = useState<Movie[]>([]);
  const [tvShows, setTvShows] = useState<Movie[]>([]);
  const [failedPosters, setFailedPosters] = useState<Set<number>>(new Set());

  const [initialLoading, setInitialLoading] = useState(true)
  const [loadingTop50, setLoadingTop50] = useState(true)
  const [loadingTv, setLoadingTv] = useState(true)

  // Search Results State
  const [searchResults, setSearchResults] = useState<Movie[]>([])
  const [isSearching, setIsSearching] = useState(false)

  // Data State
  const [moviesCache, setMoviesCache] = useState<Record<string, Movie>>({})
  const [processedGenres, setProcessedGenres] = useState<string[]>([])
  const [moviesByGenre, setMoviesByGenre] = useState<Record<string, Movie[]>>({})
  // Search + filtering
  const [filters, setFilters] = useState<Filters>(EMPTY)
  const [facets, setFacets] = useState<Facets | null>(null)
  const [vocabulary, setVocabulary] = useState<Facets | null>(null)
  const [totalResults, setTotalResults] = useState(0)

  // Load Movies
  useEffect(() => {
    // Safety timer: Enforce max loading time
    const safetyTimer = setTimeout(() => {
      console.warn("Safety timer triggered: Force exiting loading state");
      setInitialLoading(false);
    }, 5000);

    const loadData = async () => {
      // Start minimum timer (2500ms)
      const minLoadTime = new Promise(resolve => setTimeout(resolve, 2500));

      try {
        // Wrap fetch in a catch so it doesn't fail Promise.all immediately
        const fetchPromise = loadMoviesFromAPI().catch(err => {
          console.error("API Fetch failed:", err);
          return null;
        });

        // Wait for BOTH (now safe from failing early)
        await Promise.all([fetchPromise, minLoadTime]);
      } catch (e) {
        console.error("Load data failed:", e);
      } finally {
        setInitialLoading(false);
      }
    };

    loadData();

    return () => clearTimeout(safetyTimer);
  }, []);

  const loadMoviesFromAPI = async () => {
    try {
      const controller = new AbortController();
      const signal = controller.signal;
      const fetchTimeout = setTimeout(() => controller.abort(), 10000); // 10s timeout

      console.log("Fetching movies from:", `${API_BASE}/api/movies`);

      // Parallelize fetches
      const [moviesRes, top50Res, tvRes] = await Promise.all([
        fetch(`${API_BASE}/api/movies`, { signal }),
        fetch(`${API_BASE}/api/movies/top50`, { signal }),
        fetch(`${API_BASE}/api/movies/tv`, { signal })
      ]);

      clearTimeout(fetchTimeout);

      // Process Movies
      if (moviesRes.ok) {
        const data = await moviesRes.json();
        if (data.movies) {
          setMoviesCache(data.movies);
          processGenres(data.movies);
        }
      } else {
        console.error(`HTTP error! status: ${moviesRes.status}`);
      }

      // Process Top 50
      if (top50Res.ok) {
        const top50Data = await top50Res.json();
        if (top50Data.results) {
          setTop50Movies(top50Data.results.map((m: any) => ({
            ...m,
            poster: m.poster_path ? `${TMDB_IMAGE_BASE}/w500${m.poster_path}` : m.poster,
            id: m.tmdbId
          })));
        }
      }
      setLoadingTop50(false);

      // Process TV Shows
      if (tvRes.ok) {
        const tvData = await tvRes.json();
        if (tvData.results) {
          setTvShows(tvData.results.map((m: any) => ({
            ...m,
            poster: m.poster_path ? `${TMDB_IMAGE_BASE}/w500${m.poster_path}` : m.poster,
            id: m.tmdbId
          })));
        }
      }
      setLoadingTv(false);

    } catch (err) {
      console.error("Failed to load movies:", err);
      // Ensure loading states are cleared on error
      setLoadingTop50(false);
      setLoadingTv(false);
    }
  }

  // Fetch missing posters for Top 50 movies
  useEffect(() => {
    if (top50Movies.length === 0) return;

    const fetchTop50Posters = async () => {
      // Find movies that need updating: have TMDB ID, no/placeholder poster, and haven't failed yet
      const moviesToUpdate = top50Movies.filter(m =>
        m.tmdbId &&
        (!m.poster || m.poster.includes('via.placeholder')) &&
        !failedPosters.has(m.tmdbId)
      );

      if (moviesToUpdate.length === 0) return;

      const newFailed = new Set(failedPosters);
      let hasChanges = false;
      const updatedMoviesMap = new Map(); // Store updates to apply

      await Promise.all(moviesToUpdate.map(async (movie) => {
        try {
          const isTv = movie.media_type === 'tv';
          let response = null;

          try {
            if (isTv) {
              response = await fetch(`https://api.themoviedb.org/3/tv/${movie.tmdbId}?api_key=${TMDB_API_KEY}`);
            } else {
              response = await fetch(`https://api.themoviedb.org/3/movie/${movie.tmdbId}?api_key=${TMDB_API_KEY}`);
              // Fallback to TV if movie fails (only if no media_type)
              if (!response.ok && !movie.media_type) {
                response = await fetch(`https://api.themoviedb.org/3/tv/${movie.tmdbId}?api_key=${TMDB_API_KEY}`);
              }
            }

            if (response && response.ok) {
              const data = await response.json();
              if (data.poster_path) {
                const newPoster = `${TMDB_IMAGE_BASE}/w500${data.poster_path}`;
                updatedMoviesMap.set(movie.tmdbId, { poster: newPoster, poster_path: data.poster_path });
                hasChanges = true;
              }
            } else {
              newFailed.add(movie.tmdbId!);
            }
          } catch (error) {
            if (movie.tmdbId) newFailed.add(movie.tmdbId);
          }
        } catch (e) {
          if (movie.tmdbId) newFailed.add(movie.tmdbId);
        }
      }));

      if (newFailed.size > failedPosters.size) {
        setFailedPosters(newFailed);
      }

      if (hasChanges) {
        setTop50Movies(prevMovies => prevMovies.map(m => {
          if (m.tmdbId && updatedMoviesMap.has(m.tmdbId)) {
            return { ...m, ...updatedMoviesMap.get(m.tmdbId) };
          }
          return m;
        }));
      }
    };

    fetchTop50Posters();
    fetchTop50Posters();
  }, [top50Movies, failedPosters]);

  // Fetch missing posters for TV Shows (reusing similar logic if needed, or rely on correct data)
  useEffect(() => {
    if (tvShows.length === 0) return;

    // Logic to fetch missing TV posters if needed. 
    // Since we hardcoded/fixed them in backend, strictly speaking unnecessary, BUT good for robustness.
    // For now, let's rely on the IDs being correct and fallback fetching in the UI component if we had one.
    // Actually, let's add the safe fetcher here too just in case.

    const fetchTVPosters = async () => {
      const moviesToUpdate = tvShows.filter(m =>
        m.tmdbId &&
        (!m.poster || m.poster.includes('via.placeholder')) &&
        !failedPosters.has(m.tmdbId)
      );

      if (moviesToUpdate.length === 0) return;

      const newFailed = new Set(failedPosters);
      let hasChanges = false;
      const updatedMap = new Map();

      await Promise.all(moviesToUpdate.map(async (movie) => {
        try {
          // Tv endpoint primarily
          let response = await fetch(`https://api.themoviedb.org/3/tv/${movie.tmdbId}?api_key=${TMDB_API_KEY}`);
          if (response.ok) {
            const data = await response.json();
            if (data.poster_path) {
              const newPoster = `${TMDB_IMAGE_BASE}/w500${data.poster_path}`;
              updatedMap.set(movie.tmdbId, { poster: newPoster });
              hasChanges = true;
            }
          } else {
            newFailed.add(movie.tmdbId!);
          }
        } catch (e) {
          if (movie.tmdbId) newFailed.add(movie.tmdbId);
        }
      }));

      if (newFailed.size > failedPosters.size) setFailedPosters(newFailed);

      if (hasChanges) {
        setTvShows(prev => prev.map(m => {
          if (m.tmdbId && updatedMap.has(m.tmdbId)) {
            return { ...m, ...updatedMap.get(m.tmdbId) };
          }
          return m;
        }));
      }
    };

    fetchTVPosters();
  }, [tvShows, failedPosters]);

  const processGenres = (movies: Record<string, any>) => {
    const genres: Record<string, Movie[]> = {}

    Object.values(movies).forEach((movie: any) => {

      // Robust Poster Logic
      let fullPoster = null;

      // 1. Check if backend provided a valid http link
      if (movie.poster && movie.poster.startsWith('http')) {
        fullPoster = movie.poster;
      }
      // 2. Check if backend provided a TMDB path (starts with /)
      else if (movie.poster && movie.poster.startsWith('/')) {
        fullPoster = `${TMDB_IMAGE_BASE}/w500${movie.poster}`;
      }
      // 3. Fallback to poster_path if available
      else if (movie.poster_path) {
        fullPoster = `${TMDB_IMAGE_BASE}/w500${movie.poster_path}`;
      }

      const processedMovie = {
        ...movie,
        id: movie.tmdbId,
        item_id: movie.item_id,
        poster: fullPoster
      };

      if (movie.genres) {
        movie.genres.forEach((genre: string) => {
          if (!genres[genre]) genres[genre] = []
          genres[genre].push(processedMovie)
        })
      }
    })

    const sortedGenres = Object.keys(genres).sort((a, b) => genres[b].length - genres[a].length)
    setMoviesByGenre(genres)
    setProcessedGenres(sortedGenres)
  }

  // Genre / decade / language vocabulary for the chips, fetched once.
  useEffect(() => {
    fetchCategories().then(setVocabulary).catch(() => { });
  }, []);

  const handleSelectMovie = async (movie: Movie) => {
    setSelectedMovie(movie)
    setFranchiseMovies([]); // Reset franchise
    if (movie.tmdbId) {
      try {
        const isTv = movie.media_type === 'tv';
        const endpoint = isTv ? `tv` : `movie`;
        const response = await fetch(`https://api.themoviedb.org/3/${endpoint}/${movie.tmdbId}?api_key=${TMDB_API_KEY}`)
        const data = await response.json()

        const details: Movie = {
          ...movie,
          title: isTv ? data.name : data.title,
          overview: data.overview,
          vote_average: data.vote_average,
          releaseDate: isTv ? data.first_air_date : data.release_date,
          runtime: isTv ? (data.episode_run_time?.[0] || 0) : data.runtime,
          genres: data.genres?.map((g: any) => g.name) || [],
          backdrop: data.backdrop_path ? `${TMDB_IMAGE_BASE}/original${data.backdrop_path}` : undefined,
          poster: data.poster_path ? `${TMDB_IMAGE_BASE}/w500${data.poster_path}` : movie.poster,
          media_type: isTv ? 'tv' : 'movie'
        }



        setMovieDetails(details)
        fetchRecommendations(movie)

        // Fetch Collection/Franchise
        if (data.belongs_to_collection) {
          try {
            const collectionRes = await fetch(`https://api.themoviedb.org/3/collection/${data.belongs_to_collection.id}?api_key=${TMDB_API_KEY}`)
            const collectionData = await collectionRes.json()

            if (collectionData.parts) {
              const parts = collectionData.parts
                .filter((p: any) => p.id !== movie.tmdbId) // Exclude current movie
                .map((p: any) => ({
                  ...p,
                  tmdbId: p.id,
                  poster: p.poster_path ? `${TMDB_IMAGE_BASE}/w342${p.poster_path}` : null,
                  backdrop: p.backdrop_path ? `${TMDB_IMAGE_BASE}/original${p.backdrop_path}` : null,
                  releaseDate: p.release_date,
                  vote_average: p.vote_average
                }))
                .sort((a: any, b: any) => new Date(a.releaseDate).getTime() - new Date(b.releaseDate).getTime()) // Sort by release date

              setFranchiseMovies(parts)
            }
          } catch (err) {
            console.error("Failed to fetch collection:", err)
          }
        }
      } catch (error) {
        console.error("Fetch details error", error)
      }
    }
  }

  const fetchRecommendations = async (movie: Movie) => {
    // The catalogue call only carries the browsable subset, so a neighbour may
    // not be in it. /api/similar returns whole films — use those directly and
    // fall back to the cache only for anything it leaves out.
    const backendId = movie.item_id !== undefined
      ? String(movie.item_id)
      : Object.keys(moviesCache).find(key => moviesCache[key].tmdbId === movie.tmdbId)
    if (backendId) {
      try {
        const response = await fetch(`${API_BASE}/api/similar/${backendId}?k=6`)
        const data = await response.json()
        const recs = (data.similar_items || []).map((item: any) => {
          const m = moviesCache[String(item.item_id)] ?? {}
          const merged = { ...m, ...item }
          return {
            ...merged,
            id: merged.tmdbId,
            poster: posterUrl(merged.poster) ?? posterUrl(merged.poster_path),
          } as Movie
        }).filter(Boolean) as Movie[]

        const recsWithPosters = await Promise.all(recs.map(async (rec) => {
          if ((!rec.poster || rec.poster.includes("via.placeholder")) && rec.tmdbId) {
            try {
              const res = await fetch(`https://api.themoviedb.org/3/movie/${rec.tmdbId}?api_key=${TMDB_API_KEY}`)
              const d = await res.json()
              if (d.poster_path) return { ...rec, poster: `${TMDB_IMAGE_BASE}/w342${d.poster_path}`, backdrop: `${TMDB_IMAGE_BASE}/w780${d.backdrop_path}` }
            } catch (e) { }
          }
          return rec
        }))

        setRecommendations(recsWithPosters)
      } catch (err) {
        console.error(err)
      }
    }
  }

  const handleNavigation = (view: string) => {
    setCurrentView(view);
    if (view === 'home') {
      setSelectedMovie(null);
      setSearchQuery('');
    }
  };

  /**
   * Run a search.
   *
   * Passing `null` for the filters lets the query speak for itself — the server
   * parses "korean thrillers" into Korean + Thriller and tells us what it
   * applied, which we then show as selected chips. Passing an object means the
   * user has touched the filter bar, and it wins over anything the text implied.
   */
  const runSearch = async (query: string, f: Filters | null) => {
    // Clearing the last filter with nothing typed leaves no request to make —
    // go back to browsing rather than showing an empty result page.
    if (!query.trim() && f && !isActive(f)) {
      setCurrentView('home');
      setFilters(EMPTY);
      setSearchResults([]);
      return;
    }
    setIsSearching(true);
    setCurrentView('results');
    setSearchQuery(query);
    if (f) setFilters(f);

    try {
      const d = await searchFilms(query, f, { limit: 60, facets: true });
      setSearchResults(d.results);
      setFacets(d.facets);
      setTotalResults(d.total);
      if (!f) setFilters(d.filters);      // adopt whatever the query itself meant
    } catch (e) {
      console.error('Search failed', e);
      setSearchResults([]);
      setFacets(null);
      setTotalResults(0);
    } finally {
      setIsSearching(false);
    }
  };

  /** A category row or a genre chip: its filters replace the current set. */
  const runCategory = (c: Category) => {
    if (c.filters.q) { runSearch(c.filters.q, EMPTY); return; }
    const next = fromCategory(c, EMPTY);
    setSearchQuery('');
    runSearch('', next);
  };

  if (initialLoading) {
    return <GlassLoader />;
  }

  return (
    <div className="min-h-screen text-white font-sans selection:bg-primary selection:text-white pb-20 overflow-x-hidden relative">
      <Navbar onNavigate={handleNavigation} currentPage={currentView} />

      <AnimatedBackground />
      {currentView === 'home' && (
        <>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1 }}
            className="relative z-10" // Added z-index to ensure content is above background
          >
            {/* Hero Search Section */}
            <div className="relative z-20 pt-32 md:pt-44 pb-8 px-4 flex flex-col items-center justify-center space-y-4 md:space-y-6">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="relative group cursor-default"
              >
                {/* Enhanced Title - Metallic Chrome Effect */}
                <div className="relative">
                  <h1 className="text-5xl sm:text-7xl md:text-9xl font-display font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white via-gray-200 to-gray-500 drop-shadow-2xl select-none relative z-10 text-center">
                    <span className="bg-gradient-to-b from-white via-gray-300 to-gray-500 bg-clip-text text-transparent filter drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)]">CINE</span>
                    <span className="bg-gradient-to-b from-primary via-red-500 to-red-900 bg-clip-text text-transparent filter drop-shadow-[0_0_10px_rgba(220,38,38,0.5)]">MIND</span>
                  </h1>
                  {/* Glow effect underneath */}
                  <div className="absolute inset-0 bg-primary/20 blur-[60px] opacity-40 rounded-full z-0 scale-75 group-hover:scale-100 transition-transform duration-700 pointer-events-none" />
                </div>
              </motion.div>

              {/* Welcome Phrase */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.8 }}
                className="relative z-20 pb-2 text-center"
              >
                <h2 className="text-lg md:text-3xl font-medium tracking-wide text-transparent bg-clip-text bg-gradient-to-r from-white via-gray-200 to-gray-400 drop-shadow-sm px-4">
                  What are you in the mood for?
                </h2>
              </motion.div>


              <SearchCommand
                value={searchQuery}
                onChange={setSearchQuery}
                onSearch={(q) => runSearch(q, null)}
                onCategory={runCategory}
                onMovie={handleSelectMovie}
                quickGenres={(vocabulary?.genres ?? []).slice(0, 8).map(g => String(g.value))}
              />
            </div>

            <main className="relative z-10 space-y-8">
              {/* Genre Rows */}
              <div className="relative z-20 pb-10 space-y-2">
                <div className="relative z-30 space-y-12">

                  {processedGenres.map((genre) => {
                    const movies = moviesByGenre[genre]
                    if (!movies || movies.length === 0) return null

                    return (
                      <MovieRow
                        key={genre}
                        title={`${genre}`}
                        movies={movies}
                        onSelectMovie={handleSelectMovie}
                      />
                    )
                  })}
                </div>
              </div>
            </main>
          </motion.div>
        </>
      )}

      {currentView === 'about' && <AboutPage />}

      {currentView === 'results' && (
        <div className="pt-24 px-4 md:px-12">
          <SearchResults
            query={searchQuery}
            results={searchResults}
            loading={isSearching}
            onBack={() => { setCurrentView('home'); setFilters(EMPTY); }}
            onSelectMovie={handleSelectMovie}
            filters={filters}
            onFilters={(f) => runSearch(searchQuery, f)}
            facets={facets}
            vocabulary={vocabulary}
            total={totalResults}
          />
        </div>
      )}

      {currentView === 'top50' && (
        <div className="pt-24 px-4 md:px-12">
          <SearchResults
            query="All-Time Top 50"
            results={top50Movies}
            loading={loadingTop50}
            onBack={() => setCurrentView('home')}
            onSelectMovie={handleSelectMovie}
          />
        </div>
      )}

      {currentView === 'tvshows' && (
        <div className="pt-24 px-4 md:px-12">
          <SearchResults
            query="Popular TV Shows"
            results={tvShows}
            loading={loadingTv}
            onBack={() => setCurrentView('home')}
            onSelectMovie={handleSelectMovie}
          />
        </div>
      )}

      <AnimatePresence>
        {selectedMovie && (
          <DetailsOverlay
            movie={movieDetails || selectedMovie}
            onClose={() => setSelectedMovie(null)}
            similarMovies={recommendations}
            franchiseMovies={franchiseMovies}
            onSelectSimilar={handleSelectMovie}
          />
        )}
      </AnimatePresence>
      <ChatInterface />
    </div >
  )
}

export default App
