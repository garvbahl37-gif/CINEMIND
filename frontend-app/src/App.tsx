import { useState, useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Search } from 'lucide-react'
import MovieRow from './components/MovieRow'
import DetailsOverlay from './components/DetailsOverlay'
import { Movie } from './types'
import GlassLoader from './components/GlassLoader'
import AnimatedBackground from './components/AnimatedBackground'

// ============================================
// API Configuration
// ============================================
import { API_BASE, TMDB_API_KEY, TMDB_IMAGE_BASE } from './config';


// Components
import Navbar from './components/Navbar';
import AboutPage from './components/AboutPage';
import SearchResults from './components/SearchResults';
import { ChatInterface } from './components/ChatInterface';

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
  const [allMoviesList, setAllMoviesList] = useState<Movie[]>([]) // For search
  const [searchSuggestions, setSearchSuggestions] = useState<Movie[]>([])

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

      console.log("Fetching movies from:", `${API_BASE}/movies`);

      // Parallelize fetches
      const [moviesRes, top50Res, tvRes] = await Promise.all([
        fetch(`${API_BASE}/movies`, { signal }),
        fetch(`${API_BASE}/movies/top50`, { signal }),
        fetch(`${API_BASE}/movies/tv`, { signal })
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
    const allMovies: Movie[] = []

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
        poster: fullPoster
      };

      allMovies.push(processedMovie);

      if (movie.genres) {
        movie.genres.forEach((genre: string) => {
          if (!genres[genre]) genres[genre] = []
          genres[genre].push(processedMovie)
        })
      }
    })

    setAllMoviesList(allMovies);

    const sortedGenres = Object.keys(genres).sort((a, b) => genres[b].length - genres[a].length)
    setMoviesByGenre(genres)
    setProcessedGenres(sortedGenres)
  }

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (query.length > 1) {
      const lowerQuery = query.toLowerCase();
      const suggestions = allMoviesList.filter(m =>
        m.title.toLowerCase().includes(lowerQuery)
      ).slice(0, 5); // Limit to 5 suggestions
      setSearchSuggestions(suggestions);
    } else {
      setSearchSuggestions([]);
    }
  }

  // Track which movies we've already tried to fetch posters for to avoid infinite loops
  const [triedPosterFetch, setTriedPosterFetch] = useState<Set<number>>(new Set());

  // Effect to fetch missing posters for search suggestions
  useEffect(() => {
    const fetchMissingPosters = async () => {
      const moviesToUpdate = searchSuggestions.filter(m =>
        m.tmdbId &&
        (!m.poster || m.poster.includes('via.placeholder')) &&
        !triedPosterFetch.has(m.tmdbId)
      );

      if (moviesToUpdate.length === 0) return;

      // Mark as tried immediately
      setTriedPosterFetch(prev => {
        const next = new Set(prev);
        moviesToUpdate.forEach(m => next.add(m.tmdbId!));
        return next;
      });

      const newFailed = new Set(failedPosters);

      const updatedMovies = await Promise.all(searchSuggestions.map(async (movie) => {
        // If it already has a good poster, ignore
        if (movie.poster && !movie.poster.includes('via.placeholder')) return movie;

        // If we just marked it as tried, we should try to fetch it
        if (movie.tmdbId && !triedPosterFetch.has(movie.tmdbId) && !failedPosters.has(movie.tmdbId)) {
          try {
            // Try movie endpoint first
            let response = await fetch(`https://api.themoviedb.org/3/movie/${movie.tmdbId}?api_key=${TMDB_API_KEY}`);

            // If movie fails (e.g., 404), try TV endpoint
            if (!response.ok) {
              response = await fetch(`https://api.themoviedb.org/3/tv/${movie.tmdbId}?api_key=${TMDB_API_KEY}`);
            }

            if (response.ok) {
              const data = await response.json();
              if (data.poster_path) {
                return {
                  ...movie,
                  poster: `${TMDB_IMAGE_BASE}/w92${data.poster_path}`,
                  poster_path: data.poster_path
                };
              }
            } else {
              newFailed.add(movie.tmdbId); // Mark as failed if both movie and TV endpoints fail
            }
          } catch (e) {
            console.error("Failed to fetch poster for search suggestion:", movie.title, e);
            if (movie.tmdbId) newFailed.add(movie.tmdbId);
          }
        }
        return movie;
      }));

      setFailedPosters(newFailed); // Update failed posters state

      // Only update state if there are actual changes to image URLs
      const hasChanges = updatedMovies.some((m, i) => m.poster !== searchSuggestions[i].poster);
      if (hasChanges) {
        setSearchSuggestions(updatedMovies);

        // Optional: Update the main cache too so we don't need to fetch again later
        updatedMovies.forEach(m => {
          if (m.poster && !m.poster.includes('via.placeholder')) {
            // Update allMoviesList in place (careful with state mutation, but for cache it's okay-ish or we can ignore)
            const idx = allMoviesList.findIndex(am => am.tmdbId === m.tmdbId);
            if (idx !== -1) {
              allMoviesList[idx].poster = m.poster;
            }
          }
        });
      }
    };

    // Debounce slightly to avoid rapid firing
    const timer = setTimeout(fetchMissingPosters, 200);
    return () => clearTimeout(timer);
  }, [searchSuggestions, triedPosterFetch, allMoviesList]);

  const handleSelectMovie = async (movie: Movie) => {
    setSelectedMovie(movie)
    setSearchSuggestions([]); // Clear suggestions on select
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
    const backendId = Object.keys(moviesCache).find(key => moviesCache[key].tmdbId === movie.tmdbId)
    if (backendId) {
      try {
        const response = await fetch(`${API_BASE}/similar/${backendId}?k=6`)
        const data = await response.json()
        const recs = data.similar_items.map((item: any) => {
          const m = moviesCache[String(item.item_id)]
          if (m) {
            return {
              ...m,
              id: m.tmdbId, // Ensure ID matches, though Movie uses tmdbId
              poster: m.poster,
              // score: item.score // Ignoring extra prop
            } as Movie
          }
          return null
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

  const searchInputRef = useRef<HTMLInputElement>(null);

  const handleSearchIconClick = () => {
    if (searchQuery.length > 0) {
      performSearch(searchQuery);
    } else {
      searchInputRef.current?.focus();
    }
  };

  const performSearch = async (query: string) => {
    if (!query) return;
    setIsSearching(true);
    setCurrentView('results');
    setSearchResults([]); // Clear previous

    try {
      const res = await fetch(`${API_BASE}/movies/search?q=${encodeURIComponent(query)}&limit=40`);
      const data = await res.json();

      if (data.results) {
        const formatted = data.results.map((m: any) => {
          let fullPoster = null;
          if (m.poster && m.poster.startsWith('http')) {
            fullPoster = m.poster;
          } else if (m.poster_path) {
            fullPoster = `${TMDB_IMAGE_BASE}/w500${m.poster_path}`;
          } else if (m.poster && m.poster.startsWith('/')) {
            fullPoster = `${TMDB_IMAGE_BASE}/w500${m.poster}`;
          }

          return {
            ...m,
            poster: fullPoster,
            id: m.tmdbId || m.item_id
          };
        });
        setSearchResults(formatted);
      }
    } catch (e) {
      console.error("Search failed", e);
    } finally {
      setIsSearching(false);
    }
  }

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


              <div className="w-full max-w-2xl relative group z-[60]">
                {/* Search Container with Layout Animation */}
                <motion.div
                  layout
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className="relative"
                >
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-primary to-rose-600 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
                  <input
                    ref={searchInputRef}
                    type="text"
                    placeholder="Search movies, genres..."
                    value={searchQuery}
                    onChange={(e) => handleSearch(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearchIconClick()}
                    className={`relative z-10 w-full h-12 md:h-14 pl-6 pr-12 bg-neutral-900/80 backdrop-blur-xl border border-white/10 text-base md:text-lg text-white placeholder:text-neutral-500 focus:outline-none focus:ring-1 focus:ring-primary/50 shadow-2xl transition-all ${searchSuggestions.length > 0 ? 'rounded-t-2xl rounded-b-none border-b-0' : 'rounded-2xl'}`}
                  />
                  <Search
                    onClick={handleSearchIconClick}
                    className="absolute right-5 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400 cursor-pointer hover:text-primary transition-colors z-20"
                  />
                </motion.div>

                <AnimatePresence>
                  {searchSuggestions.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: -10, scaleY: 0.95 }}
                      animate={{ opacity: 1, y: 0, scaleY: 1 }}
                      exit={{ opacity: 0, y: -10, scaleY: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="absolute top-full left-0 right-0 bg-neutral-900/80 backdrop-blur-xl border border-t-0 border-white/10 rounded-b-2xl overflow-hidden shadow-2xl z-50 transform origin-top"
                    >
                      {searchSuggestions.map((movie) => {
                        // Robust Poster Logic with Fallback for Suggestions
                        const posterSrc = movie.poster?.startsWith('http')
                          ? movie.poster
                          : movie.poster_path
                            ? `${TMDB_IMAGE_BASE}/w92${movie.poster_path}`
                            : `https://via.placeholder.com/92x138?text=${encodeURIComponent(movie.title)}`;

                        return (
                          <motion.div
                            key={movie.tmdbId}
                            onClick={() => handleSelectMovie(movie)}
                            whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.1)', scale: 1.02, x: 5 }}
                            transition={{ duration: 0.2 }}
                            className="flex items-center gap-4 p-4 cursor-pointer border-b border-white/5 last:border-0 group/item"
                          >
                            <div className="w-12 h-16 shrink-0 rounded-md overflow-hidden bg-neutral-800 shadow-md relative">
                              <img
                                src={posterSrc}
                                alt={movie.title}
                                className="w-full h-full object-cover"
                                onError={(e) => {
                                  (e.target as HTMLImageElement).src = `https://via.placeholder.com/92x138?text=${encodeURIComponent(movie.title)}`;
                                }}
                              />
                            </div>
                            <div className="flex-1 min-w-0">
                              <h4 className="text-white font-bold text-base truncate group-hover/item:text-primary transition-colors">{movie.title}</h4>
                            </div>
                          </motion.div>
                        );
                      })}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
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
            onBack={() => setCurrentView('home')}
            onSelectMovie={handleSelectMovie}
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
