"""
CINEMIND serving engine.

Everything expensive happens in pipelines/build_artifacts.py. At request time this
module only does dictionary and array lookups, so a similar-items call is O(k)
rather than the O(n) scan (plus a blocking LLM round-trip) the previous API ran on
every request.

Loaded once per warm instance; on Vercel's Fluid Compute that cost is amortised
across all requests the instance serves.
"""
from __future__ import annotations

import bisect
import gzip
import json
import re
import sys
import unicodedata
from array import array
from functools import lru_cache
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent.parent / "data" / "artifacts"

_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for",
         "with", "is", "it", "movie", "movies", "film", "films", "show", "shows",
         "me", "some", "good", "best", "like", "about", "that", "recommend"}

# Words people search with that don't literally appear in the metadata.
_EXPAND = {
    "scifi": "sci-fi", "sci": "sci-fi", "futuristic": "sci-fi",
    "space": "sci-fi", "romantic": "romance", "love": "romance",
    "funny": "comedy", "hilarious": "comedy", "scary": "horror",
    "spooky": "horror", "creepy": "horror", "cartoon": "animation",
    "anime": "animation", "animated": "animation", "kids": "children",
    "family": "children", "detective": "mystery", "spy": "thriller",
    "superhero": "action", "war": "war", "western": "western",
    "bollywood": "hindi", "indian": "hindi",
}

_LANGS = {"english": "en", "hindi": "hi", "french": "fr", "spanish": "es",
          "korean": "ko", "japanese": "ja", "german": "de", "italian": "it",
          "chinese": "zh", "russian": "ru", "swedish": "sv", "danish": "da"}

_LANG_NAMES = {"en": "English", "fr": "French", "ja": "Japanese", "it": "Italian",
               "es": "Spanish", "de": "German", "fi": "Finnish", "zh": "Chinese",
               "sv": "Swedish", "cn": "Cantonese", "ru": "Russian", "ko": "Korean",
               "hi": "Hindi", "da": "Danish", "nl": "Dutch", "pt": "Portuguese",
               "pl": "Polish", "no": "Norwegian", "fa": "Persian", "he": "Hebrew",
               "ar": "Arabic", "cs": "Czech", "hu": "Hungarian", "tr": "Turkish",
               "th": "Thai", "el": "Greek", "ro": "Romanian", "sr": "Serbian",
               "bn": "Bengali", "is": "Icelandic", "tl": "Tagalog", "id": "Indonesian",
               "sh": "Serbo-Croatian", "vi": "Vietnamese", "ku": "Kurdish",
               "et": "Estonian", "bs": "Bosnian", "mr": "Marathi", "ka": "Georgian",
               "ta": "Tamil", "uk": "Ukrainian", "lv": "Latvian", "ca": "Catalan"}

_DECADE = re.compile(r"\b(?:(19|20)?(\d0))['’]?s\b")
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_WORD_DECADE = {"twenties": 1920, "thirties": 1930, "forties": 1940, "fifties": 1950,
                "sixties": 1960, "seventies": 1970, "eighties": 1980,
                "nineties": 1990, "noughties": 2000}


def _singular(w: str) -> str:
    """People type "thrillers" and "comedies"; the genre list is singular."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


# Phones type curly quotes: "isn’t" must read as "isn't", or the negation is missed.
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u2032": "'", "`": "'",
                         "\u201c": '"', "\u201d": '"'})


def _fold(s: str) -> str:
    """Lowercase, straighten quotes and strip accents, so 'Amelie' finds 'Amélie'."""
    s = s.lower().translate(_QUOTES)
    if s.isascii():
        return s                # nothing to decompose — most names and titles, a million at startup
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _key(s: str) -> str:
    """A title or query reduced for matching: folded, punctuation as spaces."""
    return re.sub(r"[^a-z0-9]+", " ", _fold(s)).strip()


_ARTICLE_LAST = re.compile(r"^(.*), (the|a|an|les|la|le|il|el|der|die|das|los|las|lo|gli|i)$")


def title_variants(title: str) -> set[str]:
    """
    Every form a person might type a MovieLens title in.

    MovieLens files many titles article-last with alternates in brackets:
    "Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il)" is
    "the good the bad and the ugly", and "Seven (a.k.a. Se7en)" is also "se7en".
    A leading English article is optional, so "french connection" still finds
    "The French Connection".
    """
    f = _fold(title)
    main = re.sub(r"\s*\(.*\)\s*$", "", f).strip() or f
    alts = [re.sub(r"^a\.k\.a\.\s*", "", a).strip() for a in re.findall(r"\(([^()]*)\)", f)]
    out: set[str] = set()
    for base in [main, *alts]:
        m = _ARTICLE_LAST.match(base)
        if m:
            base = f"{m.group(2)} {m.group(1)}"
        k = _key(base)
        if not k:
            continue
        out.add(k)
        m = re.match(r"^(?:the|a|an) (.+)$", k)
        if m:
            out.add(m.group(1))
    return out


def release_memory():
    """Hand memory freed after startup back to the OS. Parsing catalog.json peaks
    far above what is kept; glibc holds on to the difference unless asked. A
    no-op where there is no glibc (macOS)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class ArtifactError(RuntimeError):
    """The artifacts on disk disagree with each other; serving them would be wrong."""


# How a film's matches were found (neighbors.npz `basis`).
BASIS = ("collaborative", "hybrid", "content")
# Artifacts with one row per film, which must agree with catalog.json.
PER_FILM = {"semantic_items.npy", "tower_items.npy", "tower_user.npz", "neighbors.npz", "browse.json"}


def catalog_path(where: Path) -> Path:
    """The catalogue file in an artifact directory: catalog.json.gz as built, or the
    catalog.json of builds before it was compressed."""
    gz = where / "catalog.json.gz"
    return gz if gz.exists() else where / "catalog.json"


def load_catalog(where: Path) -> dict:
    """The columnar catalogue, from an artifact directory or a catalogue file."""
    f = catalog_path(where) if where.is_dir() else where
    if f.suffix == ".gz":
        # Streamed: the whole text is held once while it parses, as for a plain file.
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(f.read_text(encoding="utf-8"))


class Engine:
    def __init__(self, root: Path = ART):
        for f in (catalog_path(root), root / "neighbors.npz", root / "browse.json"):
            if not f.exists():
                raise ArtifactError(f"{f} is missing — run the build (see README, Running it locally).")
        self.c = load_catalog(root)
        nz = np.load(root / "neighbors.npz")
        self.nbr, self.nsc = nz["ids"], nz["scores"].astype(np.float32)
        self.n = len(self.c["title"])
        # Older graphs predate the basis label: everything in them was co-rating based.
        self.basis = nz["basis"] if "basis" in nz.files else np.zeros(self.n, np.int8)
        self.raters = nz["positive_raters"] if "positive_raters" in nz.files else None
        self.browse = json.loads((root / "browse.json").read_text())
        # A film served twice under two MovieLens ids keeps both item_ids (browsers store
        # them), but the second is an alias: it resolves by id and is never listed.
        alias = self.c.get("alias_of") or [None] * self.n
        self.hidden = np.array([a is not None for a in alias], bool)
        self.canonical = np.array([i if a is None else a for i, a in enumerate(alias)], np.int64)
        self._validate(root)
        self.rated = sum(1 for r in self.c["rc"] if r)
        # The same names, genres and tags recur across thousands of films; one
        # copy of each string instead of one per film, in tuples (the columns are
        # never modified, and every empty one is the same object).
        for col in ("genres", "tags", "directors", "cast", "countries"):
            if col in self.c:
                self.c[col] = [tuple(sys.intern(x) for x in v) if v else () for v in self.c[col]]
        for col in ("lang", "media"):
            self.c[col] = [sys.intern(x) if isinstance(x, str) else x for x in self.c[col]]

        # Search structures, built once per instance. Read-only from here on, so
        # they are stored compactly: tuples for per-film lists, shared frozensets
        # for genre sets, int arrays for posting lists.
        self.fold = [_fold(t) for t in self.c["title"]]
        self.keys = [() if self.hidden[i] else tuple(title_variants(t)) for i, t in enumerate(self.c["title"])]
        self.key_index: dict[str, list[int]] = {}
        for i, ks in enumerate(self.keys):
            for k in ks:
                self.key_index.setdefault(k, []).append(i)
        self.key_index = {k: array("i", v) for k, v in self.key_index.items()}
        self._key_sorted = sorted(self.key_index)
        self.bayes = np.array(self.browse["bayes"], np.float32)

        def post(index: dict[str, list[int]], key: str, i: int):
            """Films are visited in id order, so each posting list stays sorted and a
            repeat can only be the film just added — lists, not sets, while building."""
            ids = index.setdefault(key, [])
            if not ids or ids[-1] != i:
                ids.append(i)

        def compact(index: dict[str, list[int]]) -> dict[str, array]:
            return {k: array("i", v) for k, v in index.items()}

        tokens: dict[str, list[int]] = {}
        for i, t in enumerate(self.fold):
            if self.hidden[i]:
                continue
            for w in re.split(r"[^a-z0-9']+", t):
                if len(w) > 1 and w not in _STOP:
                    post(tokens, w, i)
        self.tokens = compact(tokens)
        del tokens
        shared: dict[frozenset, frozenset] = {}
        self.genre_of = [shared.setdefault(fs, fs) for fs in (frozenset(g.lower() for g in gs) for gs in self.c["genres"])]
        self.tag_of = self.c["tags"]          # tuples; intersected per response, a few at a time
        self.genres_all = {g.lower() for gs in self.c["genres"] for g in gs}
        # lowercase key -> the label as it should be displayed ("sci-fi" -> "Sci-Fi")
        self.genre_label = {g.lower(): g for gs in self.c["genres"] for g in gs}
        self._cats: dict | None = None
        self._sugg: dict | None = None

        # Tag lookup, so "heist" or "time travel" finds films by theme.
        tag_index: dict[str, list[int]] = {}
        for i, ts in enumerate(self.c["tags"]):
            if self.hidden[i]:
                continue
            for t in ts:
                post(tag_index, t, i)
                for w in t.split():
                    if len(w) > 2:
                        post(tag_index, w, i)
        self.tag_index = compact(tag_index)
        del tag_index

        # Directors and cast, indexed by full name and by surname.
        people: dict[str, list[int]] = {}
        folded: dict[str, str] = {}            # a name recurs in many films: fold it once
        for i in range(self.n):
            if self.hidden[i]:
                continue
            names = (self.c.get("directors") or [[]] * self.n)[i] + \
                    (self.c.get("cast") or [[]] * self.n)[i]
            for nm in names:
                fn = folded.get(nm)
                if fn is None:
                    fn = folded[nm] = _fold(nm)
                post(people, fn, i)
                parts = fn.split()
                if len(parts) > 1:
                    post(people, parts[-1], i)
        self.people = compact(people)
        del people
        release_memory()

    def _validate(self, root: Path):
        """Fail at startup, with the file to blame, rather than per request."""
        n = self.n
        short = [k for k, v in self.c.items() if isinstance(v, list) and len(v) != n]
        if short:
            raise ArtifactError(f"catalog.json: columns {short} do not have {n} rows")
        if self.nbr.ndim != 2 or self.nbr.shape[0] != n or self.nsc.shape != self.nbr.shape:
            raise ArtifactError(f"neighbors.npz is {self.nbr.shape} for {n} films — rebuild it")
        if self.nbr.size and not ((self.nbr >= 0) & (self.nbr < n)).all():
            raise ArtifactError("neighbors.npz points at films outside the catalogue")
        if not np.isfinite(self.nsc).all():
            raise ArtifactError("neighbors.npz has non-finite scores")
        if len(self.basis) != n:
            raise ArtifactError(f"neighbors.npz basis has {len(self.basis)} rows for {n} films")
        bad = [i for i, a in enumerate(self.c.get("alias_of") or []) if a is not None
               and not (0 <= a < n and self.c["alias_of"][a] is None)]
        if bad:
            raise ArtifactError(f"catalog.json: alias_of of film {bad[0]} is not a listed film")
        if len(self.browse.get("bayes", [])) != n:
            raise ArtifactError(f"browse.json ranks {len(self.browse.get('bayes', []))} films, catalogue has {n}")
        manifest = root / "manifest.json"
        if manifest.exists():
            for name, meta in json.loads(manifest.read_text()).get("artifacts", {}).items():
                if name in PER_FILM and meta.get("rows") != n:
                    raise ArtifactError(f"{name} was built for {meta['rows']} films; catalog.json has {n}")

    def basis_of(self, i: int) -> str:
        return BASIS[int(self.basis[i])]

    # -- shaping ----------------------------------------------------------
    def item(self, i: int, score: float | None = None) -> dict:
        c = self.c
        d = {
            "item_id": i,
            "title": c["title"][i],
            "year": c["year"][i],
            "genres": c["genres"][i],
            "tags": c["tags"][i][:5],
            "tmdbId": c["tmdbId"][i],
            "movieId": c["movieId"][i],
            "poster_path": c["poster"][i],
            "backdrop_path": c["backdrop"][i],
            "overview": c["overview"][i],
            "runtime": c["runtime"][i],
            "media_type": c["media"][i],
            "original_language": c["lang"][i],
            "rating_avg": c["ra"][i],
            "rating_count": c["rc"][i],
            "tmdb_vote": c["tv"][i],
            "directors": (c.get("directors") or [[]] * self.n)[i],
            "cast": (c.get("cast") or [[]] * self.n)[i][:5],
            "imdb_id": c["imdbId"][i] if "imdbId" in c else None,
            # Metadata-only films have no MovieLens ratings: matched by content, never by co-rating.
            "rating_backed": bool(c["rc"][i]),
        }
        if score is not None:
            d["score"] = round(float(score), 4)
        return d

    def items(self, ids) -> list[dict]:
        return [self.item(int(i)) for i in ids]

    # -- recommendations --------------------------------------------------
    def similar(self, i: int, k: int = 12) -> list[dict]:
        """Precomputed hybrid neighbours: item-item CF + two-tower + content."""
        ids, sc = self.nbr[i][:k], self.nsc[i][:k]
        if len(sc) == 0:
            return []
        hi = float(sc.max()) or 1.0
        basis = self.basis_of(i)
        out = []
        for j, s in zip(ids, sc):
            d = self.item(int(j), float(s) / hi)   # 0..1 relative match strength
            d["shared_genres"] = sorted(self.genre_of[i] & self.genre_of[int(j)])
            d["shared_tags"] = sorted(set(self.tag_of[i]) & set(self.tag_of[int(j)]))[:3]
            d["match_basis"] = basis
            out.append(d)
        return out

    def blend(self, seeds: list[int], k: int = 20) -> list[dict]:
        """Neighbourhood of several films at once — a taste profile."""
        agg: dict[int, float] = {}
        seen = set(seeds)
        for s in seeds:
            if not 0 <= s < self.n:
                continue
            row, sc = self.nbr[s], self.nsc[s]
            hi = float(sc.max()) or 1.0
            for j, v in zip(row, sc):
                j = int(j)
                if j in seen:
                    continue
                agg[j] = agg.get(j, 0.0) + float(v) / hi
        top = sorted(agg.items(), key=lambda x: -x[1])[:k]
        hi = top[0][1] if top else 1.0
        return [self.item(i, s / hi) for i, s in top]

    # -- search -----------------------------------------------------------
    def names_title(self, q: str) -> bool:
        """
        True when a query of two or more words is a film's title, or the start
        of one. Such a query is a title search, and none of its words may turn
        into a filter: "star wars" is not the War genre, "children of men" is not
        Children, and "the french connection" is not a French-language film.
        Single words ("horror") and real category phrases ("korean thrillers")
        are unaffected — no title starts with them. One exception: a year that is
        also a known film's whole title ("1917", "2012") is that film, not the
        year's films; "1990", which titles no such film, stays a year.
        """
        k = _key(q)
        if len(k.split()) < 2:
            return bool(re.fullmatch(r"(18|19|20)\d\d", k)) and \
                any((self.c["rc"][i] or 0) >= 100 for i in self.key_index.get(k, ()))
        if k in self.key_index:
            return True
        j = bisect.bisect_left(self._key_sorted, k + " ")
        return j < len(self._key_sorted) and self._key_sorted[j].startswith(k + " ")

    def parse(self, q: str) -> dict:
        """Deterministic intent parsing. No network call in the request path."""
        f = _fold(q)
        out: dict = {}
        d = _DECADE.search(f)
        wd = next((v for k, v in _WORD_DECADE.items() if k in f), None)
        if wd:
            out["year_min"], out["year_max"] = wd, wd + 9
        elif d:
            cent, dec = d.group(1), int(d.group(2))
            if cent:                       # "1990s"
                y = int(cent) * 100 + dec
            else:                          # bare "90s" / "80s" / "00s"
                y = 1900 + dec if dec >= 20 else 2000 + dec
            out["year_min"], out["year_max"] = y, y + 9
        else:
            ys = _YEAR.findall(f)
            if len(ys) == 1:
                out["year_min"] = out["year_max"] = int(ys[0])
            elif len(ys) >= 2:
                out["year_min"], out["year_max"] = min(map(int, ys)), max(map(int, ys))
        for name, code in _LANGS.items():
            if name in f:
                out["lang"] = code
                break
        words = [w for w in re.split(r"[^a-z0-9'-]+", f) if w]
        gs = set()
        for w in words:
            for cand in (w, _singular(w)):
                w2 = _EXPAND.get(cand, cand)
                if w2 in self.genres_all:
                    gs.add(w2)
                    break
        if ("sci" in words and "fi" in words) or "science fiction" in f or "scifi" in f:
            gs.add("sci-fi")
        if "noir" in f:
            gs.add("film-noir")
        if gs:
            out["genres"] = sorted(self.genre_label.get(g, g) for g in gs)
        return out

    def _candidates(self, q: str, filt: dict) -> tuple[dict[int, float], str]:
        """
        Free-text scoring.

        Structural terms (genre, language, decade) are stripped out first and
        become filters instead; whatever is left is matched against titles,
        people and tags. Keeping those two roles apart is what stops a query
        like "80s horror" from returning The Rocky Horror Picture Show.
        """
        f = _fold(q).strip()
        want_g = {g.lower() for g in filt.get("genres", [])}
        lang = filt.get("lang")

        # Words already consumed as structure must not double as title text.
        consumed = set(want_g)
        for w, g in _EXPAND.items():
            if g in want_g:
                consumed.add(w)
        consumed |= {k for k in _LANGS if k in f and _LANGS[k] == lang}
        consumed |= set(_WORD_DECADE)
        consumed |= {"movie", "movies", "film", "films", "show", "shows"}
        if "sci-fi" in want_g:
            # "sci-fi" splits into two tokens; a stray "fi" otherwise matches
            # every title containing those letters (Final Fantasy, First Men...)
            consumed |= {"sci", "fi", "scifi", "science", "fiction"}
        if "film-noir" in want_g:
            consumed |= {"noir"}

        def _structural(w: str) -> bool:
            """True when this word was already used as a genre, language or date."""
            if w in consumed:
                return True
            cand = _singular(w)
            return _EXPAND.get(cand, cand) in want_g or cand in consumed

        # A year or decade is dropped only when it became a filter: "1917", the film's
        # whole title, has to stay to be matched.
        dated = "year_min" in filt or "year_max" in filt
        free = [w for w in re.split(r"[^a-z0-9']+", f)
                if len(w) > 1 and w not in _STOP and not _structural(w)
                and not (dated and re.fullmatch(r"(19|20)?\d0s|(19|20)\d{2}", w))]
        free_text = " ".join(free)
        if not free_text:
            return {}, ""

        scores: dict[int, float] = {}

        def bump(i: int, v: float):
            scores[i] = scores.get(i, 0.0) + v

        # Title match on the whole phrase. With no structure consumed that is the
        # query as typed, stop words and all, so "war of the worlds" can match
        # its title exactly rather than as "war worlds".
        phrase = _key(q) if not filt else _key(free_text)
        for i, ks in enumerate(self.keys):
            if phrase in ks:
                # An exact title outranks any mix of partial matches: "oppenheimer" is
                # Nolan's film before a documentary whose cast lists J. Robert Oppenheimer.
                bump(i, 600)
            elif any(k.startswith(phrase) for k in ks):
                bump(i, 190)
            elif any(phrase in k for k in ks):
                bump(i, 130)
        for w in free:
            for i in self.tokens.get(w, ()):
                bump(i, 48)
            if len(w) >= 4:
                for tok, ids in self.tokens.items():
                    if tok != w and tok.startswith(w):
                        for i in ids:
                            bump(i, 18)
        # people: "christopher nolan", "tom hanks"
        for i in self.people.get(free_text, ()):
            bump(i, 210)
        if len(free) > 1:
            for w in free:
                for i in self.people.get(w, ()):
                    bump(i, 26)
        for w in free:
            for i in self.tag_index.get(w, ()):
                bump(i, 34)
        return scores, free_text

    # -- filtering --------------------------------------------------------
    def _keep(self, i: int, g: set, ymin, ymax, lang, media, skip: str = "") -> bool:
        """One item against the active filters, optionally ignoring one facet."""
        if self.hidden[i]:
            return False
        if g and skip != "genres" and not g <= self.genre_of[i]:
            return False
        if lang and skip != "lang" and self.c["lang"][i] != lang:
            return False
        if skip != "decade" and (ymin or ymax):
            y = self.c["year"][i]
            if y is None or (ymin and y < ymin) or (ymax and y > ymax):
                return False
        if media and self.c["media"][i] != media:
            return False
        return True

    def count(self, genres=None, lang=None, year_min=None, year_max=None) -> int:
        g = {x.lower() for x in (genres or [])}
        return sum(1 for i in range(self.n)
                   if self._keep(i, g, year_min, year_max, lang, None))

    def search(self, q: str = "", limit: int = 40, offset: int = 0,
               genres=None, year_min=None, year_max=None, lang=None,
               media=None, sort: str = "relevance", facets: bool = False) -> dict:
        """
        Query text and explicit filters are separate inputs.

        The query is parsed for structure ("korean thrillers" -> ko + thriller)
        and those become the *initial* filter selection, which the UI shows as
        chips. Anything the caller passes explicitly replaces that selection, so
        unticking a genre the query implied actually removes it — otherwise the
        filter bar and the search box would fight each other.
        """
        parsed = self.parse(q) if q.strip() and not self.names_title(q) else {}
        eff = dict(parsed)
        if genres is not None:
            eff["genres"] = list(genres)
        for key, val in (("year_min", year_min), ("year_max", year_max),
                         ("lang", lang), ("media", media)):
            if val is not None:
                eff[key] = val
        eff = {k: v for k, v in eff.items() if v not in (None, [], "")}

        scores, free_text = self._candidates(q, parsed) if q.strip() else ({}, "")
        g = {x.lower() for x in eff.get("genres", [])}
        ymin, ymax = eff.get("year_min"), eff.get("year_max")
        lg, md = eff.get("lang"), eff.get("media")

        if not scores:
            # Browsing a category rather than searching text: every film that
            # satisfies the constraints is a candidate, ranked by quality.
            if not (g or lg or ymin or ymax or md):
                return {"results": [], "filters": eff, "total": 0, "facets": None}
            pool = [i for i in range(self.n) if self._keep(i, g, ymin, ymax, lg, md)]
            ranked = [(i, 10.0 + 26 * float(self.bayes[i])
                       - (0 if self.c["poster"][i] else 14)) for i in pool]
        else:
            pool, ranked = [], []
            for i, s0 in scores.items():
                if not self._keep(i, g, ymin, ymax, lg, md):
                    continue
                if free_text and s0 < 18:
                    continue
                pool.append(i)
                s0 += 26 * float(self.bayes[i])       # quality prior breaks ties
                if not self.c["poster"][i]:
                    s0 -= 14
                ranked.append((i, s0))

        if sort == "rating":
            ranked.sort(key=lambda x: -float(self.bayes[x[0]]))
        elif sort == "newest":
            ranked.sort(key=lambda x: (-(self.c["year"][x[0]] or 0), -x[1]))
        elif sort == "oldest":
            ranked.sort(key=lambda x: (self.c["year"][x[0]] or 9999, -x[1]))
        elif sort == "title":
            ranked.sort(key=lambda x: self.fold[x[0]])
        else:
            ranked.sort(key=lambda x: -x[1])

        page = ranked[offset:offset + limit]
        out = {
            "results": [self.item(i, s) for i, s in page],
            "filters": eff,
            "total": len(ranked),
        }
        out["facets"] = self._facets(scores, free_text, g, ymin, ymax, lg, md) \
            if facets else None
        return out

    def _facets(self, scores, free_text, g, ymin, ymax, lg, md) -> dict:
        """
        Counts for each filter dimension, computed with the *other* filters
        applied but not its own — so a genre you have already picked doesn't
        zero out every alternative and strand you there.
        """
        base = list(scores) if scores else range(self.n)
        if scores and free_text:
            base = [i for i in base if scores[i] >= 18]

        gc: dict[str, int] = {}
        dc: dict[int, int] = {}
        lc: dict[str, int] = {}
        for i in base:
            if self._keep(i, g, ymin, ymax, lg, md):
                for x in self.c["genres"][i]:
                    gc[x] = gc.get(x, 0) + 1
            if self._keep(i, g, ymin, ymax, lg, md, skip="decade"):
                y = self.c["year"][i]
                if y:
                    d = (y // 10) * 10
                    dc[d] = dc.get(d, 0) + 1
            if self._keep(i, g, ymin, ymax, lg, md, skip="lang"):
                code = self.c["lang"][i]
                if code:
                    lc[code] = lc.get(code, 0) + 1

        allg = self._cats["genres"] if self._cats else \
            [{"value": v} for v in sorted(self.genre_label.values())]
        return {
            "genres": sorted(({"value": x["value"], "count": gc.get(x["value"], 0)}
                              for x in allg),
                             key=lambda x: -x["count"]),
            "decades": [{"value": k, "label": f"{k}s", "count": dc[k]}
                        for k in sorted(dc, reverse=True)],
            "languages": [{"value": k, "label": _LANG_NAMES.get(k, k.upper()),
                           "count": v}
                          for k, v in sorted(lc.items(), key=lambda x: -x[1])
                          if k != "xx"],
        }

    def categories(self) -> dict:
        """The browsable filter vocabulary — genres, decades and languages."""
        if self._cats is None:
            self._cats = self._facets({}, "", set(), None, None, None, None)
        return self._cats

    # -- autosuggest ------------------------------------------------------
    @staticmethod
    def _prefixed(keys: list[str], f: str) -> list[str]:
        """Every key that starts with f, from a sorted list — a typeahead runs on each keystroke."""
        return keys[bisect.bisect_left(keys, f):bisect.bisect_left(keys, f + chr(0x10FFFF))]

    def suggest(self, q: str, limit: int = 6) -> list[dict]:
        f = _key(q)
        if len(f) < 2:
            return []
        hit = {i for k in self._prefixed(self._key_sorted, f) for i in self.key_index[k]}
        # A known film whose whole title this is comes first: "2012" is the film, before
        # "2012: Supernova" — but an obscure namesake doesn't bury a well-known title.
        known = {i for i in self.key_index.get(f, ()) if (self.c["rc"][i] or 0) >= 100}
        pre = sorted(sorted(hit), key=lambda i: (i not in known, -self.bayes[i]))
        sub: list[int] = []
        if len(pre) < limit:                 # titles that only contain it, when beginnings run short
            sub = [i for i, ks in enumerate(self.keys) if i not in hit and any(f in k for k in ks)]
            sub.sort(key=lambda i: -self.bayes[i])
        return self.items((pre + sub)[:limit])

    def _suggestible(self):
        """People and themes worth offering as a category, sorted for prefix lookup; built once."""
        if self._sugg is None:
            names = [n for n, ids in self.people.items() if " " in n and len(ids) >= 3]
            tags = [t for t, ids in self.tag_index.items() if len(t) >= 4 and len(ids) >= 10
                    and t not in self.genres_all and "(" not in t and ")" not in t]
            by_word: dict[str, list[str]] = {}
            for n in names:
                for w in n.split():
                    by_word.setdefault(w, []).append(n)
            # Published whole, so a concurrent request never sees it half built.
            self._sugg = {"names": sorted(names), "words": sorted(by_word), "by_word": by_word,
                          "tags": sorted(tags), "order": {k: j for j, k in enumerate(self.people)},
                          "tag_order": {k: j for j, k in enumerate(self.tag_index)}}
        return self._sugg

    def suggest_categories(self, q: str, limit: int = 5) -> list[dict]:
        """
        Category completions, so typing "rom" offers *Romance* as a filter and
        "kor" offers *Korean*, rather than only films with those letters in the
        title.

        Each one carries the filters it applies, so selecting it runs exactly
        the same filtered search the chips do — there is no second code path
        that could disagree with the first.
        """
        f = _fold(q).strip()
        if len(f) < 2:
            return []
        gcount = {c["value"]: c["count"] for c in self.categories()["genres"]}
        lcount = {c["value"]: c["count"] for c in self.categories()["languages"]}
        sing = _singular(f)
        words = [w for w in re.split(r"[^a-z0-9'-]+", f) if w]
        aliased = {_EXPAND[w] for w in words
                   if w in _EXPAND and _EXPAND[w] in self.genres_all}
        if f.startswith("noir") or "noir" in words:
            aliased.add("film-noir")

        def _genre_prefix(w: str) -> str | None:
            """The genre a partial word is heading towards: "thr" -> Thriller."""
            if len(w) < 3:
                return None
            w2 = _EXPAND.get(_singular(w), _singular(w))
            for low, label in self.genre_label.items():
                if low.startswith(w) or low == w2:
                    return label
            return None

        genres, langs, people, tags = [], [], [], []

        for low, label in self.genre_label.items():
            if low.startswith(f) or low.startswith(sing) or low in aliased:
                genres.append({"kind": "genre", "label": label,
                               "sublabel": f"{gcount.get(label, 0):,} films",
                               "filters": {"genres": [label]},
                               "count": gcount.get(label, 0)})

        for name, code in _LANGS.items():
            hit = next((w for w in words if name.startswith(w) and len(w) >= 2), None)
            if not hit or not lcount.get(code):
                continue
            filt: dict = {"lang": code}
            label = name.title()
            rest = [_genre_prefix(w) for w in words if w != hit]
            rest = [x for x in rest if x]
            n = lcount[code]
            if rest:
                filt["genres"] = sorted(set(rest))
                label = f"{label} {' '.join(filt['genres'])}"
                n = self.count(filt["genres"], code)
            elif len(words) > 1:
                continue          # extra words we could not place — don't guess
            if not n:
                continue
            langs.append({"kind": "language", "label": label,
                          "sublabel": f"{n:,} films", "filters": filt, "count": n})

        # A query that already resolves to more than one constraint is worth
        # offering whole: "korean thrillers", "90s sci-fi".
        p = self.parse(q)
        if p.get("year_min") and (p.get("genres") or p.get("lang")):
            bits = []
            if p.get("lang"):
                bits.append(_LANG_NAMES.get(p["lang"], p["lang"]))
            bits += list(p.get("genres", []))
            bits.append(f"{p['year_min']}s" if p["year_min"] % 10 == 0
                        else str(p["year_min"]))
            n = self.count(p.get("genres"), p.get("lang"),
                           p.get("year_min"), p.get("year_max"))
            if n:
                langs.insert(0, {"kind": "filter", "label": " ".join(bits),
                                 "sublabel": f"{n:,} films", "filters": p,
                                 "count": 10 ** 6})

        # Full names only (surnames are duplicates of these), by name or by any of its words.
        sg = self._suggestible()
        names = set(self._prefixed(sg["names"], f))
        names.update(n for w in self._prefixed(sg["words"], f) for n in sg["by_word"][w])
        for name in sorted(names, key=sg["order"].get):
            ids = self.people[name]
            people.append({"kind": "person", "label": name.title(),
                           "sublabel": f"{len(ids)} films",
                           "filters": {"q": name}, "count": len(ids)})

        for tag in sorted(self._prefixed(sg["tags"], f), key=sg["tag_order"].get):
            ids = self.tag_index[tag]
            tags.append({"kind": "tag", "label": tag.title(),
                         "sublabel": f"{len(ids)} films",
                         "filters": {"q": tag}, "count": len(ids)})

        for b in (genres, langs, people, tags):
            b.sort(key=lambda c: -c["count"])
        # Interleave so one crowded kind can't take every slot.
        out, seen = [], set()
        for i in range(2):
            for b in (langs, genres, people, tags):
                if i >= len(b) or len(out) >= limit:
                    continue
                if b[i]["label"].lower() in seen:
                    continue
                seen.add(b[i]["label"].lower())
                out.append(b[i])
        return out


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return Engine()
