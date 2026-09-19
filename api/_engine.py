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

import json
import re
import unicodedata
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


def _fold(s: str) -> str:
    """Lowercase + strip accents so 'Amelie' finds 'Amélie'."""
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


class Engine:
    def __init__(self, root: Path = ART):
        self.c = json.loads((root / "catalog.json").read_text())
        nz = np.load(root / "neighbors.npz")
        self.nbr, self.nsc = nz["ids"], nz["scores"].astype(np.float32)
        self.browse = json.loads((root / "browse.json").read_text())
        self.n = len(self.c["title"])

        # Search structures, built once per instance.
        self.fold = [_fold(t) for t in self.c["title"]]
        self.bayes = np.array(self.browse["bayes"], np.float32)
        self.tokens: dict[str, set[int]] = {}
        for i, t in enumerate(self.fold):
            for w in re.split(r"[^a-z0-9']+", t):
                if len(w) > 1 and w not in _STOP:
                    self.tokens.setdefault(w, set()).add(i)
        self.genre_of = [{g.lower() for g in gs} for gs in self.c["genres"]]
        self.tag_of = [set(ts) for ts in self.c["tags"]]
        self.genres_all = {g.lower() for gs in self.c["genres"] for g in gs}

        # Tag lookup, so "heist" or "time travel" finds films by theme.
        self.tag_index: dict[str, set[int]] = {}
        for i, ts in enumerate(self.c["tags"]):
            for t in ts:
                self.tag_index.setdefault(t, set()).add(i)
                for w in t.split():
                    if len(w) > 2:
                        self.tag_index.setdefault(w, set()).add(i)

        # Directors and cast, indexed by full name and by surname.
        self.people: dict[str, set[int]] = {}
        for i in range(self.n):
            names = (self.c.get("directors") or [[]] * self.n)[i] + \
                    (self.c.get("cast") or [[]] * self.n)[i]
            for nm in names:
                fn = _fold(nm)
                self.people.setdefault(fn, set()).add(i)
                parts = fn.split()
                if len(parts) > 1:
                    self.people.setdefault(parts[-1], set()).add(i)

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
        out = []
        for j, s in zip(ids, sc):
            d = self.item(int(j), float(s) / hi)   # 0..1 relative match strength
            d["shared_genres"] = sorted(self.genre_of[i] & self.genre_of[int(j)])
            d["shared_tags"] = sorted(self.tag_of[i] & self.tag_of[int(j)])[:3]
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
            out["genres"] = sorted(gs)
        return out

    def search(self, q: str, limit: int = 40) -> tuple[list[dict], dict]:
        """
        Structural terms (genre, language, decade) constrain the result set;
        whatever is left of the query is matched as free text against titles,
        people and tags. Keeping those two roles apart is what stops a query
        like "80s horror" from returning The Rocky Horror Picture Show.
        """
        f = _fold(q).strip()
        if not f:
            return [], {}
        filt = self.parse(q)
        want_g = set(filt.get("genres", []))
        ymin, ymax = filt.get("year_min"), filt.get("year_max")
        lang = filt.get("lang")

        # Strip the words already consumed as structure, so they don't double as title text.
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

        free = [w for w in re.split(r"[^a-z0-9']+", f)
                if len(w) > 1 and w not in _STOP and not _structural(w)
                and not re.fullmatch(r"(19|20)?\d0s|(19|20)\d{2}", w)]
        free_text = " ".join(free)

        scores: dict[int, float] = {}

        def bump(i: int, v: float):
            scores[i] = scores.get(i, 0.0) + v

        if free_text:
            for i, t in enumerate(self.fold):
                if t == free_text:
                    bump(i, 300)
                elif t.startswith(free_text):
                    bump(i, 190)
                elif free_text in t:
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

        # Structure-only query ("90s sci-fi", "korean thrillers"): everything that
        # satisfies the constraints is a candidate, ranked by quality.
        if not scores:
            if not (want_g or lang or ymin or ymax):
                return [], filt
            for i in range(self.n):
                bump(i, 10)

        out = []
        for i, s0 in scores.items():
            if want_g and not (self.genre_of[i] & want_g):
                continue
            if lang and self.c["lang"][i] != lang:
                continue
            y = self.c["year"][i]
            if ymin and (y is None or y < ymin):
                continue
            if ymax and (y is None or y > ymax):
                continue
            if free_text and s0 < 18:
                continue
            s0 += 26 * self.bayes[i]          # quality prior breaks ties
            if not self.c["poster"][i]:
                s0 -= 14
            out.append((i, s0))
        out.sort(key=lambda x: -x[1])
        return [self.item(i, s) for i, s in out[:limit]], filt

    def suggest(self, q: str, limit: int = 7) -> list[dict]:
        f = _fold(q).strip()
        if len(f) < 2:
            return []
        pre, sub = [], []
        for i, t in enumerate(self.fold):
            if t.startswith(f):
                pre.append(i)
            elif f in t:
                sub.append(i)
        pre.sort(key=lambda i: -self.bayes[i])
        sub.sort(key=lambda i: -self.bayes[i])
        return self.items((pre + sub)[:limit])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return Engine()
