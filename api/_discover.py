"""
Natural-language discovery: describe a film and get films that fit.

    "movies that feel like Interstellar but are more emotional"
    "psychological thrillers with unreliable narrators and a shocking ending"
    "something uplifting under two hours for tonight"

A description is read into two kinds of thing:

  Constraints, applied as hard filters — runtime ("under two hours"), era
  ("90s", "after 2010", "recent", "classic"), language ("in French",
  "foreign"), genres ("thrillers") and exclusions ("nothing scary").

  Meaning, used for ranking — anchor films ("feels like Interstellar", "Heat
  meets Drive") and a mood, the words left once the constraints are taken out
  ("psychological … unreliable narrators … shocking ending"). "More X" leans
  towards X and "less X" away from it.

The mood is embedded with the same sentence model as every film's description
(see _semantic.py) and compared by cosine. Anchors contribute through the
two-tower taste space, their descriptions and the co-rating graph, so "feels
like Interstellar" means what Interstellar's audience also loved, not just
films that mention space. Exact tag matches add a little, and a quality prior
breaks ties. Each signal is z-scored over the films that pass the filters
before they are blended, so no one signal wins by having a wider range.

Every result says why it is there, and the reply says how the description was
read — including when a filter had to be relaxed because nothing fit it.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from _engine import _LANG_NAMES, _LANGS, ART, Engine, _fold, _key, _singular
from _semantic import dot, get_model, item_matrix, mm
from _tower import get_tower

_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "ninety": 90, "hundred": 100}

_DUR = re.compile(
    r"(?P<n>\d+(?:\.\d+)?|an?|one|two|three|four|five|ninety|hundred)"
    r"(?P<half>\s+and\s+an?\s+half)?\s*-?\s*"
    r"(?P<u>hours?|hrs?|h|minutes?|mins?|m)\b"
    r"(?P<half2>\s+and\s+an?\s+half)?"
    r"(?:\s*(?P<n2>\d+)\s*(?:minutes?|mins?|m)\b)?")
_MAX_WORDS = r"under|less than|shorter than|no more than|not more than|no longer than|at most|max(?:imum)?|below|within|up to|tops"
_MIN_WORDS = r"over|more than|longer than|at least|min(?:imum)?|above"

_YEAR = r"((?:19|20)\d{2})"
_DECADE = re.compile(r"\b(?:the\s+)?(?:(19|20)?(\d0))['’]?s\b")
_WORD_DECADE = {"twenties": 1920, "thirties": 1930, "forties": 1940, "fifties": 1950,
                "sixties": 1960, "seventies": 1970, "eighties": 1980, "nineties": 1990, "noughties": 2000}

_ANCHOR = re.compile(r"\b(?:feels?\s+like|feel\s+similar\s+to|similar\s+to|in\s+the\s+vein\s+of|reminds?\s+(?:me\s+)?of|"
                     r"comparable\s+to|along\s+the\s+lines\s+of|in\s+the\s+style\s+of|the\s+next|a\s+la|à\s+la|like|"
                     # "darker than Toy Story" — but not "longer than two hours".
                     r"than(?!\s+(?:\d|an?\s+hour|one|two|three|four|five|half|ninety)))\s+")
# Words that follow "like" without naming a film: "I'd like something uplifting".
_NOT_A_TITLE = {"something", "anything", "a", "an", "some", "to", "that", "this", "it", "i", "me", "watching",
                "watch", "those", "these", "more", "less", "movies", "movie", "films", "film", "stories",
                "story", "ones", "one", "when", "how", "what", "being", "having", "people", "someone"}

_FILLER = {"movies", "movie", "films", "film", "something", "anything", "show", "me", "give", "find", "recommend",
           "want", "wanna", "looking", "for", "a", "an", "some", "that", "which", "are", "is", "be", "i", "im",
           "i'd", "i'm", "would", "like", "to", "watch", "tonight", "today", "please", "but", "and", "the", "with",
           "feel", "feels", "of", "kind", "sort", "type", "about", "where", "who", "it", "its", "it's", "one", "ones",
           "has", "have", "got", "in", "on", "good", "great", "really", "very", "just", "maybe", "perhaps", "also",
           "even", "slightly", "bit", "lot", "much", "so", "too", "can", "could", "should", "my", "we", "us", "our",
           "from", "set", "during", "around", "era", "years", "decade", "made", "released", "sure", "mood",
           "idea", "know", "dunno", "than", "not", "no", "something's", "anything's"}

# Mood words, and the viewer tags that express them. The sentence model can't
# spell every word — "heist" and "uplifting" fall outside its vocabulary and
# split into noise ("he"+"##ist") — so a mood is also read through the tags
# viewers actually use, which this catalogue has thousands of.
_MOOD_LEX = {
    "uplifting": ["feel-good", "inspirational", "heartwarming", "uplifting", "feel good movie"],
    "feel-good": ["feel-good", "heartwarming", "feel good movie"], "feelgood": ["feel-good", "heartwarming"],
    "heartwarming": ["heartwarming", "feel-good", "touching"], "hopeful": ["inspirational", "feel-good"],
    "inspiring": ["inspirational", "inspiring"], "inspirational": ["inspirational", "inspiring"],
    "happy": ["feel-good", "fun"], "cheerful": ["feel-good", "fun"],
    "emotional": ["emotional", "touching", "moving", "bittersweet"], "moving": ["moving", "touching", "emotional"],
    "touching": ["touching", "moving", "emotional"], "sad": ["sad", "depressing", "bittersweet"],
    "tearjerker": ["sad", "emotional", "touching"], "bittersweet": ["bittersweet", "melancholy"],
    "melancholy": ["melancholy", "bittersweet"], "melancholic": ["melancholy", "bittersweet"],
    "dark": ["dark", "bleak", "disturbing", "grim"], "bleak": ["bleak", "dark", "depressing"],
    "grim": ["grim", "dark", "bleak"], "disturbing": ["disturbing", "dark"],
    "creepy": ["creepy", "atmospheric", "suspense"], "suspenseful": ["suspense", "tense", "intense"],
    "tense": ["tense", "suspense", "intense"], "intense": ["intense", "tense"], "gripping": ["intense", "suspense"],
    "cozy": ["heartwarming", "cute", "feel-good", "charming"], "cosy": ["heartwarming", "cute", "feel-good"],
    "cute": ["cute", "charming"], "charming": ["charming", "whimsical"], "whimsical": ["whimsical", "quirky"],
    "quirky": ["quirky", "whimsical"], "weird": ["weird", "bizarre", "surreal"], "surreal": ["surreal", "weird"],
    "romantic": ["romantic", "love story"], "mind-bending": ["mindfuck", "surreal", "twist ending"],
    "mindbending": ["mindfuck", "twist ending"], "twisty": ["twist ending", "plot twist"],
    "twist": ["twist ending", "plot twist"], "shocking": ["twist ending", "plot twist"],
    "thought-provoking": ["thought-provoking", "philosophical"], "philosophical": ["philosophical", "thought-provoking"],
    "smart": ["intelligent", "clever"], "clever": ["clever", "intelligent"], "intelligent": ["intelligent", "clever"],
    "slow": ["slow", "atmospheric"], "atmospheric": ["atmospheric"], "violent": ["violence", "violent", "gore"],
    "gory": ["gore"], "bloody": ["gore", "violence"], "epic": ["epic"],
    "beautiful": ["visually appealing", "visually stunning", "beautiful"], "stunning": ["visually stunning", "visually appealing"],
    "gorgeous": ["visually stunning", "visually appealing"], "nostalgic": ["nostalgic", "nostalgia"],
    "fun": ["fun", "funny"], "hilarious": ["hilarious", "funny"], "lonely": ["loneliness"],
    "unreliable": ["psychological", "twist ending"], "psychological": ["psychological"],
    "heist": ["heist"], "heists": ["heist"], "revenge": ["revenge"], "dystopian": ["dystopia"],
    # Comparatives, for "like X but funnier".
    "funnier": ["funny", "hilarious", "comedy"], "lighter": ["feel-good", "fun", "lighthearted"],
    "darker": ["dark", "bleak", "disturbing"], "scarier": ["scary", "creepy", "terrifying"],
    "sadder": ["sad", "depressing", "bittersweet"], "happier": ["feel-good", "heartwarming"],
    "weirder": ["weird", "surreal", "bizarre"], "smarter": ["intelligent", "clever"],
    "slower": ["slow", "atmospheric"], "faster": ["fast paced", "action packed"],
    "lighthearted": ["lighthearted", "feel-good", "fun"], "light-hearted": ["lighthearted", "feel-good", "fun"],
}
# Tags too generic to carry meaning on their own.
_WEAK_TAGS = {"good", "story", "plot", "ending", "time", "light", "travel", "great", "film", "movie"}

_EXCLUDE = re.compile(r"\b(?:not\s+in\s+the\s+mood\s+for|anything\s+but|no|not|without|nothing|non|avoid|except|"
                      r"isn'?t|aren'?t|never)\s+(?:a\s+|an\s+|too\s+|overly\s+|very\s+|any\s+|anything\s+|something\s+)?"
                      r"([a-z][\w-]*)")
_LESS = re.compile(r"\b(?:less|not\s+too|not\s+so|without\s+much|fewer|minus)\s+([a-z][\w-]*(?:\s+[a-z][\w-]*)?)")

# "New York" is not a request for recent films, nor "old man" for classics.
_RECENT = re.compile(r"\b(?:recent|newer|modern|latest|contemporary|new\s+(?:films?|movies?|releases?|ones?|stuff))\b")
_CLASSIC = re.compile(r"\b(?:classics?|older|vintage|golden\s+age|old[\s-](?:school|films?|movies?|ones?|hollywood))\b")
_BW = re.compile(r"\bblack\s+(?:and|&)\s+white\b|\bb&w\b")
_FOREIGN = re.compile(r"\b(?:foreign|subtitled|non-english|not\s+in\s+english|world\s+cinema|international)\b")
_SHORT = re.compile(r"\b(?:short|quick|brief)\b(?!\s+films?\b)")
_NOT_LONG = re.compile(r"\bnot\s+(?:too|very|overly|that)\s+long\b")
_LONG = re.compile(r"\b(?:long|epic|lengthy|sprawling)\b")


_GENRE_ALIAS = {"scifi": "sci-fi", "scary": "horror", "funny": "comedy", "comedic": "comedy",
                "romantic": "romance", "animated": "animation", "cartoon": "animation", "anime": "animation",
                "noir": "film-noir", "documentarie": "documentary", "musicals": "musical",
                "westerns": "western", "thriller": "thriller", "suspense": "thriller"}


def _minutes(m: re.Match) -> int:
    n = m.group("n")
    v = float(n) if n[0].isdigit() else float(_NUM[n])
    if m.group("half") or m.group("half2"):
        v += 0.5
    unit = m.group("u")
    mins = v * 60 if unit.startswith("h") else v
    if m.group("n2") and unit.startswith("h"):
        mins += int(m.group("n2"))
    return int(round(mins))


def _z(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """z-score over the films that pass the filters; zero when there's no spread."""
    sel = x[mask]
    if sel.size < 2:
        return np.zeros_like(x)
    sd = float(sel.std())
    return (x - float(sel.mean())) / sd if sd > 1e-9 else np.zeros_like(x)


@dataclass
class Reading:
    """How a description was read."""
    runtime_min: int | None = None
    runtime_max: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    lang: str | None = None
    not_lang: str | None = None
    genres: list[str] = field(default_factory=list)
    avoid_genres: list[str] = field(default_factory=list)
    anchors: list[int] = field(default_factory=list)
    mood: str = ""
    avoid: str = ""
    chips: list[dict] = field(default_factory=list)

    def chip(self, kind: str, label: str):
        self.chips.append({"kind": kind, "label": label})

    @property
    def constrained(self) -> bool:
        return any(v is not None for v in (self.runtime_min, self.runtime_max, self.year_min, self.year_max,
                                           self.lang, self.not_lang)) or bool(self.genres or self.avoid_genres)


class Discover:
    def __init__(self, e: Engine, root: Path = ART):
        self.e = e
        self.sem = get_model(root)
        self.X = item_matrix(root)
        self.tower = get_tower(root)
        self.TT = self.tower.V
        c = e.c
        self.runtime = np.array([r or 0 for r in c["runtime"]], np.float32)
        self.year = np.array([y or 0 for y in c["year"]], np.int32)
        self.lang = np.array([x or "" for x in c["lang"]], object)
        self.rc = np.array([x or 0 for x in c["rc"]], np.float32)
        order = np.argsort(e.bayes, kind="stable")
        self.quality = np.empty(e.n, np.float32)
        self.quality[order] = np.arange(e.n) / max(e.n - 1, 1)
        self.genre_mat = {g: np.array([g in gs for gs in e.genre_of]) for g in e.genres_all}
        self.tag_sets = [set(t.lower() for t in ts) for ts in c["tags"]]
        self.all_tags = {t for t in set().union(*self.tag_sets) if len(e.tag_index.get(t, ())) >= 3}
        self.lex = {w: [t for t in ts if t in self.all_tags] for w, ts in _MOOD_LEX.items()}

    # -- reading ------------------------------------------------------------
    def read(self, q: str) -> Reading:
        r = Reading()
        f = re.sub(r"\s+", " ", _fold(q)).strip(" ?!.")
        rest = f

        def take(span: str):
            nonlocal rest
            rest = rest.replace(span, " ", 1)

        # Anchor films come first, so their titles can't turn into constraints:
        # "feels like The French Connection" is not a French-language request.
        rest = self._anchors(f, r, take, rest)

        # Runtime.
        for m in re.finditer(rf"\b({_MAX_WORDS})\s+{_DUR.pattern}", rest):
            r.runtime_max = _minutes(m)
            take(m.group(0))
            break
        for m in re.finditer(rf"\b({_MIN_WORDS})\s+{_DUR.pattern}", rest):
            r.runtime_min = _minutes(m)
            take(m.group(0))
            break
        if r.runtime_max is None and r.runtime_min is None and _NOT_LONG.search(rest):
            r.runtime_max = 120
            take(_NOT_LONG.search(rest).group(0))
        if r.runtime_max is None and r.runtime_min is None:
            if _SHORT.search(rest):
                r.runtime_max = 100
                take(_SHORT.search(rest).group(0))
            elif _LONG.search(rest):
                r.runtime_min = 140
                take(_LONG.search(rest).group(0))
        if r.runtime_max:
            r.chip("runtime", f"Under {self._hm(r.runtime_max)}")
        if r.runtime_min:
            r.chip("runtime", f"Over {self._hm(r.runtime_min)}")

        # Era.
        m = re.search(rf"\bbetween\s+{_YEAR}\s+and\s+{_YEAR}\b", rest)
        if m:
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            r.year_min, r.year_max = a, b
            take(m.group(0))
        if r.year_min is None:
            m = re.search(rf"\b(after|since|post|newer than)\s+{_YEAR}\b", rest)
            if m:
                r.year_min = int(m.group(2)) + (1 if m.group(1) in ("after", "post", "newer than") else 0)
                take(m.group(0))
        if r.year_max is None:
            m = re.search(rf"\b(before|pre|older than|prior to|until|up to)\s+{_YEAR}\b", rest)
            if m:
                r.year_max = int(m.group(2)) - (1 if m.group(1) in ("before", "pre", "older than", "prior to") else 0)
                take(m.group(0))
        if r.year_min is None and r.year_max is None:
            wd = next((k for k in _WORD_DECADE if re.search(rf"\b{k}\b", rest)), None)
            m = _DECADE.search(rest)
            if wd:
                r.year_min = _WORD_DECADE[wd]
                take(wd)
            elif m:
                cent, dec = m.group(1), int(m.group(2))
                r.year_min = int(cent) * 100 + dec if cent else (1900 + dec if dec >= 20 else 2000 + dec)
                take(m.group(0))
            if r.year_min is not None:
                r.year_max = r.year_min + 9
        if r.year_min is None and r.year_max is None:
            m = re.search(rf"\b(?:in|from|made in|released in)\s+{_YEAR}\b", rest)
            if m:
                r.year_min = r.year_max = int(m.group(1))
                take(m.group(0))
        if r.year_min is None and r.year_max is None:
            if _BW.search(rest):
                r.year_max = 1959
                take(_BW.search(rest).group(0))
            elif _RECENT.search(rest):
                r.year_min = 2005
                take(_RECENT.search(rest).group(0))
            elif _CLASSIC.search(rest):
                r.year_max = 1979
                take(_CLASSIC.search(rest).group(0))
        if r.year_min and r.year_max and r.year_max - r.year_min == 9 and r.year_min % 10 == 0:
            r.chip("year", f"{r.year_min}s")
        elif r.year_min and r.year_max and r.year_min == r.year_max:
            r.chip("year", str(r.year_min))
        elif r.year_min and r.year_max:
            r.chip("year", f"{r.year_min}–{r.year_max}")
        elif r.year_min:
            r.chip("year", f"{r.year_min} or later")
        elif r.year_max:
            r.chip("year", f"{r.year_max} or earlier")

        # Language.
        if _FOREIGN.search(rest):
            r.not_lang = "en"
            take(_FOREIGN.search(rest).group(0))
            r.chip("language", "Not in English")
        else:
            for name, code in _LANGS.items():
                # "in French", "Korean thrillers", "French-language" — not "English countryside".
                m = re.search(rf"\b(?:in|from)\s+{name}\b|\b{name}(?:-language|\s+(?:language|films?|movies?|"
                              rf"cinema|thrillers?|comed(?:y|ies)|dramas?|horror|romances?|animation|crime|war|"
                              rf"westerns?|musicals?|documentar(?:y|ies)|classics?))\b", rest)
                if m:
                    r.lang = code
                    take(m.group(0).split()[0] if m.group(0).split()[0] == name else m.group(0))
                    r.chip("language", _LANG_NAMES.get(code, name.title()))
                    break

        # Exclusions before inclusions: "nothing scary" must not add Horror, and
        # "a heist that isn't too violent" must steer away from violence, not towards it.
        for m in list(_EXCLUDE.finditer(rest)):
            w = m.group(1)
            g = self._genre_word(w)
            if g:
                r.avoid_genres.append(g)
                take(m.group(0))
                r.chip("avoid", f"No {self.e.genre_label.get(g, g)}")
            elif w not in _FILLER and (w in self.lex or (w in self.all_tags and w not in _WEAK_TAGS)):
                r.avoid = (r.avoid + " " + w).strip()
                take(m.group(0))
                r.chip("avoid", f"{'Less' if re.search(r'too|overly|very', m.group(0)) else 'Not'} {w}")
        for m in list(_LESS.finditer(rest)):
            r.avoid = (r.avoid + " " + m.group(1)).strip()
            take(m.group(0))
            r.chip("avoid", f"Less {m.group(1)}")

        # Genres named outright.
        words = re.findall(r"[a-z][\w-]*", rest)
        for w in words:
            g = self._genre_word(w)
            if g and g not in r.genres and g not in r.avoid_genres:
                r.genres.append(g)
                rest = re.sub(rf"\b{re.escape(w)}\b", " ", rest, count=1)
        if re.search(r"\bsci\s+fi\b|\bscience\s+fiction\b", rest) and "sci-fi" not in r.genres:
            r.genres.append("sci-fi")
            rest = re.sub(r"\bsci\s+fi\b|\bscience\s+fiction\b", " ", rest)
        for g in r.genres:
            r.chip("genre", self.e.genre_label.get(g, g))

        # What's left is the mood.
        mood = [w for w in re.findall(r"[a-z0-9'][\w'-]*", rest) if w not in _FILLER and len(w) > 1]
        r.mood = " ".join(mood)
        if r.mood:
            r.chip("mood", r.mood)
        return r

    def _anchors(self, f: str, r: Reading, take, rest: str) -> str:
        """Find films named as anchors, longest title first, and remove them."""
        found = []
        for m in _ANCHOR.finditer(f):
            tail = f[m.end():].split()
            if not tail or tail[0] in _NOT_A_TITLE:
                continue
            hit = self._longest_title(tail)
            if hit:
                i, n = hit
                found.append((i, m.group(0) + " ".join(tail[:n])))
                # "like heat and drive", "like alien or aliens": a second title after a joiner.
                more = tail[n:]
                while len(more) > 1 and more[0] in ("and", "or", "&", "plus", "meets", "with", "mixed"):
                    more = more[1:] if more[0] != "mixed" else more[2:]
                    hit2 = self._longest_title(more)
                    if not hit2:
                        break
                    found.append((hit2[0], " ".join(more[:hit2[1]])))
                    more = more[hit2[1]:]
        m = re.match(r"^(.+?)\s+meets\s+(.+)$", f)
        if m and not found:
            a, b = self._longest_title(m.group(1).split(), full=True), self._longest_title(m.group(2).split())
            if a and b:
                found += [(a[0], m.group(1)), (b[0], " ".join(m.group(2).split()[:b[1]]))]
        if not found:
            # The whole description is a title: treat it as "films like it".
            i, how = self.find(f)
            if i is not None and how == "exact" and len(f.split()) <= 8:
                found.append((i, f))
        for i, span in found:
            if i not in r.anchors:
                r.anchors.append(i)
                r.chip("anchor", f"Like {self._title(i)}")
            rest = rest.replace(span, " ", 1)
        return rest

    def _longest_title(self, words: list[str], full: bool = False):
        """(film, words used) for the longest run of words that names a film."""
        for n in (range(len(words), 0, -1) if full else range(min(len(words), 9), 0, -1)):
            if full and n != len(words):
                break
            span = " ".join(words[:n])
            i, how = self.find(span)
            if i is not None and (how == "exact" or (how == "prefix" and n >= 2)):
                return i, n
        return None

    def find(self, span: str) -> tuple[int | None, str]:
        """
        A film named by the span: an exact title form, or the start of a
        well-known title. No fuzzy matching — that is for typos in chat, and
        here it would turn ordinary words into films. As in chat, a far better
        known title the span begins beats an obscure exact one ("lord of the
        rings" is Peter Jackson's).
        """
        e = self.e
        k = _key(span)
        if len(k) < 2:
            return None, ""
        rc = lambda i: e.c["rc"][i] or 0  # noqa: E731
        exact = sorted(e.key_index.get(k, []), key=lambda i: -rc(i))
        j = bisect.bisect_left(e._key_sorted, k + " ")
        pre = []
        while j < len(e._key_sorted) and e._key_sorted[j].startswith(k + " ") and len(pre) < 400:
            pre.extend(e.key_index[e._key_sorted[j]])
            j += 1
        pre = sorted(set(pre), key=lambda i: -rc(i))
        if exact:
            if pre and rc(pre[0]) >= 10_000 and rc(pre[0]) >= 10 * max(rc(exact[0]), 1):
                return pre[0], "prefix"
            return exact[0], "exact"
        if pre and rc(pre[0]) >= 1000:
            return pre[0], "prefix"
        return None, ""

    def _genre_word(self, w: str) -> str | None:
        """A genre named outright. Search's looser aliases ("love" -> Romance,
        "family" -> Children, "space" -> Sci-Fi) would turn moods into filters
        here — "a family drama" is not a children's film — so only these count."""
        for cand in (w, _singular(w)):
            g = _GENRE_ALIAS.get(cand, cand)
            if g in self.e.genres_all:
                return g
        return None

    def _title(self, i: int) -> str:
        t = re.sub(r"\s*\(.*\)\s*$", "", self.e.c["title"][i]).strip() or self.e.c["title"][i]
        m = re.match(r"^(.*), (The|A|An)$", t)
        return f"{m.group(2)} {m.group(1)}" if m else t

    @staticmethod
    def _hm(mins: int) -> str:
        h, m = divmod(int(mins), 60)
        return (f"{h}h {m}m" if m else f"{h} hours" if h > 1 else "1 hour") if h else f"{m} minutes"

    # -- ranking ------------------------------------------------------------
    def mask(self, r: Reading, genres: bool = True) -> np.ndarray:
        e = self.e
        m = np.ones(e.n, bool)
        if r.runtime_max:
            m &= (self.runtime > 0) & (self.runtime <= r.runtime_max)
        if r.runtime_min:
            m &= self.runtime >= r.runtime_min
        if r.year_min:
            m &= self.year >= r.year_min
        if r.year_max:
            m &= (self.year > 0) & (self.year <= r.year_max)
        if r.lang:
            m &= self.lang == r.lang
        if r.not_lang:
            m &= (self.lang != r.not_lang) & (self.lang != "")
        for g in r.avoid_genres:
            m &= ~self.genre_mat[g]
        if genres:
            for g in r.genres:
                m &= self.genre_mat[g]
        if r.anchors:
            m[r.anchors] = False
        return m

    def search(self, q: str, limit: int = 24) -> dict:
        e = self.e
        r = self.read(q)
        notes: list[str] = []
        m = self.mask(r)
        relaxed = False
        if r.genres and m.sum() < 12:
            m = self.mask(r, genres=False)
            relaxed = True
            notes.append(f"Few films are all of {', '.join(self.e.genre_label.get(g, g) for g in r.genres)} "
                         f"with your other filters, so genre counts as a preference rather than a rule.")
        if m.sum() == 0:
            return {"query": q, "understood": self._understood(r), "results": [], "total": 0,
                    "notes": notes + ["Nothing in the catalogue fits all of that. Try loosening the runtime or era."]}

        score = np.zeros(e.n, np.float32)
        direct, related = self._mood_tags(r.mood)
        mood_vec = self._mood_vector(r.mood, direct, related) if r.mood else None
        a_direct, a_related = self._mood_tags(r.avoid)
        avoid_vec = self._mood_vector(r.avoid, a_direct, a_related) if r.avoid else None
        anchor_tt = anchor_sem = None
        if r.anchors:
            # The anchors as a little history: who liked these, and what else did they like?
            anchor_tt = self.tower.user(r.anchors)
            anchor_sem = self.X[r.anchors].mean(0)
            anchor_sem /= np.linalg.norm(anchor_sem) or 1
            graph = np.zeros(e.n, np.float32)
            for a in r.anchors:
                sc = e.nsc[a]
                graph[e.nbr[a]] += sc / (float(sc.max()) or 1.0)
            score += 0.40 * _z(dot(self.TT, anchor_tt), m)
            score += 0.18 * _z(dot(self.X, anchor_sem), m)
            score += 0.30 * graph
            if mood_vec is not None:
                lean = anchor_sem + 1.15 * mood_vec
                lean /= np.linalg.norm(lean) or 1
                score += 0.42 * _z(dot(self.X, lean), m)
        elif mood_vec is not None:
            score += 0.80 * _z(dot(self.X, mood_vec), m)
        if avoid_vec is not None and avoid_vec.any():
            score -= 0.30 * _z(dot(self.X, avoid_vec), m)
        if mood_vec is None and not r.anchors:
            score += 0.9 * _z(self.quality, m)         # filters only: the best films that fit
        else:
            score += 0.22 * _z(self.quality, m)

        # Films viewers tagged with what the description names ("heist",
        # "twist ending") — or with the tags its mood words stand for.
        tag_boost = np.zeros(e.n, np.float32)
        for t in direct:
            tag_boost[list(e.tag_index[t])] += 0.8
        for t in related:
            tag_boost[list(e.tag_index[t])] += 0.35
        score += np.minimum(tag_boost, 1.2)
        hits = direct + [t for t in related if t not in direct]
        if relaxed:
            for g in r.genres:
                score += 0.6 * self.genre_mat[g]
        score -= np.where(self.rc < 20, 0.6, np.where(self.rc < 100, 0.2, 0.0)).astype(np.float32)

        score = np.where(m, score, -np.inf)
        k = int(min(limit, m.sum()))
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.argsort(-score[top])]
        hi, lo = float(score[top[0]]), float(score[top[-1]])
        results = []
        for i in top:
            d = e.item(int(i), (float(score[i]) - lo) / (hi - lo) if hi > lo else 1.0)
            d["why"] = self._why(int(i), r, mood_vec, hits)
            results.append(d)
        return {"query": q, "understood": self._understood(r), "results": results,
                "total": int(m.sum()), "notes": notes}

    def _mood_tags(self, mood: str) -> tuple[list[str], list[str]]:
        """(tags the mood names outright, tags its mood words stand for)."""
        words = mood.split()
        direct: list[str] = []
        for n in (3, 2, 1):
            for j in range(len(words) - n + 1):
                for t in {" ".join(words[j:j + n]), " ".join([*words[j:j + n - 1], _singular(words[j + n - 1])])}:
                    if t in self.all_tags and t not in _WEAK_TAGS and t not in self.e.genres_all \
                            and not any(t in o for o in direct):
                        direct.append(t)
        related: list[str] = []
        for w in words:
            for t in self.lex.get(w, []):
                if t not in direct and t not in related:
                    related.append(t)
        return direct[:4], related[:6]

    @lru_cache(maxsize=4096)
    def _proto(self, tag: str) -> np.ndarray:
        """A tag's prototype: the mean description of every film tagged with it."""
        v = self.X[list(self.e.tag_index[tag])].mean(0)
        return v / (np.linalg.norm(v) or 1)

    def _mood_vector(self, text: str, direct: list[str], related: list[str]) -> np.ndarray | None:
        """
        The mood as one vector: the sentence model's reading of the words, plus
        the prototypes of the tags they name or stand for. Words the model
        spells badly (split into short fragments) count for less.
        """
        if not text:
            return None
        known = sum(1 for w in text.split() if len(self.sem.tokenise(w)) == 1)
        v = self.sem.encode(text) * (0.6 if known >= max(1, len(text.split()) // 2) else 0.25)
        for t in direct:
            v = v + 1.0 * self._proto(t)
        for t in related:
            v = v + 0.55 * self._proto(t)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    @lru_cache(maxsize=20000)
    def _tag_vec(self, tag: str) -> np.ndarray:
        return self.sem.encode(tag)

    def _why(self, i: int, r: Reading, mood_vec, hits) -> list[str]:
        c = self.e.c
        why = []
        for a in r.anchors[:2]:
            shared = sorted(self.e.genre_of[i] & self.e.genre_of[a])
            if i in set(int(x) for x in self.e.nbr[a]):
                why.append(f"Loved by {self._title(a)}'s audience")
            elif shared:
                why.append(f"Shares {self.e.genre_label.get(shared[0], shared[0])} with {self._title(a)}")
        tags = c["tags"][i] or []
        named = [t for t in tags if t.lower() in set(hits)]
        why += [f"“{t}”" for t in named[:2]]
        if mood_vec is not None and len(why) < 3 and tags:
            sims = sorted(((float(mm(self._tag_vec(t.lower()), mood_vec)), t) for t in tags if t not in named),
                          reverse=True)
            why += [f"“{t}”" for s, t in sims[:2] if s >= 0.35][: 3 - len(why)]
        if r.runtime_max or r.runtime_min:
            rt = c["runtime"][i]
            if rt:
                why.append(self._hm(rt).replace(" hours", "h").replace(" minutes", "m"))
        if not why and c["ra"][i] is not None and (c["rc"][i] or 0) > 0:
            why.append(f"{c['ra'][i]:.1f}★ from {c['rc'][i]:,} ratings")
        return why[:3]

    def _understood(self, r: Reading) -> dict:
        return {"chips": r.chips, "anchors": [{"item_id": a, "title": self._title(a)} for a in r.anchors],
                "mood": r.mood or None, "avoid": r.avoid or None}
