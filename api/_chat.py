"""
The film assistant.

It answers from the catalogue and the recommendation graph, never from a
language model, so every fact it states — a director, a runtime, a rating, a
match — is one the API itself would return, and an answer costs milliseconds.

A message is read in three steps:

  1. What is being asked: about a film, who directed or stars in it, when it
     came out, how long it runs, whether it is any good, films like it (perhaps
     "but funnier"), films between two others, a person's films, or a category.
  2. Which film is meant. Titles are matched in every form MovieLens files
     them under, then by prefix, then by spelling distance; "it" and "that one"
     mean the film the conversation is already about, which the client sends
     back as `context`.
  3. The answer, built from the engine's data, with cards to open and a few
     follow-ups worth asking next.
"""
from __future__ import annotations

import difflib
import re

import numpy as np

from _engine import _STOP, Engine, _fold, _key

# -- vocabulary -------------------------------------------------------------

_PRONOUN = re.compile(r"^(?:it|that|this|that one|this one|that film|this film|that movie|"
                      r"this movie|the film|the movie|them|those|him|her)$")

_FILLER_LEAD = re.compile(r"^(?:the (?:movie|film)|movie|film|watching|the show)\s+")
_FILLER_TAIL = re.compile(r"\s+(?:movie|film|please|pls|thanks|thank you|for me|again|"
                          r"too|then|now|right|anyway|lol)$")

# (intent, pattern). Each pattern names the film span `x`. Order matters: the
# specific questions are tried before the general "about".
_ASK: list[tuple[str, re.Pattern]] = [(k, re.compile(p)) for k, p in [
    ("director", r"(?:who|which person) (?:directed|made|helmed) (?P<x>.+)"),
    ("director", r"(?:who is |who was |who's )?(?:the )?director of (?P<x>.+)"),
    ("director", r"(?:who is |who was |who's )?(?P<x>.+?)(?:'s| s) director"),
    ("director", r"^who directed$"),
    ("cast", r"who (?:is|was|are|were|stars?|starred|acts?|acted|plays?|played) in (?P<x>.+)"),
    ("cast", r"(?:the )?(?:cast|actors|stars) (?:of|in) (?P<x>.+)"),
    ("cast", r"who(?:'s| is) in (?P<x>.+)"),
    ("year", r"when (?:did|was|is) (?P<x>.+?) (?:come out|released|release|made|out|from)"),
    ("year", r"what year (?:is|was|did) (?P<x>.+?)(?: come out| released| made| from)?$"),
    ("year", r"(?:release (?:date|year)|year) (?:of|for) (?P<x>.+)"),
    ("year", r"how old is (?P<x>.+)"),
    ("runtime", r"how long is (?P<x>.+)"),
    ("runtime", r"(?:runtime|run time|running time|length|duration) (?:of|for) (?P<x>.+)"),
    ("runtime", r"how (?:many )?(?:minutes|hours) is (?P<x>.+)"),
    ("rating", r"(?:is|was) (?P<x>.+?) (?:any good|good|worth (?:it|watching|a watch|seeing)|"
               r"overrated|underrated|popular|rated well|well rated)$"),
    ("rating", r"how (?:good|popular|well rated|highly rated) is (?P<x>.+)"),
    ("rating", r"(?:rating|score|reviews?) (?:of|for) (?P<x>.+)"),
    ("rating", r"should i (?:watch|see) (?P<x>.+)"),
    ("rating", r"what do (?:people|viewers|you) think (?:of|about) (?P<x>.+)"),
    ("genre", r"what (?:genre|kind of (?:film|movie)|type of (?:film|movie)|sort of (?:film|movie)) is (?P<x>.+)"),
    ("is_genre", r"^(?:is|was) (?P<x>.+?) (?:a |an )?(?P<g>horror|comedy|thriller|drama|romance|romantic|sci-fi|"
                 r"science fiction|western|war|musical|documentary|crime|mystery|fantasy|animated|animation|"
                 r"action|adventure|scary|funny|for kids|kids|family)(?: film| movie| one)?$"),
    ("similar", r"(?:something|anything|films?|movies?|more|others?|stuff|shows?)? ?"
                r"(?:like|similar to|such as|in the vein of|comparable to|along the lines of|close to) (?P<x>.+)"),
    ("similar", r"if i (?:liked|loved|enjoyed|like|love|enjoy) (?P<x>.+?)(?:,? (?:what|then|which|recommend).*)?$"),
    ("similar", r"^i (?:really )?(?:liked|loved|enjoyed|adored|like|love|enjoy) (?P<x>.+?)"
                r"(?:,? (?:what|so|now|any|then|recommend|suggest|give|show).*)?$"),
    ("similar", r"(?:recommendations?|matches|alternatives?) (?:for|to) (?P<x>.+)"),
    ("similar", r"(?:films?|movies?|something|anything)? ?(?:between|that mix|that blend|mixing|blending) (?P<x>.+)"),
    ("similar", r"(?:what|which) (?:should i watch|to watch) after (?P<x>.+)"),
    ("about", r"(?:tell me|talk|tell us|info|information|details|know|anything) (?:about|on) (?P<x>.+)"),
    ("about", r"what(?:'s| is| was) (?P<x>.+?) about"),
    ("about", r"(?:plot|synopsis|summary|story|premise) (?:of|for) (?P<x>.+)"),
    ("about", r"(?:describe|explain|summarise|summarize) (?P<x>.+)"),
    ("about", r"(?:have you (?:seen|heard of)|do you know) (?P<x>.+)"),
    ("about", r"what(?:'s| is) (?P<x>.+)"),
]]

# "…but funnier": how to narrow a list of matches.
_MODS: list[tuple[re.Pattern, str, object]] = [(re.compile(p), kind, val) for p, kind, val in [
    (r"\b(?:funn(?:y|ier)|lighter|comed(?:y|ies)|laughs?|light ?hearted|feel ?good)\b", "genre", "comedy"),
    (r"\b(?:scar(?:y|ier)|horror|terrifying|creepier|creepy)\b", "genre", "horror"),
    (r"\b(?:darker|grittier|dark|gritty|bleaker|more intense|tense|tenser)\b", "dark", None),
    (r"\b(?:romantic|romance|love story)\b", "genre", "romance"),
    (r"\b(?:animated|animation|cartoons?|anime)\b", "genre", "animation"),
    (r"\b(?:for kids|kid ?friendly|family|for children|children)\b", "genre", "children"),
    (r"\b(?:more action|action|explosive)\b", "genre", "action"),
    (r"\b(?:sci-?fi|science fiction|futuristic)\b", "genre", "sci-fi"),
    (r"\b(?:documentar(?:y|ies)|true story)\b", "genre", "documentary"),
    (r"\b(?:older|classic|old|vintage|before it)\b", "older", None),
    (r"\b(?:newer|recent|modern|more recent|after it)\b", "newer", None),
    (r"\b(?:shorter|short|quick|under (?:2|two) hours)\b", "shorter", None),
    (r"\b(?:better rated|highly rated|better|acclaimed)\b", "better", None),
    (r"\bin (english|hindi|french|spanish|korean|japanese|german|italian|chinese|russian|swedish|danish)\b", "lang", None),
    (r"\b(?:foreign|not in english|non-english|subtitled)\b", "foreign", None),
]]

_LANG_CODE = {"english": "en", "hindi": "hi", "french": "fr", "spanish": "es", "korean": "ko",
              "japanese": "ja", "german": "de", "italian": "it", "chinese": "zh",
              "russian": "ru", "swedish": "sv", "danish": "da"}
_LANG_NAME = {v: k.title() for k, v in _LANG_CODE.items()}

_DARK = {"thriller", "crime", "horror", "war", "film-noir", "mystery"}

_GREETING = re.compile(r"^(?:hi|hello|hey|hiya|yo|hola|good (?:morning|afternoon|evening)|sup|what'?s up)\b(?: there)?[!. ]*$")
_THANKS = re.compile(r"^(?:thanks|thank you|thx|cheers|ty|great|awesome|nice|cool|perfect)\b")
_HELP = re.compile(r"^(?:help|what can you do|how does this work|who are you|what are you)\b")
_OPEN = re.compile(r"\b(?:what should i watch|what to watch|recommend (?:me )?(?:something|anything|a |some |films?|movies?)|"
                   r"suggest (?:something|anything|a |some )|i'?m bored|surprise me|any (?:suggestions|recommendations)|"
                   r"give me (?:something|a film|a movie))")

_PERSON_ASK = re.compile(r"^(?:(?:films?|movies?|work|anything) (?:by|from|with|starring|featuring|directed by)|"
                         r"directed by|starring|featuring) (?P<x>.+)$")
_PERSON_TAIL = re.compile(r"^(?P<x>.+?)(?:'s| s)? (?:films?|movies?|filmography)$")

# "Funnier" is relative to a film; "funny" is a mood of its own.
_COMPARATIVE = re.compile(r"\b(?:but|more|less|ones|darker|older|newer|shorter|better|lighter|grittier|"
                          r"scarier|funnier|tenser|bleaker|creepier|similar)\b")

_GENRE_WORD = {"romantic": "romance", "science fiction": "sci-fi", "animated": "animation",
               "scary": "horror", "funny": "comedy", "for kids": "children", "kids": "children",
               "family": "children"}

# Words that mark a message as a description of what to watch.
_DESCRIBE = re.compile(r"\b(?:movies?|films?|something|anything|stories|story|about|where|vibes?|mood|feels?|"
                       r"set in|with an?|with some|watch|tonight|kind of|type of|that are|that is)\b")

# Words that show free text is a request, not a title or a theme.
_CHATTER = _STOP | {"tell", "give", "show", "want", "need", "find", "can", "could", "would",
                    "you", "your", "what", "which", "who", "how", "why", "please", "joke",
                    "watch", "watching", "tonight", "today", "something", "anything", "i", "im"}


def _clean(span: str, tail: bool = True) -> str:
    """Trim quotes and filler; with tail=False, keep a trailing "film" or "movie"."""
    s = span.strip().strip("\"'“”‘’?!.,;: ")
    s = _FILLER_LEAD.sub("", s)
    for _ in range(3 if tail else 0):
        s2 = _FILLER_TAIL.sub("", s).strip("?!., ")
        if s2 == s:
            break
        s = s2
    return s.strip()


def _display(title: str) -> str:
    """'Postman, The (Postino, Il)' -> 'The Postman'."""
    t = re.sub(r"\s*\(.*\)\s*$", "", title).strip() or title
    m = re.match(r"^(.*), (The|A|An)$", t)
    return f"{m.group(2)} {m.group(1)}" if m else t


def _join(xs: list[str], word: str = "and") -> str:
    xs = [x for x in xs if x]
    if len(xs) <= 1:
        return "".join(xs)
    return f"{', '.join(xs[:-1])} {word} {xs[-1]}"


def _runtime(m: int | None) -> str | None:
    if not m:
        return None
    h, r = divmod(int(m), 60)
    return (f"{h}h {r}m" if r else f"{h}h") if h else f"{r}m"


class Assistant:
    def __init__(self, e: Engine):
        self.e = e
        c = e.c
        # Percentile of each film's Bayesian rating, for "is it any good?".
        order = np.argsort(e.bayes, kind="stable")
        self.pct = np.empty(e.n, np.float32)
        self.pct[order] = np.arange(e.n) / max(e.n - 1, 1)
        # Who directed what, and who acted in what — kept apart for roles.
        self.directed: dict[str, set[int]] = {}
        self.acted: dict[str, set[int]] = {}
        self.person_name: dict[str, str] = {}
        for i in range(e.n):
            for nm in c["directors"][i] or []:
                self.directed.setdefault(_fold(nm), set()).add(i)
                self.person_name[_fold(nm)] = nm
            for nm in c["cast"][i] or []:
                self.acted.setdefault(_fold(nm), set()).add(i)
                self.person_name[_fold(nm)] = nm
        self._all_keys = list(e.key_index)
        self._discover = None
        self._people = list(self.person_name)

    # -- resolving what is meant ------------------------------------------
    def find_film(self, span: str) -> tuple[int | None, str]:
        """
        (film, how) for a span of text, where how is "exact", "prefix" or "fuzzy".

        Exact title forms win, then titles the span begins, then close spellings;
        ties go to the film more people have rated, so "Heat" is Michael Mann's,
        not the 1972 one. A prefix only counts for a well-known film: "lord of
        the rings" may mean The Fellowship of the Ring, but "get out" must not
        quietly become Get Out Your Handkerchiefs (1978).
        """
        # "A Serbian Film" and "Date Movie" end in the words that are usually
        # filler ("the film inception"), so the span is tried as typed first.
        light, full = _clean(span, tail=False), _clean(span)
        if light != full and _key(light) in self.e.key_index:
            return self._find(light)
        return self._find(full)

    def _find(self, span: str) -> tuple[int | None, str]:
        e = self.e
        year = None
        m = re.search(r"\(?\b((?:19|20)\d{2})\)?$", span)
        if m and len(span) > 5:
            year, span = int(m.group(1)), span[:m.start()].strip(" ,(")
        k = _key(span)
        if len(k) < 2 or _PRONOUN.match(k):
            return None, ""

        def rank(ids):
            ids = [i for i in ids if year is None or abs((e.c["year"][i] or 0) - year) <= 1]
            return sorted(set(ids), key=lambda i: -(e.c["rc"][i] or 0))

        exact = rank(e.key_index.get(k, []))
        pre = rank([i for kk, ids in e.key_index.items()
                    if kk.startswith(k + " ") or (len(k) >= 6 and kk.startswith(k))
                    for i in ids]) if len(k) >= 4 else []
        if exact:
            # "lord of the rings" is exactly the 1978 animated film, but someone
            # asking almost certainly means Peter Jackson's: when a title the
            # query begins is ten times better known, it wins.
            ex_rc = e.c["rc"][exact[0]] or 0
            whole = [i for i in pre if any(kk.startswith(k + " ") for kk in e.keys[i])]
            pop = e.c["rc"][whole[0]] or 0 if whole else 0
            if year is None and whole and pop >= 10_000 and pop >= 10 * max(ex_rc, 1):
                return whole[0], "prefix"
            return exact[0], "exact"
        if pre and (e.c["rc"][pre[0]] or 0) >= 1000:
            return pre[0], "prefix"
        if len(k) >= 5:
            close = difflib.get_close_matches(k, self._all_keys, n=6, cutoff=0.86)
            fuzzy = rank([i for kk in close for i in e.key_index[kk]])
            if fuzzy:
                return fuzzy[0], "fuzzy"
        return None, ""

    def maybe(self, span: str) -> list[int]:
        """Films the span might have meant, for "did you mean"."""
        e = self.e
        k = _key(_clean(span))
        if len(k) < 2:
            return []
        pre = [i for kk, ids in e.key_index.items() if kk.startswith(k) for i in ids]
        close = difflib.get_close_matches(k, self._all_keys, n=8, cutoff=0.7)
        ids = pre + [i for kk in close for i in e.key_index[kk]]
        return sorted(set(ids), key=lambda i: -(e.c["rc"][i] or 0))[:4]

    def find_person(self, span: str) -> str | None:
        k = _fold(_clean(span))
        if k in self.directed or k in self.acted:
            return k
        if len(k) >= 6 and " " in k:
            close = difflib.get_close_matches(k, self._people, n=1, cutoff=0.88)
            if close:
                return close[0]
        return None

    # -- facts ------------------------------------------------------------
    def name(self, i: int) -> str:
        y = self.e.c["year"][i]
        t = _display(self.e.c["title"][i])
        return f"{t} ({y})" if y else t

    def verdict(self, i: int) -> str:
        c = self.e.c
        ra, rc = c["ra"][i], c["rc"][i] or 0
        if ra is None or rc == 0:
            return f"Nobody has rated {self.name(i)} on MovieLens yet, so I can't say how it lands."
        p = float(self.pct[i])
        if p >= 0.97:
            v = "one of the best-loved films in the whole catalogue"
        elif p >= 0.85:
            v = "very well liked"
        elif p >= 0.6:
            v = "well liked"
        elif p >= 0.35:
            v = "liked, with reservations"
        elif p >= 0.15:
            v = "divisive"
        else:
            v = "poorly rated"
        line = (f"MovieLens viewers rate it {ra:.1f} out of 5 across {rc:,} ratings, "
                f"ahead of {int(p * 100)}% of the {self.e.n:,} films here: {v}.")
        if rc < 50:
            line += f" That's only {rc} ratings, so treat it as a rough read."
        return line

    def two_hop(self, i: int) -> list[tuple[int, float]]:
        """
        Neighbours of neighbours, scored through the graph. Used when a filter
        ("but funnier") leaves too few of a film's own 24 to be useful: the next
        ring out is still grounded in the same co-rating graph, not a guess.
        """
        e = self.e
        agg: dict[int, float] = {}
        row, sc = e.nbr[i], e.nsc[i]
        hi = float(sc.max()) or 1.0
        for j, s in zip(row, sc):
            j, w = int(j), float(s) / hi
            agg[j] = max(agg.get(j, 0.0), w)
            r2, s2 = e.nbr[j], e.nsc[j]
            h2 = float(s2.max()) or 1.0
            for j2, v in zip(r2, s2):
                j2 = int(j2)
                if j2 != i:
                    agg[j2] = agg.get(j2, 0.0) + 0.45 * w * float(v) / h2
        agg.pop(i, None)
        return sorted(agg.items(), key=lambda x: -x[1])

    def _fits(self, i: int, j: int, mods) -> bool:
        e, c = self.e, self.e.c
        y0, rt0 = c["year"][i] or 0, c["runtime"][i] or 0
        gs = e.genre_of[j]
        for kind, val in mods:
            if kind == "genre" and val not in gs:
                return False
            if kind == "dark" and not (gs & _DARK):
                return False
            if kind == "older" and not ((c["year"][j] or 9999) < y0 - 2):
                return False
            if kind == "newer" and not ((c["year"][j] or 0) > y0 + 2):
                return False
            if kind == "shorter" and not (c["runtime"][j] and (not rt0 or c["runtime"][j] < rt0 - 10)):
                return False
            if kind == "better" and not (e.bayes[j] > e.bayes[i]):
                return False
            if kind == "lang" and c["lang"][j] != val:
                return False
            if kind == "foreign" and c["lang"][j] in (None, "en"):
                return False
        return True

    # -- the conversation -------------------------------------------------
    def reply(self, message: str, context: int | None = None) -> dict:
        e = self.e
        raw = (message or "").strip()
        f = re.sub(r"\s+", " ", _fold(raw)).strip(" ?!.")
        f = re.sub(r"\b(what|who|how|that|where)s\b", r"\1's", f)     # "whats" -> "what's"
        ctx = context if context is not None and 0 <= context < e.n else None

        if not f:
            return self._out("Name a film you liked, or a genre and a decade, and I'll take it from there.")
        if _GREETING.match(f):
            return self._out("Hi. Ask me about any film — what it's about, who made it, whether it's any good — "
                             "or for films like one you loved. I know 17,719 films, up to 2015.",
                             follow=["Tell me about Heat", "Films like Spirited Away", "90s horror"])
        if _HELP.match(f):
            return self._help()
        if _THANKS.match(f) and len(f.split()) <= 4:
            return self._out("Any time. Ask about another film whenever you like.", focus=ctx)

        # "…but funnier" is peeled off before anything reads the film's name.
        main, mods = self._split_mods(f)

        # A person's films: "films by nolan", "tom hanks movies", "starring cate blanchett".
        for pat in (_PERSON_ASK, _PERSON_TAIL):
            m = pat.match(main)
            if m:
                who = self.find_person(m.group("x"))
                if who:
                    return self._person(who, mods)

        # The questions, most specific first. A pattern whose film doesn't
        # resolve hands over to the next: "some like it hot" contains "like",
        # but it is a title, which the "about" patterns then find.
        unresolved = unresolved_intent = None
        for intent, pat in _ASK:
            m = pat.search(main)
            if not m:
                continue
            span = m.groupdict().get("x") or ""
            target = self._target(span, ctx)
            if target == "pronoun-missing":
                return self._out("Which film do you mean? Name it and I'll answer.",
                                 follow=["Tell me about Alien", "Films like Amelie"])
            if intent == "similar":
                # "better rated ones like it", "funnier films like alien": a mood
                # can come before the film as well as after it.
                mods = mods + [x for x in self._mods_in(main[:m.start("x")]) if x not in mods]
            if target is not None:
                if intent == "is_genre":
                    return self._is_genre(target, m.group("g"))
                out = self._answer(intent, target, mods)
                if intent in ("about", "director", "cast", "year", "runtime", "rating"):
                    self._note_namesakes(out, span, target)
                return out
            if intent == "similar":
                # "amelie in french": the trailing mood is not part of the title.
                span2, more = self._strip_trailing_mods(span)
                if more:
                    i2, _ = self.find_film(span2)
                    if i2 is not None:
                        return self._similar(i2, mods + more)
                blended = self._blend_span(span)
                if blended:
                    return blended
                who = self.find_person(span)
                if who:
                    return self._person(who, mods)
            if unresolved is None:
                unresolved, unresolved_intent = span, intent

        # A short follow-up — "but funnier", "older ones", "more" — is about the
        # film in hand when it compares; a plain mood ("something scary") is not.
        short = len(f.split()) <= 5
        moods = mods or (self._mods_in(f) if short else [])
        if ctx is not None and f in ("more", "more like it", "more like that", "more please", "others",
                                     "another", "any others", "what else", "next"):
            return self._similar(ctx, [], skip=4)
        if ctx is not None and short and (f in ("similar", "similar ones") or (moods and _COMPARATIVE.search(f))):
            return self._similar(ctx, moods)

        # A description rather than a question: "something uplifting under two
        # hours", "like Interstellar but more emotional".
        described = self._describe(raw, f, main, unresolved_intent)
        if described:
            return described

        # "What should I watch?", with or without a mood. Only the mood counts:
        # the rest of the sentence is not a title to search for.
        if _OPEN.search(f):
            if self._structured(f):
                return self._browse(self.e.parse(f))
            moods = mods or self._mods_in(f)
            return self._mood(moods) if moods else self._picks()

        if moods and (not main or short) and not self._film_in(main):
            # "90s horror" is a mood *and* a decade: keep every constraint it names.
            return self._browse(self.e.parse(f)) if self._structured(f) else self._mood(moods)

        # The whole message might just be a title, or a person.
        if not mods and not _PERSON_ASK.match(main):
            if e.names_title(main) or not self._structured(main):
                i, how = self.find_film(main)
                if i is not None and how == "exact":
                    return self._answer("about", i, [])
            who = self.find_person(main)
            if who and " " in who:
                return self._person(who, [])

        if unresolved is not None and not self._structured(unresolved):
            return self._not_found(unresolved)
        return self._category(raw)

    @property
    def discover(self):
        """Natural-language discovery, built on first use (it loads the sentence model)."""
        if self._discover is None:
            from _discover import Discover
            self._discover = Discover(self.e)
        return self._discover

    def _describe(self, raw: str, f: str, main: str, unresolved_intent: str | None) -> dict | None:
        """
        Answer with discovery when the message describes what to watch. Not for
        a question about one film ("tell me about la la land" must still say it
        isn't here), a bare title or person, or a plain category, which keep
        their own answers.
        """
        if unresolved_intent in ("about", "director", "cast", "year", "runtime", "rating", "genre", "is_genre"):
            return None
        if self._film_in(main) or (not unresolved_intent and self.find_person(main)):
            return None
        d = self.discover
        r = d.read(raw)
        direct, related = d._mood_tags(r.mood)
        cue = _DESCRIBE.search(f)
        useful = (r.runtime_min or r.runtime_max
                  # "something that isn't scary": a category answer would serve Horror.
                  or r.avoid_genres or r.avoid
                  or (r.mood and (direct or related or cue))
                  or (r.anchors and (r.mood or r.constrained or unresolved_intent == "similar")))
        if not useful:
            return None
        out = d.search(raw, 8)
        res = out["results"]
        if not res:
            note = out["notes"][-1] if out["notes"] else "Nothing in the catalogue fits all of that."
            return self._out(note + " Try loosening one part of it.")
        read = ", ".join(c["label"] for c in out["understood"]["chips"])
        text = f"Here's what fits: {_join([self.name(x['item_id']) for x in res[:4]], 'and')}."
        if read:
            text += f" I read that as: {read}."
        if out["notes"]:
            text += " " + " ".join(out["notes"])
        first = _display(res[0]["title"])
        return self._out(text, [x["item_id"] for x in res], scores=[x.get("score") for x in res],
                         follow=[f"Tell me about {first}", f"Films like {first}", "Something shorter"])

    def namesakes(self, span: str, chosen: int) -> list[int]:
        """Other films filed under the same title, unless the span gave a year."""
        s = _clean(span)
        if re.search(r"\b(?:19|20)\d{2}\)?$", s):
            return []
        same = set(self.e.key_index.get(_key(s), [])) | set(self.e.key_index.get(_key(_clean(span, tail=False)), []))
        same.discard(chosen)
        return sorted(same, key=lambda i: -(self.e.c["rc"][i] or 0))[:2]

    def _note_namesakes(self, out: dict, span: str, chosen: int):
        k = _key(_clean(span))
        if not k or _PRONOUN.match(k):
            return                      # "it" means the film in hand, not films titled "It"
        others = self.namesakes(span, chosen)
        if others:
            out["response"] += (f" (There's also {_join([self.name(i) for i in others], 'and')}; "
                                f"add the year to ask about {'it' if len(others) == 1 else 'one of those'}.)")

    def _target(self, span: str, ctx: int | None):
        k = _key(_clean(span))
        if not k or _PRONOUN.match(k):
            return ctx if ctx is not None else "pronoun-missing"
        i, _ = self.find_film(span)
        return i

    def _split_mods(self, f: str) -> tuple[str, list[tuple[str, object]]]:
        main, tail = f, ""
        m = re.search(r"\s*,?\s*\b(?:but|only|except|just|and make it|that(?:'s| is| are)|which (?:is|are))\s+(.+)$", f)
        if m:
            main, tail = f[:m.start()], m.group(1)
        elif re.match(r"^(?:but|something|anything|any|more|a|an|some)\s", f):
            main, tail = "", f
        mods = []
        for pat, kind, val in _MODS:
            mm = pat.search(tail)
            if mm:
                mods.append((kind, _LANG_CODE.get(mm.group(1), mm.group(1)) if kind == "lang" else val))
        if not mods:
            return f, []                  # the "but" was part of a title, not a modifier
        return main.strip(), mods

    def _mods_in(self, text: str) -> list[tuple[str, object]]:
        out = []
        for pat, kind, val in _MODS:
            mm = pat.search(text)
            if mm:
                out.append((kind, _LANG_CODE.get(mm.group(1), mm.group(1)) if kind == "lang" else val))
        return out

    def _strip_trailing_mods(self, span: str) -> tuple[str, list[tuple[str, object]]]:
        """'amelie in french' -> ('amelie', [lang fr]); stops at the first non-mood word."""
        s, out = _clean(span), []
        for _ in range(3):
            hit = None
            for pat, kind, val in _MODS:
                for mm in pat.finditer(s):
                    if re.fullmatch(r"\s*(?:films?|movies?|ones?)?\s*", s[mm.end():]):
                        hit = (mm, kind, val)
            if not hit:
                break
            mm, kind, val = hit
            out.append((kind, _LANG_CODE.get(mm.group(1), mm.group(1)) if kind == "lang" else val))
            s = s[:mm.start()].strip(" ,")
        return s, out

    def _film_in(self, text: str) -> bool:
        i, how = self.find_film(text) if text else (None, "")
        return i is not None and how == "exact"

    def _structured(self, text: str) -> bool:
        """True when text names a genre, a language or a decade."""
        p = self.e.parse(text)
        return bool(p.get("genres") or p.get("lang") or p.get("year_min"))

    # -- answers ----------------------------------------------------------
    def _answer(self, intent: str, i: int, mods) -> dict:
        c = self.e.c
        name = self.name(i)
        title = _display(c["title"][i])
        dirs, cast = c["directors"][i] or [], c["cast"][i] or []
        series = c["media"][i] == "tv"
        if intent == "director":
            if not dirs:
                return self._out(f"I don't have the director of {name} on record.", [i], focus=i)
            others = sorted((j for d in dirs for j in self.directed.get(_fold(d), ()) if j != i),
                            key=lambda j: -self.e.bayes[j])[:4]
            # A series has creators rather than one director; TMDB lists those.
            text = (f"{name} is a series created by {_join(dirs)}." if series
                    else f"{name} was directed by {_join(dirs)}.")
            if others:
                text += f" Their other films here include {_join([self.name(j) for j in others])}."
            return self._out(text, [i, *others], focus=i,
                             follow=[f"Who's in {title}?", f"Films like {title}", "Is it any good?"])
        if intent == "cast":
            if not cast:
                return self._out(f"I don't have a cast list for {name}.", [i], focus=i)
            text = f"{name} stars {_join(cast[:6])}."
            if dirs:
                text += f" It was {'created' if series else 'directed'} by {_join(dirs)}."
            return self._out(text, [i], focus=i,
                             follow=[f"Films like {title}", "Is it any good?", f"{cast[0]} films"])
        if intent == "year":
            y = c["year"][i]
            text = f"{title} came out in {y}." if y else f"I don't have a release year for {name}."
            return self._out(text, [i], focus=i, follow=["What's it about?", f"Films like {title}"])
        if intent == "runtime":
            rt = c["runtime"][i]
            if c["media"][i] == "tv":
                text = (f"{name} is a series; its episodes run {_runtime(rt)} ({rt} minutes)." if rt
                        else f"{name} is a series, and I don't have its episode length.")
            else:
                text = (f"{name} runs {_runtime(rt)} ({rt} minutes)." if rt
                        else f"I don't have a running time for {name}.")
            return self._out(text, [i], focus=i, follow=["Is it any good?", f"Films like {title} but shorter"])
        if intent == "rating":
            return self._out(f"{name}: {self.verdict(i)}", [i], focus=i,
                             follow=[f"Films like {title}", "What's it about?", "Better rated ones like it"])
        if intent == "genre":
            gs = c["genres"][i] or []
            tags = (c["tags"][i] or [])[:4]
            text = f"{name} is filed under {_join(gs)}." if gs else f"{name} has no genre on record."
            if tags:
                text += f" Viewers tag it {_join([f'“{t}”' for t in tags])}."
            return self._out(text, [i], focus=i, follow=[f"Films like {title}", "Is it any good?"])
        if intent == "similar":
            return self._similar(i, mods)
        return self._about(i)

    def _is_genre(self, i: int, word: str) -> dict:
        c = self.e.c
        g = _GENRE_WORD.get(word, word)
        gs = c["genres"][i] or []
        has = g in self.e.genre_of[i]
        answer = "Yes" if has else "Not really"
        text = f"{answer}: {self.name(i)} is filed under {_join(gs)}." if gs else \
            f"I have no genre on record for {self.name(i)}."
        tags = (c["tags"][i] or [])[:3]
        if tags:
            text += f" Viewers tag it {_join([f'“{t}”' for t in tags])}."
        title = _display(c["title"][i])
        return self._out(text, [i], focus=i, follow=[f"Films like {title}", "Is it any good?"])

    def _about(self, i: int) -> dict:
        c = self.e.c
        dirs, cast = c["directors"][i] or [], c["cast"][i] or []
        who = []
        if dirs:
            who.append(f"{'created' if c['media'][i] == 'tv' else 'directed'} by {_join(dirs)}")
        if cast:
            who.append(f"with {_join(cast[:3])}")
        head = f"{self.name(i)}, {', '.join(who)}." if who else f"{self.name(i)}."
        rt = _runtime(c["runtime"][i])
        if c["media"][i] == "tv":
            rt = f"a series with {rt} episodes" if rt else "a series"
        facts = [x for x in (_join(c["genres"][i] or []), rt) if x]
        if facts:
            head += f" {', '.join(facts)}."
        parts = [head]
        if c["overview"][i]:
            parts.append(c["overview"][i].strip())
        parts.append(self.verdict(i))
        sims = self.e.similar(i, 8)
        if sims:
            parts.append(f"If you like it, the closest films are {_join([self.name(s['item_id']) for s in sims[:3]])}.")
        if (c["rc"][i] or 0) < 300:
            parts.append("If you meant a newer film with this title: the catalogue ends in 2015.")
        title = _display(c["title"][i])
        return self._out(" ".join(parts), [i] + [s["item_id"] for s in sims],
                         scores=[None] + [s.get("score") for s in sims], focus=i,
                         follow=[f"Films like {title}", "Who directed it?", "Who's in it?"])

    def _similar(self, i: int, mods, skip: int = 0) -> dict:
        c = self.e.c
        name, title = self.name(i), _display(c["title"][i])
        if not mods and skip:
            sims = self.e.similar(i, 24)[skip:skip + 8]
            if not sims:
                return self._out(f"That's all 24 of {name}'s matches.", [i], focus=i)
            text = f"More like {name}: {_join([self.name(s['item_id']) for s in sims[:5]])}."
            return self._out(text, [s["item_id"] for s in sims], scores=[s.get("score") for s in sims], focus=i,
                             follow=[f"Films like {title} but funnier", "Older ones", "Darker ones"])
        if not mods:
            sims = self.e.similar(i, 8)
            if not sims:
                return self._out(f"I have no matches stored for {name}.", [i], focus=i)
            top = sims[0]
            genres = [g.title() for g in top.get("shared_genres") or []][:3]
            tags = [t for t in (top.get("shared_tags") or [])
                    if t.lower() not in {g.lower() for g in top.get("shared_genres") or []}]
            why = [x for x in (_join(genres), f"“{tags[0]}”" if tags else "") if x]
            same_dir = set(map(_fold, c["directors"][i] or [])) & set(map(_fold, top.get("directors") or []))
            reason = f": they share {_join(why)}" if why else ""
            if same_dir:
                reason += ", and the same director" if reason else ": same director"
            text = (f"If you liked {name}, start with {self.name(top['item_id'])}{reason}. "
                    f"Then {_join([self.name(s['item_id']) for s in sims[1:4]])}.")
            return self._out(text, [s["item_id"] for s in sims], scores=[s.get("score") for s in sims], focus=i,
                             follow=[f"Films like {title} but funnier", "Darker ones", "Older ones"])
        pool = [(j, s) for j, s in self.two_hop(i) if self._fits(i, j, mods)][:8]
        desc = self._mods_text(mods)
        if not pool:
            if any(k in ("lang", "genre") for k, _ in mods):
                alt = self._mood(mods)
                alt["response"] = f"Nothing close to {name} is {desc}. " + alt["response"]
                return alt
            return self._out(f"Nothing close to {name} is {desc}. Try loosening it, or ask for plain matches.",
                             [i], focus=i, follow=[f"Films like {title}"])
        hi = pool[0][1] or 1.0
        text = f"Closest to {name}, but {desc}: {_join([self.name(j) for j, _ in pool[:4]])}."
        return self._out(text, [j for j, _ in pool], scores=[s / hi for _, s in pool], focus=i,
                         follow=[f"Films like {title}", f"Tell me about {_display(c['title'][pool[0][0]])}"])

    def _mods_text(self, mods) -> str:
        genre = {"comedy": "funnier", "horror": "scarier", "romance": "more romantic",
                 "animation": "animated", "children": "for kids", "action": "with more action",
                 "sci-fi": "sci-fi", "documentary": "a documentary"}
        other = {"dark": "darker", "older": "older", "newer": "newer", "shorter": "shorter",
                 "better": "better rated", "foreign": "not in English"}
        words = []
        for kind, val in mods:
            if kind == "genre":
                words.append(genre.get(str(val), str(val)))
            elif kind == "lang":
                words.append(f"in {_LANG_NAME.get(str(val), str(val))}")
            else:
                words.append(other[kind])
        return _join(words)

    def _blend_span(self, span: str) -> dict | None:
        parts = [p for p in re.split(r"\s*(?:,|&|\band\b|\bplus\b|\bor\b|\bmixed with\b|\bmeets\b|\bcrossed with\b)\s*",
                                     _clean(span)) if p]
        if len(parts) < 2:
            return None
        ids = []
        for p in parts:
            i, how = self.find_film(p)
            if i is None or how == "fuzzy":
                return None
            ids.append(i)
        res = self.e.blend(ids, 8)
        if not res:
            return None
        text = (f"Films that sit between {_join([self.name(i) for i in ids])}: "
                f"{_join([self.name(r['item_id']) for r in res[:4]])}. "
                f"They're ranked by how close each is to all of them at once.")
        return self._out(text, [r["item_id"] for r in res], scores=[r.get("score") for r in res],
                         follow=[f"Tell me about {_display(self.e.c['title'][res[0]['item_id']])}"])

    def _person(self, who: str, mods) -> dict:
        e = self.e
        d, a = self.directed.get(who, set()), self.acted.get(who, set())
        nm = self.person_name.get(who, who.title())
        ids = sorted(d | a, key=lambda i: -e.bayes[i])
        if mods:
            ids = [i for i in ids if all((k != "genre" or v in e.genre_of[i]) and
                                         (k != "dark" or e.genre_of[i] & _DARK) for k, v in mods)]
        if not ids:
            return self._out(f"None of {nm}'s films here fit that.")
        n = len(d | a)
        role = "directed" if len(d) >= len(a) else "appears in"
        text = (f"{nm} {role} {n} film{'s' if n != 1 else ''} in the catalogue. "
                f"The best rated: {_join([self.name(i) for i in ids[:4]])}.")
        return self._out(text, ids[:8], follow=[f"Tell me about {_display(e.c['title'][ids[0]])}"])

    def _mood(self, mods) -> dict:
        genres = [str(v).title() if v != "sci-fi" else "Sci-Fi" for k, v in mods if k == "genre"]
        lang = next((v for k, v in mods if k == "lang"), None)
        if not genres and not lang:
            return self._picks()
        r = self.e.search("", genres=genres or None, lang=lang, limit=8)
        if not r["results"]:
            return self._picks()
        what = " ".join(x for x in (_LANG_NAME.get(lang, "") if lang else "", " and ".join(genres)) if x)
        text = (f"The best-rated {what} films here: "
                f"{_join([self.name(x['item_id']) for x in r['results'][:4]])}.")
        return self._out(text, [x["item_id"] for x in r["results"]],
                         follow=[f"Tell me about {_display(r['results'][0]['title'])}"])

    def _browse(self, p: dict) -> dict:
        r = self.e.search("", genres=p.get("genres"), lang=p.get("lang"),
                          year_min=p.get("year_min"), year_max=p.get("year_max"), limit=8)
        if not r["results"]:
            return self._picks()
        return self._category_text(r)

    def _picks(self) -> dict:
        ids = self.e.browse["hero"][:8]
        text = (f"A few of the best-loved films here: {_join([self.name(i) for i in ids[:4]])}. "
                f"Tell me one you've enjoyed and I'll find what's closest to it.")
        return self._out(text, ids, follow=["Something funny", "Something scary", "Films like Inception"])

    def _category(self, raw: str) -> dict:
        e = self.e
        r = e.search(raw, limit=8)
        res, filters = r["results"], r["filters"]
        structured = bool(filters.get("genres") or filters.get("lang") or filters.get("year_min"))
        if res and not structured:
            # Free text only. Trust it when it names a title, a person or a theme;
            # a sentence that merely shares a word with some titles is not a search.
            words = [w for w in re.split(r"[^a-z0-9']+", _fold(raw)) if len(w) > 2 and w not in _CHATTER]
            phrase = _key(raw)
            meaningful = (res[0].get("score", 0) >= 150 or phrase in e.tag_index or phrase in e.people
                          or any(len(e.tag_index.get(w, ())) >= 5 for w in words))
            if not meaningful or not words:
                return self._help(prefix="I'm not sure what you're after. ")
        if not res or (not structured and len(_key(raw).split()) <= 3):
            k = _key(raw)
            typo = [i for kk in difflib.get_close_matches(k, self._all_keys, n=3, cutoff=0.65)
                    for i in self.e.key_index[kk] if (self.e.c["rc"][i] or 0) >= 2000]
            if typo and not res:
                i = max(typo, key=lambda j: self.e.c["rc"][j] or 0)
                return self._out(f"Did you mean {self.name(i)}?", [i], focus=i,
                                 follow=[f"Tell me about {_display(self.e.c['title'][i])}",
                                         f"Films like {_display(self.e.c['title'][i])}"])
        if not res:
            near = self.maybe(raw)
            if near:
                return self._out(f"I couldn't place “{raw}”. Did you mean {_join([self.name(i) for i in near], 'or')}?",
                                 near, follow=[f"Tell me about {_display(e.c['title'][near[0]])}"])
            return self._help(prefix=f"I couldn't find anything for “{raw}”. ")
        if structured:
            return self._category_text(r)
        text = f"Here's what matches “{raw}”: {_join([self.name(x['item_id']) for x in res[:4]])}."
        return self._out(text, [x["item_id"] for x in res],
                         follow=[f"Tell me about {_display(res[0]['title'])}", f"Films like {_display(res[0]['title'])}"])

    def _category_text(self, r: dict) -> dict:
        res, filters = r["results"], r["filters"]
        adj = []
        if filters.get("lang"):
            adj.append(_LANG_NAME.get(filters["lang"], filters["lang"]))
        if filters.get("genres"):
            adj.append(" and ".join(filters["genres"]))
        when = ""
        if filters.get("year_min"):
            lo, hi = filters["year_min"], filters.get("year_max", filters["year_min"])
            when = (f" from the {lo}s" if hi - lo == 9 and lo % 10 == 0 else
                    f" from {lo}" if lo == hi else f" from {lo} to {hi}")
        text = (f"The best-rated {' '.join(adj) + ' ' if adj else ''}films{when}: "
                f"{_join([self.name(x['item_id']) for x in res[:4]])}. {r['total']:,} films fit in all.")
        return self._out(text, [x["item_id"] for x in res],
                         follow=[f"Tell me about {_display(res[0]['title'])}", f"Films like {_display(res[0]['title'])}"])

    def _not_found(self, span: str) -> dict:
        s = _clean(span)
        near = self.maybe(s)
        text = f"I couldn't find a film called “{s}”."
        if near:
            text += f" The closest titles I have are {_join([self.name(i) for i in near], 'and')}."
        text += " The catalogue covers 17,719 films from MovieLens up to 2015, so newer releases aren't here."
        return self._out(text, near, follow=[f"Tell me about {_display(self.e.c['title'][near[0]])}"] if near else [])

    def _help(self, prefix: str = "") -> dict:
        return self._out(prefix + "I can tell you about any film here — what it's about, who directed it, who's in it, "
                         "when it came out, how long it runs and how it's rated — find films like one you loved "
                         "(even “like Alien but funnier”, or “between Heat and Drive”), list a director's or "
                         "actor's films, or browse a category like “90s horror”.",
                         follow=["Tell me about Inception", "Films like Alien but funnier", "Christopher Nolan films"])

    def _out(self, text: str, ids: list[int] | None = None, scores: list | None = None,
             focus: int | None = None, follow: list[str] | None = None) -> dict:
        ids = ids or []
        scores = scores or [None] * len(ids)
        seen, cards = set(), []
        for i, s in zip(ids, scores):
            if i in seen:
                continue
            seen.add(i)
            cards.append(self.e.item(int(i), s))
        return {"response": text, "recommendations": cards[:9], "focus": focus,
                "follow_ups": (follow or [])[:3]}
