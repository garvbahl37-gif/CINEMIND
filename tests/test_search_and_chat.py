"""
Behaviour the search and the film assistant must keep.

Runs against the committed artifacts, with no server:

    pip install -r requirements.txt pytest
    pytest tests/

Each case here was a real failure before it was fixed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from _chat import Assistant  # noqa: E402
from _engine import get_engine  # noqa: E402


@pytest.fixture(scope="module")
def e():
    return get_engine()


@pytest.fixture(scope="module")
def bot(e):
    return Assistant(e)


def top(e, q, n=3):
    return [x["title"] for x in e.search(q, limit=n)["results"]]


# -- search: words inside titles must not turn into filters ------------------

@pytest.mark.parametrize("q, expect", [
    ("star wars", "Star Wars"),                      # "wars" was the War genre
    ("children of men", "Children of Men"),          # "children" was the Children genre
    ("the french connection", "The French Connection"),  # "french" was a language filter: 0 results
    ("war of the worlds", "War of the Worlds"),      # 0 results
    ("the italian job", "The Italian Job"),
    ("spy kids", "Spy Kids"),
    ("the good the bad and the ugly", "Good, the Bad and the Ugly"),  # filed article-last
    ("se7en", "Seven"),                              # an alternate title in brackets
])
def test_title_queries_find_the_title(e, q, expect):
    r = e.search(q, limit=3)
    assert r["filters"] == {}, f"{q!r} became filters {r['filters']}"
    assert any(expect.lower() in t.lower() for t in top(e, q)), top(e, q)


@pytest.mark.parametrize("q, filters", [
    ("horror", {"genres": ["Horror"]}),
    ("korean thrillers", {"lang": "ko", "genres": ["Thriller"]}),
    ("90s horror", {"year_min": 1990, "year_max": 1999, "genres": ["Horror"]}),
    ("romantic comedies", {"genres": ["Comedy", "Romance"]}),
    ("war movies", {"genres": ["War"]}),
])
def test_category_queries_still_filter(e, q, filters):
    assert e.search(q, limit=1)["filters"] == filters


# -- recommendations ---------------------------------------------------------

def test_neighbour_artifacts_are_sound(e):
    import numpy as np
    n = e.n
    assert (e.nbr != np.arange(n)[:, None]).all(), "a film recommends itself"
    assert ((e.nbr >= 0) & (e.nbr < n)).all()
    assert (np.diff(e.nsc, axis=1) <= 1e-6).all(), "a score row is not sorted best first"


@pytest.mark.parametrize("film, expect", [
    ("Toy Story", "Toy Story 2"),
    ("Spirited Away", "Howl's Moving Castle"),
    ("Alien", "Aliens"),
])
def test_top_match_is_sensible(bot, e, film, expect):
    i, _ = bot.find_film(film)
    assert expect in [x["title"].split(" (")[0] for x in e.similar(i, 3)]


# -- the assistant -----------------------------------------------------------

def ask(bot, msg, ctx=None):
    return bot.reply(msg, ctx)


def test_about_a_film_answers_with_facts(bot):
    r = ask(bot, "tell me about inception")
    assert r["response"].startswith("Inception (2010)")
    assert "Christopher Nolan" in r["response"]
    assert r["focus"] is not None


def test_follow_ups_use_the_film_in_hand(bot):
    ctx = ask(bot, "tell me about heat")["focus"]
    assert "Michael Mann" in ask(bot, "who directed it?", ctx)["response"]
    assert "Al Pacino" in ask(bot, "whos in it", ctx)["response"]
    assert "but older" in ask(bot, "older ones", ctx)["response"]
    assert "Which film" in ask(bot, "who directed it?")["response"]   # no context


@pytest.mark.parametrize("msg, needle", [
    ("tell me about some like it hot", "Some Like It Hot (1959)"),   # contains "like"
    ("whats fight club about", "Fight Club (1999)"),                  # no apostrophe
    ("how long is lord of the rings", "Fellowship of the Ring"),      # not the 1978 film
    ("films like amelie in french", "but in French"),
    ("films between heat and drive", "between Heat (1995) and Drive (2011)"),
    ("is alien a horror movie", "Yes"),
    ("incepshun", "Did you mean Inception (2010)?"),
    ("i loved amelie, what next?", "If you liked Amelie (2001)"),
])
def test_assistant_understands(bot, msg, needle):
    assert needle in ask(bot, msg)["response"]


def test_recommend_something_funny_is_a_mood_not_a_title_search(bot):
    r = ask(bot, "recommend me something funny")
    assert "Comedy" in r["response"]
    assert "Something Wild" not in r["response"]


def test_films_after_2015_are_not_invented(bot):
    r = ask(bot, "tell me about la la land")
    assert "couldn't find" in r["response"] and "2015" in r["response"]


def test_off_topic_gets_help_not_a_film_list(bot):
    assert "not sure what you're after" in ask(bot, "tell me a joke")["response"]
