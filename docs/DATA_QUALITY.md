# Catalogue data quality

Generated 2026-10-26T16:11:18Z by `pipelines/build_catalog.py`. Machine-readable:
`data/artifacts/data_quality.json`.

## Size

| | |
|---|---|
| Films | **88,514** (88,114 films, 400 series) |
| Served before this build (ids unchanged) | 17,719 |
| Added | 70,795 |
| With MovieLens ratings | 84,404 (43,865 with five or more) |
| Metadata only (no ratings) | 4,110 |
| MovieLens 32M ids read | 87,585 |
| Duplicates merged (shared TMDB id, matching year and title) | 30 |
| Duplicates among films served before, kept as aliases | 6 |
| Shared TMDB ids kept apart (conflicts) | 7 |
| Excluded as invalid | 0 |

## Field coverage

| Field | All | Served before | Added |
|---|---|---|---|
| year | 99.9% | 100.0% | 99.9% |
| genres | 99.2% | 99.9% | 99.0% |
| tags | 57.9% | 89.9% | 49.9% |
| runtime | 98.5% | 99.1% | 98.4% |
| lang | 99.1% | 99.5% | 99.1% |
| overview | 98.2% | 99.3% | 97.9% |
| poster | 97.4% | 99.1% | 96.9% |
| directors | 98.7% | 99.1% | 98.5% |
| cast | 95.6% | 98.6% | 94.9% |
| countries | 99.3% | 99.4% | 99.3% |
| imdbId | 98.9% | 94.6% | 100.0% |
| tmdbId | 99.8% | 99.7% | 99.9% |

## Flags

| Flag | All films | Added films |
|---|---|---|
| `alias_of_served_film` | 6 | 4 |
| `has_merged_duplicate` | 36 | 17 |
| `legacy_tmdb_data_was_another_film` | 7 | 0 |
| `missing_genres` | 720 | 693 |
| `missing_imdb_id` | 959 | 0 |
| `missing_language` | 757 | 661 |
| `missing_overview` | 1,592 | 1,468 |
| `missing_poster` | 2,324 | 2,163 |
| `missing_runtime` | 1,306 | 1,153 |
| `missing_year` | 71 | 68 |
| `no_ratings` | 4,110 | 3,155 |
| `no_tmdb_id` | 89 | 89 |
| `runtime_outlier` | 14 | 14 |
| `runtime_zero_cleared` | 6 | 6 |
| `title_from_tmdb` | 1 | 1 |
| `tmdb_imdb_mismatch` | 36 | 29 |
| `tmdb_not_found` | 1,187 | 1,120 |
| `tmdb_year_conflict` | 7 | 1 |
| `wikidata_ambiguous` | 42 | 32 |
| `wikidata_year_conflict` | 368 | 349 |

`no_ratings` films are metadata-only: recommended by content, never by
co-rating. `tmdb_imdb_mismatch` and the `*_year_conflict` flags mark a record
from that source that disagreed with MovieLens and was **not used** (a TMDB
record with MovieLens' own IMDb id is used whatever its year).
`legacy_tmdb_data_was_another_film` marks a film served before whose own TMDB
id turned out to be another film's: the artwork and synopsis it was shown with
were that film's, and are cleared.

## Duplicates merged

| TMDB id | Kept movieId | Merged movieId | Titles | Years |
|---|---|---|---|---|
| 3687 | 59935 | 212437 | Flesh Gordon / Flesh Gordon | 1974 / 1974 |
| 4912 | 6003 | 144606 | Confessions of a Dangerous Mind / Confessions of a Dangerous Mind | 2002 / 2002 |
| 9775 | 34330 | 178755 | November / November | 2004 / 2003 |
| 13020 | 33154 | 202599 | Enron: The Smartest Guys in the Room / Enron: The Smartest Guys in the Room | 2005 / 2005 |
| 15028 | 5264 | 144426 | Clockstoppers / Clockstoppers | 2002 / 2002 |
| 19666 | 6776 | 181751 | Lagaan: Once Upon a Time in India / Lagaan: Once Upon a Time in India | 2001 / 2001 |
| 24487 | 79329 | 232217 | Passing Strange / Passing Strange | 2009 / 2009 |
| 30508 | 66511 | 191775 | Berlin Calling / Berlin Calling | 2008 / 2008 |
| 34696 | 65633 | 137813 | Fuel / Fuel | 2008 / 2009 |
| 44251 | 95771 | 206323 | Dragon Ball Z: Broly Second Coming (Doragon bôru Z 10: Kiken na futari! Sûpâ senshi wa nemurenai) / Dragon Ball Z: Broly – Second Coming | 1994 / 1994 |
| 45555 | 144858 | 219145 | God's Angry Man / God's Angry Man | 1980 / 1980 |
| 46667 | 110515 | 213187 | Superdad / Superdad | 1973 / 1973 |
| 97995 | 70155 | 163246 | Seven Years Bad Luck / Seven Years Bad Luck | 1921 / 1921 |
| 116440 | 115877 | 252320 | The Simpsons: The Longest Daycare / Maggie Simpson in The Longest Daycare | 2012 / 2012 |
| 133919 | 7396 | 272905 | Scenes From a Marriage (Scener ur ett äktenskap) / Scenes from a Marriage | 1973 / 1974 |
| 159898 | 167510 | 275753 | The Boxing Cats (Prof. Welton's) / The Boxing Cats | 1894 / 1894 |
| 264068 | 176543 | 266856 | That's Sexploitation! / That's Sexploitation! | 2013 / 2013 |
| 265189 | 114342 | 175813 | Force Majeure (Turist) / Force Majeure | 2014 / 2014 |
| 282848 | 117192 | 241346 | Doctor Who: The Time of the Doctor / Doctor Who: The Time of the Doctor | 2013 / 2013 |
| 298721 | 142861 | 162632 | Cemetery of Splendour / Cemetery of Splendor | 2015 / 2015 |
| 315360 | 120757 | 283525 | Beck - Rum 302 / Beck 27 - Room 302 | 2015 / 2015 |
| 372631 | 148976 | 215011 | Marvel Super Hero Adventures: Frost Fight! / Marvel Super Hero Adventures: Frost Fight! | 2015 / 2015 |
| 411012 | 174527 | 215119 | Chasing Trane: The John Coltrane Documentary / Chasing Trane | 2017 / 2017 |
| 411081 | 168000 | 218855 | Summer Love / Summer Love | 2016 / 2016 |
| 412862 | 208020 | 195339 | Ratsasan / Ratsasan | 2018 / 2018 |
| 413765 | 163585 | 187127 | David Lynch: The Art Life / David Lynch: The Art Life | 2016 / 2017 |
| 521777 | 203218 | 258279 | Good Boys / Good Boys | 2019 / 2019 |
| 566222 | 204294 | 229671 | The Great Hack / The Great Hack | 2019 / 2019 |
| 604196 | 203449 | 206117 | The Lonely Island Presents: The Unauthorized Bash Brothers Experience / The Lonely Island Presents: The Unauthorized Bash Brothers Experience | 2019 / 2019 |
| 642732 | 254100 | 258689 | Roadrunner: A Film About Anthony Bourdain / Roadrunner: A Film About Anthony Bourdain | 2021 / 2021 |

## Duplicates among films served before

Both ids may be stored in browsers, so neither row is removed: the alias keeps
its item_id and resolves, but is never listed — search, matches, home rows and
recommendations skip it — and its ratings, and a like of it, count for the film
it duplicates. Pairs the automatic rule cannot see, because their titles differ,
are listed in `pipelines/confirmed_duplicates.json` once a person has confirmed
them.

| TMDB id | Listed movieId | Alias movieId | Titles | Years | Found by |
|---|---|---|---|---|---|
| 12599 | 3799 | 91752 | Pokémon the Movie 2000 / Pokemon: The Movie 2000 | 2000 / 2000 | shared TMDB id, year and title |
| 13209 | 48682 | 80330 | Offside / Offside | 2006 / 2006 | shared TMDB id, year and title |
| 10991 | 4241 | 174533 | Pokémon 3: The Movie / Pokémon: Spell of the Unown | 2001 / 2000 | confirmed by review |
| 12600 | 5672 | 174543 | Pokemon 4 Ever (a.k.a. Pokémon 4: The Movie) / Pokémon 4Ever: Celebi - Voice of the Forest | 2002 / 2001 | confirmed by review |
| 542757 | 215891 | 203453 | Hope You Die Next Time :-) / Remélem legközelebb sikerül meghalnod :) | 2018 / 2018 | confirmed by review |
| 792517 | 242100 | 239670 | Hate by Dani Rovira / Dani Rovira: Odio | 2021 / 2021 | confirmed by review |

- **Pokémon 3: The Movie** / Pokémon: Spell of the Unown: TMDB 10991, 'Pokémon 3: The Movie' (2000-07-08), carries tt0235679, the second id's IMDb id; Wikidata Q890665, the item for tt0235679, is 'Pokémon 3: The Movie'; both are directed by Kunihiko Yuyama. The 2000 Japanese release and its 2001 US release.
- **Pokemon 4 Ever (a.k.a. Pokémon 4: The Movie)** / Pokémon 4Ever: Celebi - Voice of the Forest: TMDB 12600, 'Pokémon 4Ever' (2001-07-07), carries tt0287635, the second id's IMDb id; Wikidata Q82579, the item for tt0287635, is 'Pokémon 4Ever'; same director (Kunihiko Yuyama) and runtime (80 min). The 2001 Japanese release and its 2002 US release.
- **Hope You Die Next Time :-)** / Remélem legközelebb sikerül meghalnod :): The Hungarian original title and its English one, under two IMDb ids (tt8529512, tt8723836); one TMDB record (542757), the same director (Mihály Schwechtje), runtime (96 min) and year.
- **Hate by Dani Rovira** / Dani Rovira: Odio: 'Odio' is Spanish for 'Hate': one stand-up special by Dani Rovira under its English and Spanish titles and two IMDb ids (tt13323128, tt13323166), with one TMDB record (792517) and the same year.

## Shared TMDB ids kept apart

| TMDB id | movieIds | Titles | Years | Why not merged |
|---|---|---|---|---|
| 21753 | 145972, 222065 | Everything You Want / Love Surreal | 2005 / 2015 | years differ |
| 39172 | 93928, 137809 | After... / After | 2006 / 2009 | years differ |
| 58451 | 200908, 258991 | The Adventures of Pinocchio / The Adventures of Pinocchio | 1975 / 1972 | years differ |
| 79782 | 895, 181393 | Venice/Venice / Venice | 1992 / 2010 | years differ |
| 84198 | 2679, 103208 | Finding North / A Place at the Table | 1998 / 2012 | years differ |
| 109962 | 5848, 102367 | Rich and Famous / Rich and Famous (Gong woo ching) | 1981 / 1987 | years differ |
| 132641 | 864, 134006 | The Wife / Wife | 1995 / 1953 | years differ |

## Sources

- **movielens-32m** — GroupLens MovieLens 32M (2023-10). Research use; transformations may be redistributed under the same terms; no commercial use without permission.
- **movielens-20m** — GroupLens MovieLens 20M, for films served before the expansion that ML-32M no longer lists.
- **tmdb** — TMDB API, fetched per film with the project's key; see docs/AUDIT.md for the terms.
- **tmdb-legacy** — TMDB data cached before this build, fetch date unknown.
- **wikidata** — Wikidata (CC0), matched by IMDb id.

TMDB cache: not_found: 1,187, ok: 87,096.
Wikidata: 76,361 IMDb ids matched.
