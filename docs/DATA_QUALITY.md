# Catalogue data quality

Generated 2026-10-26T06:22:27Z by `pipelines/build_catalog.py`. Machine-readable:
`data/artifacts/data_quality.json`.

## Size

| | |
|---|---|
| Films | **88,514** (88,114 films, 400 series) |
| Served before this build (ids unchanged) | 17,719 |
| Added | 70,795 |
| With MovieLens ratings | 84,408 (43,867 with five or more) |
| Metadata only (no ratings) | 4,106 |
| MovieLens 32M ids read | 87,585 |
| Duplicates merged (shared TMDB id, matching year and title) | 30 |
| Duplicates among films served before, kept as aliases | 2 |
| Shared TMDB ids kept apart (conflicts) | 45 |
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
| `alias_of_served_film` | 2 | 0 |
| `has_merged_duplicate` | 30 | 15 |
| `legacy_tmdb_data_was_another_film` | 7 | 0 |
| `missing_genres` | 720 | 693 |
| `missing_imdb_id` | 959 | 0 |
| `missing_language` | 758 | 662 |
| `missing_overview` | 1,594 | 1,470 |
| `missing_poster` | 2,326 | 2,165 |
| `missing_runtime` | 1,307 | 1,154 |
| `missing_year` | 71 | 68 |
| `no_ratings` | 4,106 | 3,151 |
| `no_tmdb_id` | 89 | 89 |
| `runtime_outlier` | 14 | 14 |
| `runtime_zero_cleared` | 6 | 6 |
| `title_from_tmdb` | 1 | 1 |
| `tmdb_imdb_mismatch` | 36 | 29 |
| `tmdb_not_found` | 1,187 | 1,120 |
| `tmdb_shared_id_not_this_film` | 2 | 2 |
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
recommendations skip it, and a like of it counts for the film it duplicates.

| TMDB id | Listed movieId | Alias movieId | Titles | Years |
|---|---|---|---|---|
| 12599 | 3799 | 91752 | Pokémon the Movie 2000 / Pokemon: The Movie 2000 | 2000 / 2000 |
| 13209 | 48682 | 80330 | Offside / Offside | 2006 / 2006 |

## Shared TMDB ids kept apart

| TMDB id | movieIds | Titles | Years | Why not merged |
|---|---|---|---|---|
| 744 | 1101, 96062 | Top Gun / Jekyll | 1986 / 2007 | years differ |
| 3556 | 95214, 27002 | Mad Dog Coll / From the Earth to the Moon | 1961 / 1998 | years differ |
| 4043 | 26599, 107718 | Law of Desire (Ley del deseo, La) / State of Play | 1987 / 2003 | years differ |
| 9045 | 6041, 26849 | Amen. / The Stand | 2002 / 1994 | years differ |
| 9343 | 2970, 108977 | Fitzcarraldo / Paranoia Agent | 1982 / 2004 | years differ |
| 9687 | 42197, 51024 | Keeping Mum / The Odyssey | 2005 / 1997 | years differ |
| 9693 | 48774, 93040 | Children of Men / The Civil War | 2006 / 1990 | years differ |
| 10991 | 4241, 174533 | Pokémon 3: The Movie / Pokémon: Spell of the Unown | 2001 / 2000 | titles differ |
| 12117 | 2676, 51185 | Instinct / Human Trafficking | 1999 / 2005 | years differ |
| 12584 | 4710, 62970 | The Shootist / Tin Man | 1976 / 2007 | years differ |
| 12600 | 5672, 174543 | Pokemon 4 Ever (a.k.a. Pokémon 4: The Movie) / Pokémon 4Ever: Celebi - Voice of the Forest | 2002 / 2001 | titles differ |
| 13812 | 62733, 42987 | Quarantine / Fingersmith | 2008 / 2005 | years differ |
| 13851 | 60979, 87308 | Batman: Gotham Knight / Jane Eyre | 2008 / 2006 | years differ |
| 13894 | 27647, 52784 | Tipping the Velvet / Sharkwater | 2002 / 2006 | years differ |
| 14968 | 72178, 48268 | Welcome to Dongmakgol / Empire Falls | 2005 / 2005 | titles differ |
| 16133 | 37495, 27180 | Survive Style 5+ / Joan of Arc | 2004 / 1999 | years differ |
| 18197 | 4537, 26887 | Running on Empty / The Langoliers | 1988 / 1995 | years differ |
| 19002 | 53827, 170279 | Sybil / Darkman III: Die Darkman Die | 1976 / 1996 | years differ |
| 19614 | 8962, 26693 | Fade to Black / It | 2004 / 1990 | years differ |
| 19654 | 7646, 140743 | Rose Red / The Man I Love | 2002 / 1997 | years differ |
| 19997 | 2939, 59869 | Niagara / The Andromeda Strain | 1953 / 2008 | years differ |
| 21753 | 145972, 222065 | Everything You Want / Love Surreal | 2005 / 2015 | years differ |
| 30991 | 240714, 108979 | The Curse of the Living Corpse / Cowboy Bebop | 1964 / 1998 | years differ |
| 32166 | 104072, 27292 | The Count of Monte Cristo / Chasing Sleep | 1998 / 2000 | years differ |
| 34129 | 99792, 256549 | Painted Lady / Shrill Cries of Summer | 1997 / 2008 | years differ |
| 37306 | 80137, 125501 | If God Is Willing and da Creek Don't Rise / Voodoo Moon | 2010 / 2006 | years differ |
| 37939 | 44597, 79222 | Youth of the Beast (Yaju no seishun) / The Phantom | 1963 / 2009 | years differ |
| 38973 | 93719, 228233 | Dragon Age: Redemption / Kamen Rider Den-O The Movie: I’m Born! | 2011 / 2007 | years differ |
| 39172 | 93928, 137809 | After... / After | 2006 / 2009 | years differ |
| 39468 | 86487, 105811 | Mildred Pierce / Godzilla vs. Megaguirus (Gojira tai Megagirasu: Jî shômetsu sakusen) | 2011 / 2000 | years differ |
| 42553 | 114376, 53771 | The Musketeers of Pig Alley / Alice in Wonderland | 1912 / 1985 | years differ |
| 43016 | 125067, 103277 | Waltz of the Toreadors / Hatfields & McCoys | 1962 / 2012 | years differ |
| 43065 | 95717, 105091 | Treasure Island / Nightmares in Red White & Blue: The Evolution of the American Horror Film | 2012 / 2009 | years differ |
| 44264 | 82459, 96347 | True Grit / Endeavour | 2010 / 2012 | years differ |
| 44434 | 94041, 57236 | Appropriate Adult / LOL | 2011 / 2006 | years differ |
| 46976 | 27724, 8400 | Hitler: The Rise of Evil / Wish You Were Here | 2003 / 1987 | years differ |
| 58451 | 200908, 258991 | The Adventures of Pinocchio / The Adventures of Pinocchio | 1975 / 1972 | years differ |
| 79782 | 895, 181393 | Venice/Venice / Venice | 1992 / 2010 | years differ |
| 84198 | 2679, 103208 | Finding North / A Place at the Table | 1998 / 2012 | years differ |
| 109962 | 5848, 102367 | Rich and Famous / Rich and Famous (Gong woo ching) | 1981 / 1987 | years differ |
| 126193 | 115955, 285825 | Category 7: The End of the World / I... Dreaming | 2005 / 1988 | years differ |
| 128062 | 32853, 128305 | Sorrow and the Pity, The (Le chagrin et la pitié) / They All Lie | 1969 / 2009 | years differ |
| 132641 | 864, 134006 | The Wife / Wife | 1995 / 1953 | years differ |
| 542757 | 203453, 215891 | Remélem legközelebb sikerül meghalnod :) / Hope You Die Next Time :-) | 2018 / 2018 | titles differ |
| 792517 | 242100, 239670 | Hate by Dani Rovira / Dani Rovira: Odio | 2021 / 2021 | titles differ |

## Sources

- **movielens-32m** — GroupLens MovieLens 32M (2023-10). Research use; transformations may be redistributed under the same terms; no commercial use without permission.
- **movielens-20m** — GroupLens MovieLens 20M, for films served before the expansion that ML-32M no longer lists.
- **tmdb** — TMDB API, fetched per film with the project's key; see docs/AUDIT.md for the terms.
- **tmdb-legacy** — TMDB data cached before this build, fetch date unknown.
- **wikidata** — Wikidata (CC0), matched by IMDb id.

TMDB cache: not_found: 1,187, ok: 87,096.
Wikidata: 76,361 IMDb ids matched.
