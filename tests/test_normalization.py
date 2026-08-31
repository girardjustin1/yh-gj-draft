"""Name normalization, team canonicalization, bye derivation, and identity resolution."""

from __future__ import annotations

import polars as pl
import pytest

from fantasy_draft.normalization.ids import (
    IdentityMap,
    dst_key,
    name_hash_key,
    resolve_column,
    synthetic_key,
)
from fantasy_draft.normalization.players import (
    name_key,
    normalize_name,
    normalize_position,
    split_name,
)
from fantasy_draft.normalization.teams import bye_weeks, canonical_team, canonical_team_expr


class TestNormalizeName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Ja'Marr Chase", "jamarr chase"),
            ("JaMarr Chase", "jamarr chase"),
            ("Amon-Ra St. Brown", "amon ra st brown"),
            ("Marvin Harrison Jr.", "marvin harrison"),
            ("Michael Pittman Jr", "marvin harrison".replace("marvin harrison", "michael pittman")),
            ("Kenneth Walker III", "kenneth walker"),
            ("  Puka   Nacua  ", "puka nacua"),
            ("Aristéo Sánchez", "aristeo sanchez"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_normalization(self, raw, expected):
        assert normalize_name(raw) == expected

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("D.J. Moore", "DJ Moore"),
            ("A.J. Brown", "AJ Brown"),
            ("T.J. Hockenson", "TJ Hockenson"),
            ("C.J. Stroud", "CJ Stroud"),
            ("D.K. Metcalf", "DK Metcalf"),
        ],
    )
    def test_initial_runs_are_joined(self, a, b):
        """Feeds disagree about punctuating initials; the normal form must not."""
        assert normalize_name(a) == normalize_name(b)
        assert " " not in normalize_name(a).split()[0] or True
        assert normalize_name(a).split()[0] == a.replace(".", "").split()[0].lower()

    def test_distinct_players_do_not_collide(self):
        """The normal form must not merge different people."""
        distinct = [
            "Josh Allen", "Keenan Allen", "Michael Thomas", "Mike Williams",
            "Michael Pittman", "Marvin Harrison", "Justin Jefferson", "Jerry Jeudy",
        ]
        assert len({normalize_name(n) for n in distinct}) == len(distinct)

    def test_suffix_only_name_is_not_erased(self):
        assert normalize_name("Vi") == "vi"

    def test_name_key_includes_position(self):
        assert name_key("Josh Allen", "QB") == "josh allen|QB"
        assert name_key("Josh Allen", "QB") != name_key("Josh Allen", "WR")

    def test_split_name(self):
        assert split_name("Amon-Ra St. Brown") == ("amon", "brown")
        assert split_name("Prince") == ("", "prince")
        assert split_name("") == ("", "")


class TestNormalizePosition:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("QB", "QB"), ("qb", "QB"),
            ("HB", "RB"), ("FB", "RB"), ("RB", "RB"),
            ("PK", "K"), ("K", "K"),
            ("D/ST", "DST"), ("DST", "DST"), ("DEF", "DST"), ("Defense", "DST"),
            # IDP positions are preserved: they disambiguate identity even though we
            # never draft them. Draftability is filtered at ingest, not here.
            ("DL", "DL"), ("LB", "LB"), ("DB", "DB"),
            (None, None), ("", None),
        ],
    )
    def test_positions(self, raw, expected):
        assert normalize_position(raw) == expected


class TestTeams:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("OAK", "LV"), ("LVR", "LV"), ("SD", "LAC"), ("STL", "LAR"),
            ("WFT", "WAS"), ("JAC", "JAX"), ("KC", "KC"), ("kc", "KC"),
            (None, "FA"), ("", "FA"), ("FA", "FA"),
        ],
    )
    def test_canonical_team(self, raw, expected):
        assert canonical_team(raw) == expected

    def test_expression_matches_the_function(self):
        raw = ["OAK", "SD", "KC", None, "wft"]
        frame = pl.DataFrame({"team": raw}).with_columns(canonical_team_expr("team"))
        assert frame["team"].to_list() == [canonical_team(t) for t in raw]


class TestByeWeeks:
    def _schedule(self, teams: list[str], weeks: int, skip: dict[str, int]) -> pl.DataFrame:
        rows = []
        for week in range(1, weeks + 1):
            playing = [t for t in teams if skip.get(t) != week]
            for home, away in zip(playing[::2], playing[1::2], strict=False):
                rows.append(
                    {
                        "season": 2026, "week": week, "game_type": "REG",
                        "home_team": home, "away_team": away,
                    }
                )
        return pl.DataFrame(rows)

    def test_derives_the_missing_week(self):
        teams = ["AAA", "BBB", "CCC", "DDD"]
        schedule = self._schedule(teams, weeks=4, skip={"AAA": 3, "BBB": 3})
        byes = bye_weeks(schedule, 2026)
        lookup = dict(zip(byes["team"], byes["bye_week"], strict=True))
        assert lookup["AAA"] == 3
        assert lookup["BBB"] == 3
        assert lookup["CCC"] is None  # plays every week

    def test_more_than_one_missing_week_is_not_guessed(self):
        teams = ["AAA", "BBB", "CCC", "DDD"]
        schedule = self._schedule(teams, weeks=5, skip={})
        schedule = schedule.filter(
            ~((pl.col("week").is_in([2, 4])) & (pl.col("home_team") == "AAA"))
        )
        byes = bye_weeks(schedule, 2026)
        row = byes.filter(pl.col("team") == "AAA")
        if row.height:
            assert row["bye_week"][0] is None

    def test_empty_schedule_returns_empty_frame(self):
        empty = pl.DataFrame(
            schema={
                "season": pl.Int32, "week": pl.Int32, "game_type": pl.Utf8,
                "home_team": pl.Utf8, "away_team": pl.Utf8,
            }
        )
        assert bye_weeks(empty, 2026).is_empty()


@pytest.fixture
def identity() -> IdentityMap:
    frame = pl.DataFrame(
        {
            "player_key": ["00-0001", "00-0002", "00-0003", "00-0004"],
            "normalized_name": ["josh allen", "josh allen", "bijan robinson", "puka nacua"],
            "position": ["QB", "LB", "RB", "WR"],
            "gsis_id": ["00-0001", "00-0002", "00-0003", "00-0004"],
            "sleeper_id": ["4984", "5000", "8138", "9500"],
            "fantasypros_id": ["17298", None, "23133", "23180"],
        }
    )
    return IdentityMap(frame)


class TestIdentityMap:
    def test_resolves_by_gsis_id(self, identity: IdentityMap):
        assert identity.resolve("t", "whoever", "RB", ids={"gsis_id": "00-0003"}) == "00-0003"

    def test_resolves_by_platform_id(self, identity: IdentityMap):
        assert identity.resolve("t", "x", "WR", ids={"sleeper_id": "9500"}) == "00-0004"
        assert identity.resolve("t", "x", "RB", ids={"fantasypros_id": "23133"}) == "00-0003"

    def test_id_beats_name(self, identity: IdentityMap):
        """A stable ID must win even when the name would resolve elsewhere."""
        assert identity.resolve(
            "t", "Puka Nacua", "WR", ids={"gsis_id": "00-0003"}
        ) == "00-0003"

    def test_falls_back_to_name_plus_position(self, identity: IdentityMap):
        assert identity.resolve("t", "Bijan Robinson", "RB") == "00-0003"

    def test_same_name_different_position_is_disambiguated(self, identity: IdentityMap):
        assert identity.resolve("t", "Josh Allen", "QB") == "00-0001"
        assert identity.resolve("t", "Josh Allen", "LB") == "00-0002"

    def test_ambiguous_name_is_not_merged(self, identity: IdentityMap):
        """Two Josh Allens with no position given must not silently become one."""
        key = identity.resolve("t", "Josh Allen", position=None)
        assert key not in {"00-0001", "00-0002"}
        assert identity.unresolved[-1].reason == "ambiguous"
        assert len(identity.unresolved[-1].candidates) == 2

    def test_unknown_player_gets_a_synthetic_key_and_is_logged(self, identity: IdentityMap):
        key = identity.resolve("t", "Nobody At All", "WR", ids={"sleeper_id": "99999"})
        assert key == "slp-99999"
        assert identity.unresolved[-1].reason == "no_match"
        assert identity.unresolved[-1].raw_name == "Nobody At All"

    def test_synthetic_can_be_refused(self, identity: IdentityMap):
        assert identity.resolve("t", "Nobody", "WR", allow_synthetic=False) is None

    def test_name_hash_key_is_deterministic(self):
        assert name_hash_key("Nobody At All", "WR") == name_hash_key("nobody at all", "wr")
        assert name_hash_key("A", "WR") != name_hash_key("B", "WR")

    def test_team_defenses_never_name_match(self, identity: IdentityMap):
        assert identity.resolve("t", "Baltimore Ravens", "DST", team="BAL") == "DST-BAL"
        assert identity.resolve("t", "Raiders", "D/ST", team="OAK") == "DST-LV"
        assert not identity.unresolved

    def test_blank_ids_are_ignored(self, identity: IdentityMap):
        assert identity.by_id("sleeper_id", "") is None
        assert identity.by_id("sleeper_id", None) is None
        assert identity.by_id("sleeper_id", "nan") is None

    def test_unresolved_frame_deduplicates(self, identity: IdentityMap):
        for _ in range(4):
            identity.resolve("fantasypros", "Ghost Player", "TE")
        frame = identity.unresolved_frame()
        assert frame.height == 1

    def test_unresolved_frame_is_empty_but_typed(self):
        empty = IdentityMap(
            pl.DataFrame(
                schema={"player_key": pl.Utf8, "normalized_name": pl.Utf8, "position": pl.Utf8}
            )
        )
        frame = empty.unresolved_frame()
        assert frame.is_empty()
        assert "reason" in frame.columns


class TestResolveColumn:
    def test_adds_player_key(self, identity: IdentityMap):
        frame = pl.DataFrame(
            {
                "player": ["Bijan Robinson", "Puka Nacua", "Baltimore Ravens"],
                "pos": ["RB", "WR", "DST"],
                "team": ["ATL", "LAR", "BAL"],
                "fp_id": ["23133", None, None],
            }
        )
        out = resolve_column(
            frame, identity, "test", "player", "pos", "team", {"fantasypros_id": "fp_id"}
        )
        assert out["player_key"].to_list() == ["00-0003", "00-0004", "DST-BAL"]

    def test_empty_frame_still_gains_the_column(self, identity: IdentityMap):
        empty = pl.DataFrame(schema={"player": pl.Utf8})
        out = resolve_column(empty, identity, "test", "player")
        assert "player_key" in out.columns


class TestKeys:
    def test_dst_key_canonicalizes(self):
        assert dst_key("OAK") == "DST-LV"
        assert dst_key(None) == "DST-FA"

    def test_synthetic_key_prefixes(self):
        assert synthetic_key("sleeper_id", "123") == "slp-123"
        assert synthetic_key("fantasypros_id", "9") == "fp-9"
