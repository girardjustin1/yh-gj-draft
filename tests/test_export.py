"""Static export: what goes onto a public web host, and what must never.

GitHub Pages runs no Python, so the exported board is the engine's *output*. These tests
guard the two things that matter: that the exported numbers match the live board, and
that nothing private is written to a directory destined for a public URL.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from fantasy_draft.api.export import EXPORT_COLUMNS, export_board, write_static_site


@pytest.fixture
def seeded(db, tmp_config):
    """A database with just enough for a real board."""
    from datetime import datetime

    now = datetime.now()
    players = pl.DataFrame(
        {
            "player_key": [f"00-000{i:04d}" for i in range(40)],
            "full_name": [f"Player {i}" for i in range(40)],
            "normalized_name": [f"player {i}" for i in range(40)],
            "position": (["RB"] * 12 + ["WR"] * 14 + ["QB"] * 8 + ["TE"] * 6),
            "team": ["KC"] * 40,
            "source": ["test"] * 40,
            "ingested_at": [now] * 40,
        }
    )
    db.replace_table("players", players)
    db.replace_table(
        "projections",
        pl.DataFrame(
            {
                "player_key": players["player_key"],
                "source": ["test"] * 40,
                "player_name": players["full_name"],
                "position": players["position"],
                "team": players["team"],
                "season": [tmp_config.league.season] * 40,
                "fantasy_points": [300.0 - 5.0 * i for i in range(40)],
                "ingested_at": [now] * 40,
            }
        ),
    )
    db.replace_table(
        "rankings",
        pl.DataFrame(
            {
                "player_key": players["player_key"],
                "source": ["test"] * 40,
                "player_name": players["full_name"],
                "position": players["position"],
                "team": players["team"],
                "season": [tmp_config.league.season] * 40,
                "ranking_type": ["redraft-overall"] * 40,
                "ecr": [float(i + 1) for i in range(40)],
                "sd": [6.0] * 40,
                "best": [max(1, i - 4) for i in range(40)],
                "worst": [i + 8 for i in range(40)],
                "ingested_at": [now] * 40,
            }
        ),
    )
    return db, tmp_config


class TestExportContent:
    def test_exports_players_and_league_shape(self, seeded):
        db, cfg = seeded
        payload = export_board(db, cfg)
        assert payload["players"]
        assert payload["league"]["teams"] == 12
        assert payload["league"]["lineup_slots"]
        assert payload["season"] == cfg.league.season

    def test_numbers_match_the_live_board(self, seeded):
        """The export is the engine's output, not a second calculation."""
        from fantasy_draft.analytics.board import build_board

        db, cfg = seeded
        live = build_board(db, cfg).frame
        payload = export_board(db, cfg)
        by_key = {p["player_key"]: p for p in payload["players"]}
        for row in live.head(10).to_dicts():
            exported = by_key[row["player_key"]]
            assert exported["player_score"] == pytest.approx(row["player_score"], abs=0.05)
            assert exported["vbd"] == pytest.approx(row["vbd"], abs=0.05)
            assert exported["projection"] == pytest.approx(row["projected_points"], abs=0.05)

    def test_replacement_levels_are_included(self, seeded):
        db, cfg = seeded
        payload = export_board(db, cfg)
        assert payload["replacement"]
        for level in payload["replacement"].values():
            assert level["rank"] >= 1
            assert "explanation" in level

    def test_draft_dependent_fields_are_omitted(self, seeded):
        """Survival and two-pick EV depend on live state; a baked value would be a lie."""
        db, cfg = seeded
        player = export_board(db, cfg)["players"][0]
        for key in ("probability_gone", "probability_available",
                    "two_pick_expected_value", "lineup_upgrade"):
            assert key not in player

    def test_adp_spread_is_carried(self, seeded):
        """The browser needs adp_sd to reproduce the ADP-only survival curve."""
        db, cfg = seeded
        assert export_board(db, cfg)["players"][0]["adp_sd"] is not None

    def test_capabilities_are_declared_honestly(self, seeded):
        """The page states what it does and does not do; that claim is checked here.

        Since engine.js was ported, survival and the Monte Carlo run in the browser. Live
        Sleeper sync genuinely cannot, because it needs network calls and the player-id
        map, so it must stay in the honest "requires the local engine" list.
        """
        db, cfg = seeded
        caps = export_board(db, cfg)["capabilities"]
        # Only genuine impossibilities belong here. Sleeper serves CORS "*", and both
        # the strategy classifier and draft-room read are pure functions of state the
        # browser already has — calling those "requires the local engine" was false.
        assert caps["requires_local_engine"] == [
            "re-scoring the board after a data refresh"
        ]
        assert "live Sleeper draft sync" in caps["not_yet_in_browser"]
        assert "roster-aware survival probability" in caps["computed_in_browser"]
        assert "two-pick expected value" in caps["computed_in_browser"]
        # Nothing may be claimed in two places at once.
        assert not (set(caps["computed_in_browser"]) & set(caps["requires_local_engine"]))

    def test_empty_board_refuses_rather_than_exporting_nothing(self, db, tmp_config):
        with pytest.raises(RuntimeError, match="ff data refresh"):
            export_board(db, tmp_config)


class TestPrivacy:
    """This directory becomes a public URL. Nothing private may reach it."""

    def test_no_draft_state_is_exported(self, seeded):
        db, cfg = seeded
        payload = export_board(db, cfg)
        assert not {"draft", "picks", "drafted", "roster"} & set(payload)

    def test_no_platform_identifiers(self, seeded):
        db, cfg = seeded
        raw = json.dumps(export_board(db, cfg))
        for forbidden in ("sleeper_id", "league_id", "user_id", "draft_id",
                          "access_token", "refresh_token", "client_secret"):
            assert forbidden not in raw

    def test_player_keys_are_public_nflverse_ids(self, seeded):
        db, cfg = seeded
        keys = [p["player_key"] for p in export_board(db, cfg)["players"]]
        assert all(k.startswith(("00-", "DST-", "nm-", "slp-", "fp-", "mfl-")) for k in keys)


class TestSiteOutput:
    def test_writes_a_servable_directory(self, seeded, tmp_path):
        db, cfg = seeded
        out = tmp_path / "site"
        summary = write_static_site(db, cfg, out)
        assert (out / "index.html").is_file()
        assert (out / "board.json").is_file()
        assert (out / ".nojekyll").is_file()      # stop Jekyll mangling the build
        assert summary["players"] > 0

    def test_board_json_is_valid_json(self, seeded, tmp_path):
        db, cfg = seeded
        out = tmp_path / "site"
        write_static_site(db, cfg, out)
        json.loads((out / "board.json").read_text())

    def test_page_falls_back_when_there_is_no_api(self, seeded, tmp_path):
        """The page must detect a static host and switch modes, not fail on every click."""
        db, cfg = seeded
        out = tmp_path / "site"
        write_static_site(db, cfg, out)
        page = (out / "index.html").read_text()
        assert "board.json" in page
        assert 'MODE="offline"' in page
        assert "requires_local_engine" in page or "renderOfflineBanner" in page

    def test_page_makes_no_external_requests(self, seeded, tmp_path):
        """Local-first survives publication: the page must not phone anywhere."""
        import re

        db, cfg = seeded
        out = tmp_path / "site"
        write_static_site(db, cfg, out)
        page = (out / "index.html").read_text()
        external = re.findall(r'(?:src|href)="https?://[^"]+', page)
        assert external == []

    def test_export_columns_exist_on_a_real_board(self, seeded):
        from fantasy_draft.analytics.board import build_board

        db, cfg = seeded
        columns = set(build_board(db, cfg).frame.columns)
        core = {"player_key", "player_name", "position", "vbd", "player_score",
                "adp", "adp_sd", "floor_points", "ceiling_points", "tier"}
        assert core <= set(EXPORT_COLUMNS)
        assert core <= columns
