"""The JavaScript port must agree with the Python engine.

GitHub Pages runs no Python, so `api/static/engine.js` reimplements the parts of the
decision engine that depend on live draft state: opponent roster needs, roster-aware
survival, the Monte Carlo, and two-pick expected value.

Two implementations can drift, and a drifted copy is worse than no copy — it would give
confident, different advice on the phone than the terminal gives. So the port is not
trusted, it is checked: these tests run the JavaScript under node against the Python
originals on identical fixed inputs.

Deterministic functions must match to 1e-9. The Monte Carlo cannot share numpy's RNG, so
it is held to sampling error instead.

Skipped, loudly, when node is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import polars as pl
import pytest

from fantasy_draft.config import LeagueConfig
from fantasy_draft.draft.availability import adp_survival, survival_probabilities
from fantasy_draft.draft.fixtures import build_fixture_draft
from fantasy_draft.draft.opponent_needs import (
    market_prior,
    opponent_needs,
    position_probabilities,
    unfilled_starters,
)
from fantasy_draft.draft.snake import picks_for_slot, slot_for_pick
from fantasy_draft.models import RosterSnapshot

ENGINE = Path(__file__).parent.parent / "src/fantasy_draft/api/static/engine.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is required to check the JavaScript port against Python",
)

#: Mirrors config/league.example.yaml, in the shape engine.js expects.
JS_LEAGUE = {
    "teams": 12, "rounds": 15, "draft_type": "snake", "third_round_reversal": False,
    "dedicated": {"QB": 1, "RB": 2, "WR": 2, "TE": 1},
    "flex_counts": {"FLEX": 1},
    "lineup_slots": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
}


def run_js(body: str, **payload) -> object:
    """Execute a snippet against engine.js and return its JSON result.

    The file is read and evaluated rather than ``require``d, which is both more robust
    and more faithful. More robust because ``/Users/<you>/package.json`` may declare
    ``"type": "module"``, which makes node treat every .js beneath it as ESM — ``module``
    is then undefined, the CommonJS export line is skipped, and ``require`` hands back an
    empty namespace. More faithful because a browser loads this as a classic script into
    the global scope, which is exactly what evaluating the source does.
    """
    script = textwrap.dedent(f"""
        const fs = require("fs");
        const src = fs.readFileSync({str(ENGINE)!r}, "utf8");
        const m = {{exports: {{}}}};
        new Function("module", "exports", src)(m, m.exports);
        const E = m.exports;
        const IN = {json.dumps(payload)};
        {body}
    """)
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise AssertionError(f"node failed:\n{result.stderr[:1500]}")
    return json.loads(result.stdout)


@pytest.fixture
def league() -> LeagueConfig:
    return LeagueConfig.model_validate(
        {"teams": 12, "draft": {"type": "snake", "rounds": 15, "slot": 7},
         "roster": {"qb": 1, "rb": 2, "wr": 2, "te": 1, "flex": 1,
                    "k": 1, "dst": 1, "bench": 6}}
    )


def _board_rows(n: int = 90) -> list[dict]:
    positions = ["RB", "WR", "QB", "TE"] * (n // 4)
    return [
        {
            "player_key": f"p{i}", "position": positions[i], "adp": float(i + 1),
            "adp_sd": 8.0 + (i % 5), "player_score": 95.0 - 0.45 * i,
        }
        for i in range(len(positions))
    ]


class TestSnakeParity:
    @pytest.mark.parametrize("slot", [1, 6, 7, 12])
    def test_picks_for_slot(self, slot):
        js = run_js(
            "console.log(JSON.stringify(E.picksForSlot(IN.slot, IN.L)))",
            slot=slot, L=JS_LEAGUE,
        )
        assert js == picks_for_slot(slot, 12, 15)

    def test_slot_for_every_pick(self):
        js = run_js(
            "const o=[];for(let p=1;p<=180;p++)o.push(E.slotFor(p,IN.L));"
            "console.log(JSON.stringify(o))",
            L=JS_LEAGUE,
        )
        assert js == [slot_for_pick(p, 12) for p in range(1, 181)]

    def test_third_round_reversal(self):
        reversed_league = {**JS_LEAGUE, "third_round_reversal": True}
        js = run_js(
            "const o=[];for(let p=1;p<=72;p++)o.push(E.slotFor(p,IN.L));"
            "console.log(JSON.stringify(o))",
            L=reversed_league,
        )
        expected = [slot_for_pick(p, 12, "snake", True) for p in range(1, 73)]
        assert js == expected


class TestAdpSurvivalParity:
    def test_matches_across_a_grid(self):
        cases = [
            {"adp": a, "sd": s, "next": n}
            for a in (1.0, 12.5, 40.0, 90.0, 240.0)
            for s in (2.0, 8.0, 25.0)
            for n in (18, 42, 55, 120)
        ]
        js = run_js(
            "console.log(JSON.stringify(IN.cases.map(c=>E.adpSurvival(c.adp,c.sd,c.next))))",
            cases=cases,
        )
        expected = [adp_survival(c["adp"], c["sd"], c["next"]) for c in cases]
        for got, want in zip(js, expected, strict=True):
            # The JS erf is an approximation with ~1.5e-7 absolute error.
            assert got == pytest.approx(want, abs=2e-7)

    def test_missing_adp_is_a_coin_flip_in_both(self):
        js = run_js("console.log(JSON.stringify(E.adpSurvival(null,null,55)))")
        assert js == adp_survival(None, None, 55) == 0.5


class TestUnfilledStartersParity:
    @pytest.mark.parametrize(
        "counts",
        [{}, {"RB": 3}, {"WR": 4}, {"QB": 1, "RB": 2, "WR": 2, "TE": 1},
         {"RB": 1, "WR": 1}, {"QB": 2, "RB": 6, "WR": 6, "TE": 3}],
    )
    def test_matches(self, league, counts):
        js = run_js(
            "console.log(JSON.stringify(E.unfilledStarters(IN.L, IN.counts)))",
            L=JS_LEAGUE, counts=counts,
        )
        positions = [p for p, n in counts.items() for _ in range(n)]
        roster = RosterSnapshot(
            team_id="t", slot=1, player_keys=[f"k{i}" for i in range(len(positions))],
            positions=positions,
        )
        expected = unfilled_starters(league, roster)
        for position, value in expected.items():
            assert js[position] == pytest.approx(value, abs=1e-9)


class TestMarketPriorParity:
    def test_matches(self):
        rows = _board_rows()
        js = run_js(
            "console.log(JSON.stringify(E.marketPrior(IN.board, IN.lo, IN.hi)))",
            board=rows, lo=40, hi=55,
        )
        frame = pl.DataFrame(rows)
        expected = market_prior(frame, 40, 55).mix
        for position, value in expected.items():
            assert js[position] == pytest.approx(value, abs=1e-9)

    def test_sparse_range_widens_identically(self):
        rows = _board_rows()
        js = run_js(
            "console.log(JSON.stringify(E.marketPrior(IN.board, IN.lo, IN.hi)))",
            board=rows, lo=300, hi=305,
        )
        expected = market_prior(pl.DataFrame(rows), 300, 305).mix
        for position, value in expected.items():
            assert js[position] == pytest.approx(value, abs=1e-9)


class TestPositionProbabilitiesParity:
    @pytest.mark.parametrize("round_number", [1, 4, 9, 14])
    @pytest.mark.parametrize("counts", [{}, {"RB": 2, "WR": 2}, {"QB": 1, "TE": 1}])
    def test_matches(self, league, round_number, counts):
        rows = _board_rows()
        prior = market_prior(pl.DataFrame(rows), 40, 55)
        js = run_js(
            "console.log(JSON.stringify(E.positionProbabilities("
            "IN.L, IN.counts, IN.prior, IN.rd)))",
            L=JS_LEAGUE, counts=counts, prior=prior.mix, rd=round_number,
        )
        positions = [p for p, n in counts.items() for _ in range(n)]
        roster = RosterSnapshot(
            team_id="t", slot=1, player_keys=[f"k{i}" for i in range(len(positions))],
            positions=positions,
        )
        expected = position_probabilities(league, roster, prior, round_number)
        for position, value in expected.items():
            assert js[position] == pytest.approx(value, abs=1e-9)


class TestSurvivalParity:
    """The model that actually decides picks. Must match exactly."""

    def test_roster_aware_survival_matches(self, league):
        state = build_fixture_draft(picks_made=41, slot=7)
        rows = _board_rows()
        frame = pl.DataFrame(rows)
        needs = opponent_needs(state, league, frame)
        python = survival_probabilities(state, league, frame, needs)

        js_needs = [
            {"overall": n.pick_overall, "slot": n.slot, "probabilities": n.probabilities}
            for n in needs
        ]
        js = run_js(
            "console.log(JSON.stringify(E.survivalProbabilities(IN.board, IN.needs)))",
            board=rows, needs=js_needs,
        )
        assert js["estimates"]
        for key, estimate in python.estimates.items():
            assert key in js["estimates"], f"{key} missing from the JS result"
            assert js["estimates"][key] == pytest.approx(
                estimate.probability_available, abs=1e-9
            )

    def test_expected_position_losses_match(self, league):
        state = build_fixture_draft(picks_made=41, slot=7)
        rows = _board_rows()
        frame = pl.DataFrame(rows)
        needs = opponent_needs(state, league, frame)
        python = survival_probabilities(state, league, frame, needs)
        js_needs = [
            {"overall": n.pick_overall, "slot": n.slot, "probabilities": n.probabilities}
            for n in needs
        ]
        js = run_js(
            "console.log(JSON.stringify(E.survivalProbabilities(IN.board, IN.needs)))",
            board=rows, needs=js_needs,
        )
        for position, value in python.expected_position_losses.items():
            assert js["losses"][position] == pytest.approx(value, abs=1e-9)


class TestOpponentNeedsParity:
    def test_per_pick_probabilities_match(self, league):
        """Opponent rosters must reconstruct the same way on both sides."""
        state = build_fixture_draft(picks_made=41, slot=7)
        rows = _board_rows()
        needs = opponent_needs(state, league, pl.DataFrame(rows))

        # The JS reconstructs rosters from pick order; hand it the same order.
        order = [p.player_key for p in state.picks]
        board_with_fixture = rows + [
            {"player_key": p.player_key, "position": p.position, "adp": None,
             "adp_sd": None, "player_score": 0.0}
            for p in state.picks
        ]
        js = run_js(
            "console.log(JSON.stringify(E.opponentNeeds("
            "IN.L, IN.order, IN.cur, IN.next, IN.board)))",
            L=JS_LEAGUE, order=order, cur=state.my_current_pick,
            next=state.my_next_pick, board=board_with_fixture,
        )
        assert len(js) == len(needs)
        for js_need, py_need in zip(js, needs, strict=True):
            assert js_need["overall"] == py_need.pick_overall
            assert js_need["slot"] == py_need.slot
            for position, value in py_need.probabilities.items():
                assert js_need["probabilities"][position] == pytest.approx(
                    value, abs=1e-3
                ), f"pick {py_need.pick_overall} {position}"


class TestMonteCarloParity:
    """The simulation cannot share numpy's RNG, so it is held to sampling error."""

    def test_survival_agrees_within_sampling_error(self, league, tmp_config):
        from fantasy_draft.draft.simulator import simulate_to_next_pick

        state = build_fixture_draft(picks_made=41, slot=7)
        rows = _board_rows()
        frame = pl.DataFrame(rows)
        needs = opponent_needs(state, league, frame)
        python = simulate_to_next_pick(
            tmp_config, state, frame, needs, iterations=6000
        )
        js_needs = [
            {"overall": n.pick_overall, "slot": n.slot, "probabilities": n.probabilities}
            for n in needs
        ]
        js = run_js(
            "console.log(JSON.stringify(E.simulate(IN.board, IN.needs, "
            "{iterations:6000, seed:20260831})))",
            board=rows, needs=js_needs,
        )
        shared = [k for k in python.survival if k in js["survival"]][:40]
        assert len(shared) >= 20
        gaps = [abs(python.survival[k] - js["survival"][k]) for k in shared]
        assert max(gaps) < 0.10, f"worst disagreement {max(gaps):.3f}"
        assert sum(gaps) / len(gaps) < 0.035

    def test_two_pick_values_agree_within_sampling_error(self, league, tmp_config):
        from fantasy_draft.draft.simulator import simulate_to_next_pick

        state = build_fixture_draft(picks_made=41, slot=7)
        rows = _board_rows()
        frame = pl.DataFrame(rows).with_columns(
            pl.col("player_score").alias("player_score")
        )
        needs = opponent_needs(state, league, frame)
        candidates = [r["player_key"] for r in rows[:5]]
        python = simulate_to_next_pick(
            tmp_config, state, frame, needs, candidates=candidates, iterations=6000
        )
        js_needs = [
            {"overall": n.pick_overall, "slot": n.slot, "probabilities": n.probabilities}
            for n in needs
        ]
        js = run_js(
            "console.log(JSON.stringify(E.simulate(IN.board, IN.needs, "
            "{iterations:6000, seed:20260831, candidates:IN.cands})))",
            board=rows, needs=js_needs, cands=candidates,
        )
        for key in candidates:
            assert key in js["twoPick"]
            assert js["twoPick"][key]["combined"] == pytest.approx(
                python.two_pick[key].combined, abs=3.0
            ), key

    def test_the_likely_next_position_agrees(self, league, tmp_config):
        """The "then RB" half of the recommendation — omitted from the port at first, so
        the panel rendered a literal question mark."""
        from fantasy_draft.draft.simulator import simulate_to_next_pick

        state = build_fixture_draft(picks_made=41, slot=7)
        rows = _board_rows()
        frame = pl.DataFrame(rows)
        needs = opponent_needs(state, league, frame)
        candidates = [r["player_key"] for r in rows[:4]]
        python = simulate_to_next_pick(
            tmp_config, state, frame, needs, candidates=candidates, iterations=6000
        )
        js_needs = [
            {"overall": n.pick_overall, "slot": n.slot, "probabilities": n.probabilities}
            for n in needs
        ]
        js = run_js(
            "console.log(JSON.stringify(E.simulate(IN.board, IN.needs, "
            "{iterations:6000, seed:20260831, candidates:IN.cands})))",
            board=rows, needs=js_needs, cands=candidates,
        )
        for key in candidates:
            assert js["twoPick"][key]["likely_next_position"] is not None, key
            # The dominant position must match; the exact shares are sampling noise.
            assert (
                js["twoPick"][key]["likely_next_position"]
                == python.two_pick[key].likely_next_position
            ), key
            share = js["twoPick"][key]["position_mix"]
            assert sum(share.values()) == pytest.approx(1.0, abs=0.02)


class TestLineupParity:
    @pytest.mark.parametrize(
        "roster",
        [[], [["RB", 100.0]], [["RB", 100.0], ["RB", 90.0], ["RB", 80.0]],
         [["QB", 92.0], ["WR", 95.0], ["TE", 70.0]]],
    )
    def test_best_lineup_matches(self, league, roster):
        from fantasy_draft.analytics.lineup_value import best_lineup_points

        js = run_js(
            "console.log(JSON.stringify(E.bestLineup(IN.L, IN.roster)))",
            L=JS_LEAGUE, roster=roster,
        )
        expected = best_lineup_points(league, [(p, v) for p, v in roster])
        assert js == pytest.approx(expected, abs=1e-9)


class TestTeamStrengthParity:
    """Coverage is now the headline readout, so the two implementations must agree."""

    def _board_and_league(self):
        rows = [
            {"player_key": "rb1", "position": "RB", "vbd": 124.0, "adp": 5.0, "adp_sd": 3.0,
             "player_score": 90.0},
            {"player_key": "rb2", "position": "RB", "vbd": 100.0, "adp": 9.0, "adp_sd": 3.0,
             "player_score": 88.0},
            {"player_key": "rb3", "position": "RB", "vbd": 70.0, "adp": 30.0, "adp_sd": 6.0,
             "player_score": 80.0},
            {"player_key": "wr1", "position": "WR", "vbd": 110.0, "adp": 6.0, "adp_sd": 3.0,
             "player_score": 89.0},
            {"player_key": "wr2", "position": "WR", "vbd": 89.0, "adp": 12.0, "adp_sd": 4.0,
             "player_score": 85.0},
            {"player_key": "qb1", "position": "QB", "vbd": 106.0, "adp": 40.0, "adp_sd": 8.0,
             "player_score": 84.0},
            {"player_key": "te1", "position": "TE", "vbd": 62.0, "adp": 50.0, "adp_sd": 9.0,
             "player_score": 76.0},
        ]
        return rows

    @pytest.mark.parametrize(
        "mine", [[], ["rb1"], ["rb1", "rb2"], ["rb1", "wr1", "qb1"]]
    )
    def test_required_slots_and_coverage_match(self, league, mine):
        """The Python side is exercised through the same arithmetic, not the service,
        because the service needs a live draft; the numbers are what must agree."""
        rows = self._board_and_league()
        losses = {"RB": 2.0, "WR": 3.0, "QB": 1.0, "TE": 0.5}
        picks = [{"k": k, "mine": True} for k in mine]
        js = run_js(
            "console.log(JSON.stringify(E.teamStrength(IN.L, IN.picks, 7, IN.board, IN.losses, [])))",
            L={**JS_LEAGUE, "flex_eligibility": {"FLEX": ["RB", "WR", "TE"]}},
            picks=picks, board=rows, losses=losses,
        )
        by_position = {r["position"]: r for r in js["positions"]}

        # Required slots must equal Python's starter_demand per team, rounded.
        for position in ("QB", "RB", "WR", "TE"):
            expected = max(1, round(league.starter_demand(position) / league.teams))
            assert by_position[position]["required"] == expected, position

        # Coverage: value held over value of a full complement.
        vbd = {r["player_key"]: r["vbd"] for r in rows}
        position_of = {r["player_key"]: r["position"] for r in rows}
        for position in ("QB", "RB", "WR", "TE"):
            required = by_position[position]["required"]
            held = sorted(
                (vbd[k] for k in mine if position_of[k] == position), reverse=True
            )[:required]
            have = sum(held)
            pool = sorted(
                (r["vbd"] for r in rows if r["position"] == position), reverse=True
            )
            target = have + sum(pool[: max(0, required - len(held))])
            coverage = (have / target * 100) if target > 0 else (100 if required == len(held) else 0)
            assert by_position[position]["coverage"] == pytest.approx(
                round(coverage), abs=1
            ), position

    def test_league_comparison_only_appears_with_opponent_picks(self):
        """The bug this replaced: every position read "1 of 1" when only own picks existed."""
        rows = self._board_and_league()
        L = {**JS_LEAGUE, "flex_eligibility": {"FLEX": ["RB", "WR", "TE"]}}
        only_mine = run_js(
            "console.log(JSON.stringify(E.teamStrength(IN.L, IN.picks, 7, IN.board, {}, [])))",
            L=L, picks=[{"k": "rb1", "mine": True}], board=rows,
        )
        assert only_mine["has_league_comparison"] is False
        assert all(r["league"] is None for r in only_mine["positions"])

        with_others = run_js(
            "console.log(JSON.stringify(E.teamStrength(IN.L, IN.picks, 7, IN.board, {}, [])))",
            L=L,
            picks=[{"k": "rb1", "mine": True}, {"k": "wr1", "mine": False}],
            board=rows,
        )
        assert with_others["has_league_comparison"] is True

    def test_a_filled_position_is_the_lowest_priority(self):
        rows = self._board_and_league()
        js = run_js(
            "console.log(JSON.stringify(E.teamStrength(IN.L, IN.picks, 7, IN.board, {}, [])))",
            L={**JS_LEAGUE, "flex_eligibility": {"FLEX": ["RB", "WR", "TE"]}},
            picks=[{"k": "rb1", "mine": True}, {"k": "rb2", "mine": True}],
            board=rows,
        )
        by_position = {r["position"]: r for r in js["positions"]}
        assert by_position["RB"]["coverage"] == 100
        assert by_position["RB"]["priority"] < by_position["QB"]["priority"]
        assert js["top_priority"] != "RB"
