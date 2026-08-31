"""Synthetic draft fixtures.

Tests and simulations must never depend on a live Sleeper league. This builds a
realistic, fully deterministic draft: a 12-team half-PPR snake, our slot 7, with
managers who take best-available most of the time and fill obvious roster holes the
rest, so position runs and roster needs look like a real room rather than a straight
ADP walk.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from ..config import LeagueConfig
from ..models import DraftPick
from .snake import SnakeBoard
from .state import DraftState

#: A small, realistic 2026-shaped player pool. Names are real; the ordering approximates
#: an August consensus board. Points are indicative, not projections.
FIXTURE_POOL: list[tuple[str, str, str]] = [
    ("Ja'Marr Chase", "WR", "CIN"), ("Jahmyr Gibbs", "RB", "DET"),
    ("Puka Nacua", "WR", "LAR"), ("Bijan Robinson", "RB", "ATL"),
    ("Jaxon Smith-Njigba", "WR", "SEA"), ("Amon-Ra St. Brown", "WR", "DET"),
    ("CeeDee Lamb", "WR", "DAL"), ("Christian McCaffrey", "RB", "SF"),
    ("Malik Nabers", "WR", "NYG"), ("Brian Thomas Jr.", "WR", "JAX"),
    ("Jonathan Taylor", "RB", "IND"), ("Nico Collins", "WR", "HOU"),
    ("Drake London", "WR", "ATL"), ("De'Von Achane", "RB", "MIA"),
    ("Saquon Barkley", "RB", "PHI"), ("Ashton Jeanty", "RB", "LV"),
    ("Brock Bowers", "TE", "LV"), ("Trey McBride", "TE", "ARI"),
    ("Josh Allen", "QB", "BUF"), ("Lamar Jackson", "QB", "BAL"),
    ("James Cook III", "RB", "BUF"), ("Chase Brown", "RB", "CIN"),
    ("Ladd McConkey", "WR", "LAC"), ("Tee Higgins", "WR", "CIN"),
    ("Kenneth Walker III", "RB", "SEA"), ("Breece Hall", "RB", "NYJ"),
    ("Garrett Wilson", "WR", "NYJ"), ("Marvin Harrison Jr.", "WR", "ARI"),
    ("Derrick Henry", "RB", "BAL"), ("Omarion Hampton", "RB", "LAC"),
    ("Jayden Daniels", "QB", "WAS"), ("Joe Burrow", "QB", "CIN"),
    ("George Kittle", "TE", "SF"), ("Sam LaPorta", "TE", "DET"),
    ("Kyren Williams", "RB", "LAR"), ("Javonte Williams", "RB", "DAL"),
    ("DK Metcalf", "WR", "PIT"), ("Terry McLaurin", "WR", "WAS"),
    ("Courtland Sutton", "WR", "DEN"), ("Jaylen Waddle", "WR", "MIA"),
    ("Bucky Irving", "RB", "TB"), ("TreVeyon Henderson", "RB", "NE"),
    ("Jalen Hurts", "QB", "PHI"), ("Patrick Mahomes", "QB", "KC"),
    ("Travis Kelce", "TE", "KC"), ("Tyler Warren", "TE", "IND"),
    ("Jerry Jeudy", "WR", "CLE"), ("Rome Odunze", "WR", "CHI"),
    ("Chuba Hubbard", "RB", "CAR"), ("Tony Pollard", "RB", "TEN"),
    ("Jordan Addison", "WR", "MIN"), ("Khalil Shakir", "WR", "BUF"),
    ("Bo Nix", "QB", "DEN"), ("Caleb Williams", "QB", "CHI"),
    ("David Njoku", "TE", "CLE"), ("Dallas Goedert", "TE", "PHI"),
    ("Jauan Jennings", "WR", "SF"), ("Keon Coleman", "WR", "BUF"),
    ("RJ Harvey", "RB", "DEN"), ("Cam Skattebo", "RB", "NYG"),
    ("Jakobi Meyers", "WR", "LV"), ("Michael Pittman Jr.", "WR", "IND"),
    ("Trevor Lawrence", "QB", "JAX"), ("Justin Herbert", "QB", "LAC"),
    ("Colston Loveland", "TE", "CHI"), ("Evan Engram", "TE", "DEN"),
    ("Jayden Reed", "WR", "GB"), ("Deebo Samuel", "WR", "WAS"),
    ("Isiah Pacheco", "RB", "KC"), ("Rhamondre Stevenson", "RB", "NE"),
    ("Travis Hunter", "WR", "JAX"), ("Matthew Golden", "WR", "GB"),
    ("Kyler Murray", "QB", "ARI"), ("Dak Prescott", "QB", "DAL"),
    ("Jake Ferguson", "TE", "DAL"), ("Mark Andrews", "TE", "BAL"),
    ("Zay Flowers", "WR", "BAL"), ("Chris Godwin", "WR", "TB"),
    ("Tyrone Tracy Jr.", "RB", "NYG"), ("Jaylen Warren", "RB", "PIT"),
]

#: Roughly how many of each position a manager wants, in rough order of urgency.
TARGET_ROSTER: dict[str, int] = {"RB": 5, "WR": 6, "QB": 1, "TE": 1}


def fixture_league(slot: int = 7, teams: int = 12, rounds: int = 15) -> LeagueConfig:
    """The canonical test league: 12-team half-PPR snake, slot 7."""
    return LeagueConfig.model_validate(
        {
            "name": "Fixture League",
            "season": 2026,
            "teams": teams,
            "draft": {"type": "snake", "rounds": rounds, "slot": slot},
            "roster": {
                "qb": 1, "rb": 2, "wr": 2, "te": 1, "flex": 1,
                "k": 1, "dst": 1, "bench": 6, "ir": 1,
            },
            "scoring": {"reception": 0.5},
        }
    )


def build_fixture_draft(
    picks_made: int = 40,
    slot: int = 7,
    teams: int = 12,
    rounds: int = 15,
    seed: int = 20260831,
    need_weight: float = 0.55,
) -> DraftState:
    """Generate a deterministic partially-completed draft.

    ``need_weight`` is the probability a simulated manager reaches past best-available to
    fill a roster hole, which is what produces realistic position runs.
    """
    rng = random.Random(seed)
    board = SnakeBoard(teams=teams, rounds=rounds)
    pool = list(FIXTURE_POOL)
    available = list(range(len(pool)))

    slot_to_team = {s: f"team-{s:02d}" for s in range(1, teams + 1)}
    rosters: dict[int, dict[str, int]] = {s: {} for s in range(1, teams + 1)}
    picks: list[DraftPick] = []
    start = datetime(2026, 8, 30, 19, 0, 0)

    total = min(picks_made, board.total_picks, len(pool))
    for overall in range(1, total + 1):
        picking_slot = board.slot_for(overall)
        roster = rosters[picking_slot]

        # Candidate window: near the top of the board, with some noise, so the fixture
        # is not a straight ADP walk.
        window = available[: min(8, len(available))]
        choice = window[0]
        if rng.random() < need_weight:
            needed = [
                index
                for index in window
                if roster.get(pool[index][1], 0) < TARGET_ROSTER.get(pool[index][1], 0)
            ]
            if needed:
                choice = needed[0] if rng.random() < 0.7 else rng.choice(needed)
        elif len(window) > 1 and rng.random() < 0.3:
            choice = rng.choice(window[:3])

        name, position, team = pool[choice]
        available.remove(choice)
        roster[position] = roster.get(position, 0) + 1

        picks.append(
            DraftPick(
                overall=overall,
                round=board.round_for(overall),
                slot=picking_slot,
                team_id=slot_to_team[picking_slot],
                player_key=f"fx-{choice:03d}",
                player_name=name,
                position=position,
                nfl_team=team,
                picked_at=start + timedelta(seconds=45 * overall),
            )
        )

    return DraftState(
        draft_id="fixture-2026",
        platform="fixture",
        season=2026,
        board=board,
        picks=picks,
        slot_to_team=slot_to_team,
        team_names={team_id: f"Manager {s}" for s, team_id in slot_to_team.items()},
        my_slot=slot,
        my_team_id=slot_to_team[slot],
        status="drafting",
        synced_at=datetime(2026, 8, 30, 19, 30, 0),
    )


def fixture_pool_keys() -> dict[str, tuple[str, str, str]]:
    """``player_key -> (name, position, team)`` for the fixture pool."""
    return {f"fx-{i:03d}": entry for i, entry in enumerate(FIXTURE_POOL)}
