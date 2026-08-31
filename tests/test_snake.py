"""Snake-draft arithmetic.

Snake math cannot be wrong during a live draft, so this file is deliberately
exhaustive: known-good tables, round trips over every pick of a full draft, and
property checks across many league sizes.
"""

from __future__ import annotations

import pytest

from fantasy_draft.draft.snake import (
    BACKWARD,
    FORWARD,
    SnakeBoard,
    next_pick,
    pick_label,
    pick_number,
    picks_for_slot,
    picks_until_next_turn,
    round_direction,
    round_for_pick,
    slot_for_pick,
)


class TestDirection:
    def test_standard_snake_alternates(self):
        assert [round_direction(r) for r in range(1, 7)] == [
            FORWARD, BACKWARD, FORWARD, BACKWARD, FORWARD, BACKWARD
        ]

    def test_linear_is_always_forward(self):
        assert all(round_direction(r, "linear") == FORWARD for r in range(1, 20))

    def test_third_round_reversal_repeats_round_two(self):
        # R1 fwd, R2 back, R3 repeats R2, then the snake resumes.
        dirs = [round_direction(r, "snake", third_round_reversal=True) for r in range(1, 8)]
        assert dirs == [FORWARD, BACKWARD, BACKWARD, FORWARD, BACKWARD, FORWARD, BACKWARD]

    def test_round_must_be_positive(self):
        with pytest.raises(ValueError):
            round_direction(0)


class TestPickNumber:
    @pytest.mark.parametrize(
        ("round_", "slot", "expected"),
        [
            (1, 1, 1), (1, 7, 7), (1, 12, 12),
            (2, 12, 13), (2, 7, 18), (2, 1, 24),
            (3, 1, 25), (3, 7, 31), (3, 12, 36),
            (4, 12, 37), (4, 7, 42), (4, 1, 48),
        ],
    )
    def test_known_12_team_values(self, round_, slot, expected):
        assert pick_number(round_, slot, 12) == expected

    def test_matches_the_spec_example(self):
        """A 12-team league, slot 7: 1.07, 2.06, 3.07, 4.06."""
        board = SnakeBoard(teams=12, rounds=15)
        labels = [board.label(p) for p in board.picks_for(7)[:4]]
        assert labels == ["1.07", "2.06", "3.07", "4.06"]

    def test_linear_never_snakes(self):
        assert [pick_number(r, 3, 10, "linear") for r in (1, 2, 3)] == [3, 13, 23]

    def test_rejects_out_of_range_slot(self):
        with pytest.raises(ValueError, match="slot must be between"):
            pick_number(1, 13, 12)
        with pytest.raises(ValueError, match="slot must be between"):
            pick_number(1, 0, 12)


class TestInverse:
    @pytest.mark.parametrize("teams", [4, 8, 10, 12, 14, 16, 32])
    @pytest.mark.parametrize("third_round_reversal", [False, True])
    def test_round_trip_over_a_full_draft(self, teams, third_round_reversal):
        rounds = 16
        for overall in range(1, teams * rounds + 1):
            round_ = round_for_pick(overall, teams)
            slot = slot_for_pick(overall, teams, "snake", third_round_reversal)
            assert pick_number(round_, slot, teams, "snake", third_round_reversal) == overall

    @pytest.mark.parametrize("teams", [8, 12, 14])
    def test_every_slot_picks_exactly_once_per_round(self, teams):
        board = SnakeBoard(teams=teams, rounds=10)
        for round_ in range(1, 11):
            slots = {board.slot_for(p) for p in range((round_ - 1) * teams + 1, round_ * teams + 1)}
            assert slots == set(range(1, teams + 1))

    def test_picks_partition_the_draft(self):
        board = SnakeBoard(teams=12, rounds=15)
        everything = [p for slot in range(1, 13) for p in board.picks_for(slot)]
        assert sorted(everything) == list(range(1, 181))


class TestLabels:
    @pytest.mark.parametrize(
        ("overall", "label"),
        [(1, "1.01"), (12, "1.12"), (13, "2.01"), (18, "2.06"), (42, "4.06"), (180, "15.12")],
    )
    def test_label_format(self, overall, label):
        assert pick_label(overall, 12) == label


class TestNextPick:
    def test_next_pick_from_our_own_selection(self):
        # Slot 7 in a 12-team snake: 1.07 (7) -> 2.06 (18).
        assert next_pick(7, 7, 12, 15) == 18
        assert next_pick(18, 7, 12, 15) == 31

    def test_next_pick_from_someone_elses_selection(self):
        assert next_pick(10, 7, 12, 15) == 18

    def test_next_pick_returns_none_at_the_end(self):
        board = SnakeBoard(teams=12, rounds=15)
        last = board.picks_for(7)[-1]
        assert board.next_pick(last, 7) is None

    def test_picks_until_next_turn(self):
        # 1.07 -> 2.06 means 10 other managers pick in between.
        assert picks_until_next_turn(7, 7, 12, 15) == 10
        # 2.06 -> 3.07 means 12 pick in between.
        assert picks_until_next_turn(18, 7, 12, 15) == 12

    def test_turn_advantage_at_the_ends(self):
        # Slot 1 gets 2.12 and 3.01 back to back: zero picks in between.
        assert picks_until_next_turn(24, 1, 12, 15) == 0
        # Slot 12 gets 1.12 and 2.01 back to back.
        assert picks_until_next_turn(12, 12, 12, 15) == 0

    def test_third_round_reversal_moves_the_turn_advantage(self):
        standard = SnakeBoard(teams=12, rounds=15)
        reversal = SnakeBoard(teams=12, rounds=15, third_round_reversal=True)
        # Standard: slot 1 owns the 2/3 turn. 3RR: slot 12 does.
        assert standard.picks_for(1)[:3] == [1, 24, 25]
        assert reversal.picks_for(1)[:3] == [1, 24, 36]
        assert reversal.picks_for(12)[:3] == [12, 13, 25]


class TestSnakeBoard:
    def test_total_picks(self):
        assert SnakeBoard(teams=12, rounds=15).total_picks == 180

    def test_slots_between_lists_the_intervening_managers(self):
        board = SnakeBoard(teams=12, rounds=15)
        between = board.slots_between(7, 18)
        assert [p for p, _ in between] == list(range(8, 18))
        # Round 1 tail runs up to slot 12, then round 2 comes back down.
        assert [s for _, s in between] == [8, 9, 10, 11, 12, 12, 11, 10, 9, 8]

    def test_slots_between_is_empty_for_adjacent_picks(self):
        assert SnakeBoard(teams=12, rounds=15).slots_between(12, 13) == []

    def test_slots_between_clamps_to_the_draft(self):
        board = SnakeBoard(teams=12, rounds=2)
        assert board.slots_between(20, 999)[-1][0] == 24

    def test_from_league(self, league):
        board = SnakeBoard.from_league(league)
        assert (board.teams, board.rounds, board.draft_type) == (12, 15, "snake")

    def test_rejects_bad_dimensions(self):
        with pytest.raises(ValueError):
            SnakeBoard(teams=0, rounds=15)
        with pytest.raises(ValueError):
            SnakeBoard(teams=12, rounds=0)


class TestOddLeagueSizes:
    """Snake math must not assume an even number of teams."""

    @pytest.mark.parametrize("teams", [3, 5, 11])
    def test_odd_team_counts_round_trip(self, teams):
        for overall in range(1, teams * 8 + 1):
            slot = slot_for_pick(overall, teams)
            round_ = round_for_pick(overall, teams)
            assert pick_number(round_, slot, teams) == overall

    def test_picks_for_slot_is_sorted_and_unique(self):
        picks = picks_for_slot(3, 11, 12)
        assert picks == sorted(picks)
        assert len(set(picks)) == len(picks)
