"""The HTTP layer.

The API must add nothing. Its job is to hand back exactly what
``analyze_current_pick`` produced, and to turn the engine's human-readable errors into
responses that keep their message rather than flattening to "500 Internal Error".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fantasy_draft.api.service import create_app
from fantasy_draft.draft.fixtures import build_fixture_draft
from fantasy_draft.draft.store import save_state


@pytest.fixture
def client(tmp_config, db):
    db.close()
    return TestClient(create_app(tmp_config), raise_server_exceptions=False)


@pytest.fixture
def client_with_data(tmp_config, db):
    """A client whose database holds one real player, so picks can be resolved."""
    import polars as pl

    db.replace_table(
        "players",
        pl.DataFrame(
            {
                "player_key": ["00-0000001"],
                "full_name": ["Test Player"],
                "normalized_name": ["test player"],
                "position": ["RB"],
                "team": ["KC"],
                "source": ["test"],
                "ingested_at": [__import__("datetime").datetime.now()],
            }
        ),
    )
    db.close()
    client = TestClient(create_app(tmp_config), raise_server_exceptions=False)
    return client, "00-0000001", "Test Player"


@pytest.fixture
def client_with_draft(tmp_config, db):
    save_state(db, build_fixture_draft(picks_made=41, slot=7))
    db.close()
    return TestClient(create_app(tmp_config), raise_server_exceptions=False)


class TestHealthAndConfig:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_config_reports_the_league(self, client):
        payload = client.get("/api/config").json()
        assert payload["league"]["teams"] == 12
        assert payload["league"]["slot"] == 7
        assert payload["league"]["scoring_type"] == "Half-PPR"

    def test_config_reports_provider_state(self, client):
        provider = client.get("/api/config").json()["provider"]
        assert provider["platform"] == "sleeper"
        assert provider["implemented"] is True
        assert "yahoo" in provider["available"]

    def test_config_exposes_the_weights(self, client):
        """Weights live in YAML and are shown, never redefined in the interface."""
        weights = client.get("/api/config").json()["weights"]
        assert sum(weights["draft_now"].values()) == pytest.approx(1.0)


class TestErrors:
    def test_no_draft_returns_409_with_the_fix_in_it(self, client):
        response = client.post("/api/on-clock", json={"refresh": False})
        assert response.status_code == 409
        assert "ff draft mock" in response.json()["detail"]

    def test_unknown_slot_returns_409(self, tmp_config, db):
        state = build_fixture_draft(picks_made=10, slot=7)
        state.my_slot = None
        save_state(db, state)
        db.close()
        response = TestClient(create_app(tmp_config), raise_server_exceptions=False).post(
            "/api/on-clock", json={"refresh": False}
        )
        assert response.status_code == 409
        assert "draft.slot" in response.json()["detail"]

    def test_compare_requires_two_players(self, client_with_draft):
        response = client_with_draft.post("/api/compare", json={"player_keys": ["a"]})
        assert response.status_code == 422

    def test_board_without_data_returns_409(self, client):
        assert client.get("/api/board").status_code == 409


class TestManualPicks:
    """Swipe-to-draft and its undo, over the same store the CLI writes to."""

    def _start(self, client) -> None:
        response = client.post("/api/draft/start", json={"slot": 7, "draft_id": "api-manual"})
        assert response.status_code == 200

    def test_start_then_record_then_undo(self, client_with_data):
        client, key, name = client_with_data
        self._start(client)
        recorded = client.post("/api/pick", json={"player_key": key, "draft_id": "api-manual"})
        assert recorded.status_code == 200
        body = recorded.json()
        assert body["recorded"]["pick_label"] == "1.01"
        assert body["picks_made"] == 1

        undone = client.post("/api/undo", json={"draft_id": "api-manual"})
        assert undone.status_code == 200
        assert undone.json()["removed"]["name"] == name
        assert undone.json()["picks_made"] == 0

    def test_recording_the_same_player_twice_is_refused(self, client_with_data):
        client, key, _ = client_with_data
        self._start(client)
        client.post("/api/pick", json={"player_key": key, "draft_id": "api-manual"})
        again = client.post("/api/pick", json={"player_key": key, "draft_id": "api-manual"})
        assert again.status_code == 409
        assert "already been drafted" in again.json()["detail"]

    def test_unknown_player_key(self, client_with_data):
        client, _, _ = client_with_data
        self._start(client)
        response = client.post("/api/pick", json={"player_key": "nope", "draft_id": "api-manual"})
        assert response.status_code == 404

    def test_pick_without_a_draft(self, client):
        assert client.post("/api/pick", json={"player_key": "x"}).status_code == 409

    def test_pick_needs_a_key_or_a_name(self, client_with_data):
        client, _, _ = client_with_data
        self._start(client)
        assert client.post("/api/pick", json={"draft_id": "api-manual"}).status_code == 422

    def test_start_requires_a_slot_in_range(self, client):
        assert client.post("/api/draft/start", json={"slot": 99}).status_code == 422

    def test_undo_on_an_empty_draft(self, client):
        client.post("/api/draft/start", json={"slot": 7, "draft_id": "api-manual"})
        assert client.post("/api/undo", json={"draft_id": "api-manual"}).json()["removed"] is None


class TestOnClock:
    def test_returns_the_five_areas(self, client_with_draft):
        response = client_with_draft.post(
            "/api/on-clock", json={"refresh": False, "simulate": False}
        )
        assert response.status_code == 200
        payload = response.json()
        for area in ("on_the_clock", "best_available", "my_roster",
                     "who_makes_it_back", "what_if"):
            assert area in payload

    def test_snake_maths_survives_an_empty_board(self, client_with_draft):
        payload = client_with_draft.post(
            "/api/on-clock", json={"refresh": False, "simulate": False}
        ).json()
        area = payload["on_the_clock"]
        assert area["pick_label"] == "4.06"
        assert area["next_pick_label"] == "5.07"
        assert area["picks_until_next"] == 12

    def test_reports_that_the_sync_did_not_run(self, client_with_draft):
        payload = client_with_draft.post("/api/on-clock", json={"refresh": True}).json()
        assert payload["synced"] is False
        assert payload["sync_error"]

    def test_limit_is_validated(self, client_with_draft):
        assert client_with_draft.post(
            "/api/on-clock", json={"limit": 9999}
        ).status_code == 422


class TestPage:
    def test_dashboard_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "ANALYSE PICK" in response.text

    def test_page_contains_all_five_areas(self, client):
        text = client.get("/").text
        for heading in ("On the clock", "Who makes it back to me?", "What if I take",
                        "Best available", "My roster"):
            assert heading in text

    def test_page_is_mobile_first(self, client):
        """The clock does not wait for you to find a laptop."""
        text = client.get("/").text
        assert 'name="viewport"' in text
        assert "viewport-fit=cover" in text
        assert "safe-area-inset-bottom" in text          # notch-safe action bar
        assert "@media(min-width:760px)" in text          # scales up, not down

    def test_page_has_no_external_requests(self, client):
        """Local-first: the page must not phone anywhere."""
        text = client.get("/").text
        for marker in ("http://", "https://", "cdn.", "googleapis"):
            assert marker not in text.replace('xmlns="http://www.w3.org', "")

    def test_openapi_is_available(self, client):
        assert client.get("/api/openapi.json").status_code == 200
