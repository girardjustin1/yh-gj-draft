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

    def test_page_contains_every_area(self, client):
        text = client.get("/").text
        for heading in ("Team strength &amp; what you need", "Your next pick", "Players",
                        "Who makes it back to me?", "Both picks, priced",
                        "Starting lineup"):
            assert heading in text

    def test_widgets_are_in_the_requested_order(self, client):
        """Top-down: where I stand, what to take, then the pool.

        Asserted rather than assumed, because a later edit that reorders the body would
        otherwise silently undo a layout the user asked for specifically.
        """
        text = client.get("/").text
        order = ["Team strength &amp; what you need", "Your next pick", "Players"]
        positions = [text.index(h) for h in order]
        assert positions == sorted(positions), "widgets are out of order"

    def test_next_pick_shows_a_two_position_path(self, client):
        """"Take X now, then RB" — the two-pick model made visible."""
        text = client.get("/").text
        assert "pathline" in text
        assert "thenpos" in text
        assert "likely_next_position" in text

    def test_player_tabs_are_available_myteam_drafted(self, client):
        text = client.get("/").text
        assert '["available","Available"]' in text
        assert '["roster","My team"]' in text
        assert '["gone","Drafted"]' in text

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


class TestAccessControl:
    """Exposing the engine beyond localhost must not be silently open.

    These exist because the first hand-test of this appeared to pass while actually
    hitting a stale server on the same port — auth that is merely believed to work is
    worse than none.
    """

    @pytest.fixture
    def guarded(self, tmp_config, db, monkeypatch):
        monkeypatch.setenv("FF_ACCESS_TOKEN", "s3cret-token")
        db.close()
        return TestClient(create_app(tmp_config), raise_server_exceptions=False)

    def test_no_token_is_rejected(self, guarded):
        response = guarded.get("/api/health")
        assert response.status_code == 401
        assert "k=" in response.json()["detail"]

    def test_wrong_token_is_rejected(self, guarded):
        assert guarded.get("/api/health", params={"k": "nope"}).status_code == 401

    def test_query_parameter_works(self, guarded):
        assert guarded.get("/api/health", params={"k": "s3cret-token"}).status_code == 200

    def test_header_works(self, guarded):
        response = guarded.get("/api/health", headers={"X-FF-Key": "s3cret-token"})
        assert response.status_code == 200

    def test_cookie_is_set_so_a_phone_needs_the_link_once(self, guarded):
        response = guarded.get("/", params={"k": "s3cret-token"})
        assert response.status_code == 200
        assert "ff_key" in response.cookies

    def test_the_page_itself_is_protected(self, guarded):
        """Not just the API — the board is on the page too."""
        assert guarded.get("/").status_code == 401

    def test_writes_are_protected(self, guarded):
        assert guarded.post("/api/pick", json={"player_key": "x"}).status_code == 401

    def test_no_token_configured_leaves_everything_open(self, client):
        """Localhost default: unauthenticated, which is correct for a loopback bind."""
        assert client.get("/api/health").status_code == 200


class TestInsecureBindRefusal:
    def test_binding_publicly_without_a_token_is_refused(self, monkeypatch):
        from fantasy_draft.api.service import InsecureBindError, run

        monkeypatch.delenv("FF_ACCESS_TOKEN", raising=False)
        with pytest.raises(InsecureBindError, match="without an access token"):
            run(host="0.0.0.0", port=8123)

    def test_loopback_needs_no_token(self, monkeypatch):
        """Refusal must not block the normal local case."""
        import fantasy_draft.api.service as svc

        monkeypatch.delenv("FF_ACCESS_TOKEN", raising=False)
        started = {}
        monkeypatch.setattr(
            "uvicorn.run", lambda *a, **k: started.update(k) or None
        )
        svc.run(host="127.0.0.1", port=8123)
        assert started["host"] == "127.0.0.1"

    def test_public_bind_with_a_token_is_allowed(self, monkeypatch):
        import fantasy_draft.api.service as svc

        monkeypatch.setenv("FF_ACCESS_TOKEN", "abc")
        started = {}
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: started.update(k) or None)
        svc.run(host="0.0.0.0", port=8123)
        assert started["host"] == "0.0.0.0"


class TestDraftControls:
    """Two explicit controls, because inferring ownership from pick order confused me
    into shipping something the user could not follow: DRAFT means I took him, swipe
    means someone else did."""

    def test_page_offers_both_controls(self, client):
        page = client.get("/").text
        assert "DRAFT" in page                 # the blue button
        assert ">TAKEN<" in page               # the swipe reveal
        assert "data-act=\"draft\"" in page

    def test_page_has_a_search_box(self, client):
        page = client.get("/").text
        assert 'id="searchBox"' in page
        assert "matchesQuery" in page

    def test_page_explains_itself(self, client):
        """"I don't get how this app works" is a bug in the page, not the reader."""
        page = client.get("/").text
        assert "How this works" in page
        assert "Set your draft slot" in page

    def test_update_board_refetches_rather_than_re_rendering(self, client):
        """The button said UPDATE BOARD while only re-rendering what was already shown.

        Every DRAFT tap and swipe already re-renders, so it was a no-op with a label that
        implied it fetched fresh projections. It now re-reads board.json, which is the
        only way a static build ever sees a new export.
        """
        page = client.get("/").text
        assert "async reload()" in page
        assert "board.json?t=" in page          # cache-busted refetch
        assert "offlineUpdate" in page

    def test_the_board_date_is_visible(self, client):
        """You cannot judge a board without knowing when it was scored."""
        assert "Scored <b>${esc(d.generated_at" in client.get("/").text

    def test_page_shows_team_strength(self, client):
        page = client.get("/").text
        assert "Team strength &amp; what you need" in page
        assert "renderStrength" in page

    def test_mine_flag_puts_the_player_on_my_roster(self, client_with_data):
        client, key, name = client_with_data
        client.post("/api/draft/start", json={"slot": 7, "draft_id": "ctl"})
        # Pick 1.01 is slot 1, not ours — but we say it is ours explicitly.
        response = client.post(
            "/api/pick", json={"player_key": key, "draft_id": "ctl", "mine": True}
        )
        assert response.status_code == 200
        assert response.json()["recorded"]["was_ours"] is True

    def test_theirs_flag_keeps_the_player_off_my_roster(self, client_with_data):
        client, key, _ = client_with_data
        # Slot 1 so that pick 1.01 would otherwise be inferred as ours.
        client.post("/api/draft/start", json={"slot": 1, "draft_id": "ctl2"})
        response = client.post(
            "/api/pick", json={"player_key": key, "draft_id": "ctl2", "mine": False}
        )
        assert response.status_code == 200
        assert response.json()["recorded"]["was_ours"] is False

    def test_ownership_is_still_inferred_when_unspecified(self, client_with_data):
        client, key, _ = client_with_data
        client.post("/api/draft/start", json={"slot": 1, "draft_id": "ctl3"})
        response = client.post("/api/pick", json={"player_key": key, "draft_id": "ctl3"})
        assert response.json()["recorded"]["was_ours"] is True
