"""
Operator routes must not be reachable by a customer.

/metrics, /sessions, /dashboard and /history expose every conversation across
all customers, so they are gated. These tests run offline — none of them
reaches the model — and are the cheapest guard against a refactor quietly
re-opening the admin surface.
"""

import importlib
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TRENDLY_NOW", "2026-08-05T12:00:00Z")

OPERATOR_ROUTES = ["/metrics", "/sessions", "/dashboard", "/history"]


def _client(token: str | None):
    """Rebuild the app with a given ADMIN_TOKEN — it is read at import time."""
    if token is None:
        os.environ.pop("ADMIN_TOKEN", None)
    else:
        os.environ["ADMIN_TOKEN"] = token
    import app.main
    importlib.reload(app.main)
    return TestClient(app.main.app), app.main


@pytest.mark.parametrize("route", OPERATOR_ROUTES)
def test_operator_routes_reject_a_wrong_token(route):
    client, _ = _client("s3cret-token")
    # 404 rather than 403: an unauthorised caller shouldn't learn the route exists.
    assert client.get(route, headers={"x-admin-token": "wrong"}).status_code == 404
    assert client.get(route).status_code == 404


@pytest.mark.parametrize("route", OPERATOR_ROUTES)
def test_operator_routes_accept_the_right_token(route):
    client, _ = _client("s3cret-token")
    assert client.get(route, headers={"x-admin-token": "s3cret-token"}).status_code == 200
    assert client.get(route, params={"token": "s3cret-token"}).status_code == 200


def test_health_hides_configuration_from_customers():
    client, _ = _client("s3cret-token")
    public = client.get("/health").json()
    assert public == {"status": "ok"}
    assert "model" not in public

    admin = client.get("/health", headers={"x-admin-token": "s3cret-token"}).json()
    assert admin["model"]
    assert admin["admin_auth"] == "token"


def test_customer_routes_stay_open():
    """Gating the operator surface must not lock the customer out."""
    client, _ = _client("s3cret-token")
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    # Session replay is how the customer's own page restores its conversation.
    assert client.get("/session/does-not-exist").status_code == 200


def test_chat_page_links_to_no_operator_route():
    _, main = _client("s3cret-token")
    page = main.CHAT_PAGE
    for route in OPERATOR_ROUTES:
        assert f'href="{route}"' not in page, f"customer page links to {route}"


def test_localhost_only_when_no_token_configured():
    """With ADMIN_TOKEN unset the routes still work locally, so a developer
    isn't forced to configure auth to see their own dashboard."""
    client, _ = _client(None)
    for route in OPERATOR_ROUTES:
        assert client.get(route).status_code == 200
    assert client.get("/health").json()["admin_auth"] == "localhost-only"
