"""Project lookup and placement precedence for CLI create-and-link flows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import pytest
import typer

from hud.cli.utils.project import (
    Project,
    ProjectNotFound,
    ProjectSource,
    list_projects,
    projects_not_enabled,
    resolve_placement,
    resolve_project,
    resolve_writable_placement,
)
from hud.cli.utils.source import EnvironmentSource
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_ID = "11111111-1111-4111-8111-111111111111"
_BROWSER_ID = "22222222-2222-4222-8222-222222222222"
_READONLY_ID = "33333333-3333-4333-8333-333333333333"


def _record(
    project_id: str, name: str, *, is_default: bool = False, create: bool = True
) -> dict[str, Any]:
    return {
        "id": project_id,
        "name": name,
        "is_default": is_default,
        "capabilities": {"view": True, "create": create, "manage": False},
    }


@pytest.fixture
def calls() -> list[str]:
    """URLs the fake platform transport was asked for."""
    return []


@pytest.fixture
def records() -> list[dict[str, Any]]:
    return [
        _record(_DEFAULT_ID, "default", is_default=True),
        _record(_BROWSER_ID, "browser-evals"),
        _record(_READONLY_ID, "locked-down", create=False),
    ]


@pytest.fixture
def platform(
    monkeypatch: pytest.MonkeyPatch, calls: list[str], records: list[dict[str, Any]]
) -> PlatformClient:
    """A client backed by a paginated, searchable Projects API."""

    def fake_request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(url)
        parsed = urlsplit(url)
        params = parse_qs(parsed.query)
        project_id = parsed.path.rsplit("/", 1)[-1]
        if project_id in {_DEFAULT_ID, _BROWSER_ID, _READONLY_ID}:
            return next(record for record in records if record["id"] == project_id)
        search = params.get("search", [""])[0]
        matches = [
            record
            for record in records
            if search in record["name"] or search in (record.get("description") or "")
        ]
        limit = int(params.get("limit", ["50"])[0])
        offset = int(params.get("offset", ["0"])[0])
        return {
            "items": matches[offset : offset + limit],
            "total": len(matches),
            "limit": limit,
            "offset": offset,
        }

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)
    return PlatformClient("https://api.example", "key")


def _no_global_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.settings.settings.default_project", None)


def test_list_reads_paginated_items(platform: PlatformClient) -> None:
    assert [project.name for project in list_projects(platform)] == [
        "default",
        "browser-evals",
        "locked-down",
    ]


@pytest.mark.parametrize("count", [0, 50, 51, 101])
def test_list_returns_every_page(
    platform: PlatformClient, records: list[dict[str, Any]], calls: list[str], count: int
) -> None:
    records[:] = [_record(str(i), f"project-{i}") for i in range(count)]

    assert [project.id for project in list_projects(platform)] == [r["id"] for r in records]
    assert len(calls) == max(1, (count + 49) // 50)


@pytest.mark.parametrize("match_index", [0, 50, 100])
def test_resolve_searches_until_the_exact_name_is_found(
    platform: PlatformClient,
    records: list[dict[str, Any]],
    calls: list[str],
    match_index: int,
) -> None:
    records[:] = [_record(str(i), f"browser-evals-{i}") for i in range(101)]
    records[match_index] = _record(_BROWSER_ID, "browser-evals")

    assert resolve_project(platform, "Browser Evals").id == _BROWSER_ID
    assert len(calls) == match_index // 50 + 1
    assert all(parse_qs(urlsplit(url).query)["search"] == ["browser-evals"] for url in calls)


def test_resolve_exhausts_search_and_lists_all_alternatives_when_no_exact_name_exists(
    platform: PlatformClient, records: list[dict[str, Any]], calls: list[str]
) -> None:
    records[:] = [
        {**_record(str(i), f"project-{i}"), "description": "browser-evals"} for i in range(51)
    ]

    with pytest.raises(ProjectNotFound) as excinfo:
        resolve_project(platform, "browser-evals")

    assert [p.id for p in excinfo.value.available] == [r["id"] for r in records]
    queries = [parse_qs(urlsplit(url).query) for url in calls]
    assert [q.get("search") for q in queries] == [["browser-evals"], ["browser-evals"], None, None]
    assert [q["offset"] for q in queries] == [["0"], ["50"], ["0"], ["50"]]


def test_projects_not_enabled_matches_only_the_feature_gate() -> None:
    assert projects_not_enabled(
        HudRequestError(
            "forbidden",
            status_code=403,
            response_json={"error": "projects_not_enabled", "detail": "Projects disabled"},
        )
    )
    assert projects_not_enabled(
        HudRequestError(
            "forbidden",
            status_code=403,
            response_json={"error": "forbidden", "detail": "Projects are not enabled"},
        )
    )
    assert not projects_not_enabled(
        HudRequestError(
            "forbidden",
            status_code=403,
            response_json={"error": "forbidden", "detail": "Missing create scope"},
        )
    )


def test_resolve_matches_a_normalized_name(platform: PlatformClient) -> None:
    """A human-typed name resolves through the same normalization the platform applies."""
    assert resolve_project(platform, "Browser Evals").id == _BROWSER_ID
    assert resolve_project(platform, "browser-evals").id == _BROWSER_ID


def test_resolve_matches_an_id(platform: PlatformClient) -> None:
    assert resolve_project(platform, _BROWSER_ID).name == "browser-evals"
    assert resolve_project(platform, _BROWSER_ID.upper()).name == "browser-evals"


def test_resolve_reports_the_visible_alternatives(platform: PlatformClient) -> None:
    with pytest.raises(ProjectNotFound) as excinfo:
        resolve_project(platform, "nope")

    assert [p.name for p in excinfo.value.available] == [
        "default",
        "browser-evals",
        "locked-down",
    ]


def test_flag_outranks_directory_config(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hud.settings.settings.default_project", "locked-down")
    source = EnvironmentSource.open(tmp_path)
    source.save_config({"projectId": _DEFAULT_ID})

    placement = resolve_placement(platform, source, flag="browser-evals")

    assert placement.project is not None
    assert placement.project.id == _BROWSER_ID
    assert placement.source is ProjectSource.FLAG


def test_directory_config_applies_without_a_flag(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placement is a property of the environment, not of who deploys it."""
    monkeypatch.setattr("hud.settings.settings.default_project", "default")
    source = EnvironmentSource.open(tmp_path)
    source.save_config({"projectId": _BROWSER_ID})

    placement = resolve_placement(platform, source, flag=None)

    assert placement.project is not None
    assert placement.project.id == _BROWSER_ID
    assert placement.source is ProjectSource.CONFIG


def test_global_default_applies_to_an_unpinned_directory(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hud.settings.settings.default_project", "browser-evals")

    placement = resolve_placement(platform, EnvironmentSource.open(tmp_path), flag=None)

    assert placement.project is not None
    assert placement.project.id == _BROWSER_ID
    assert placement.source is ProjectSource.GLOBAL_DEFAULT


def test_unconfigured_placement_sends_no_project_and_makes_no_call(
    platform: PlatformClient,
    calls: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-config path stays free: no project on the wire, no lookup."""
    _no_global_default(monkeypatch)
    placement = resolve_placement(platform, EnvironmentSource.open(tmp_path), flag=None)

    assert placement.project_id is None
    assert placement.source is ProjectSource.TEAM_DEFAULT
    assert placement.label == "team default Project"
    assert calls == []


def test_placement_resolves_a_project_the_caller_cannot_create_in(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_global_default(monkeypatch)
    source = EnvironmentSource.open(tmp_path)
    placement = resolve_placement(platform, source, flag="locked-down")
    assert placement.project is not None
    assert placement.project.id == _READONLY_ID

    with pytest.raises(typer.Exit):
        resolve_writable_placement(
            platform,
            source,
            flag="locked-down",
            console=HUDConsole(),
        )


def test_from_record_defaults_capabilities_to_read_only() -> None:
    """A response without capabilities is not assumed writable."""
    assert Project.from_record({"id": "x", "name": "y"}).can_create is False
