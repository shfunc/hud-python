"""Project placement behavior for ``hud sync tasks``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

import hud.cli.sync as sync_module
from hud.eval import Task, Taskset
from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    from pathlib import Path


class _ReadOnlyPlatform:
    def get(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert url == "/projects"
        return {
            "items": [
                {
                    "id": "33333333-3333-4333-8333-333333333333",
                    "name": "locked-down",
                    "capabilities": {"create": False},
                }
            ]
        }


class _WritablePlatform:
    def get(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert url == "/projects"
        return {
            "items": [
                {
                    "id": "22222222-2222-4222-8222-222222222222",
                    "name": "browser-evals",
                    "capabilities": {"create": True},
                }
            ]
        }


def _run_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remote: Taskset,
    *,
    dry_run: bool,
) -> None:
    task = Task(env="example", id="solve", slug="one")
    local = Taskset("demo", [task])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sync_module, "require_api_key", lambda _: None)
    monkeypatch.setattr(
        sync_module.PlatformClient,
        "from_settings",
        lambda: _ReadOnlyPlatform(),
    )
    monkeypatch.setattr(sync_module, "_load_local_taskset", lambda *args, **kwargs: local)
    monkeypatch.setattr(sync_module, "_fetch_remote_taskset", lambda *args, **kwargs: remote)
    monkeypatch.setattr(
        sync_module,
        "upload_taskset",
        lambda *args, **kwargs: pytest.fail("read-only no-op must not upload"),
    )

    sync_module.sync_tasks_command(
        taskset="demo",
        source=".",
        taskset_id=None,
        project="locked-down",
        task_filter=None,
        exclude=None,
        yes=True,
        dry_run=dry_run,
        force=False,
        export=None,
    )


def test_read_only_project_allows_up_to_date_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = Task(env="example", id="solve", slug="one")
    _run_sync(monkeypatch, tmp_path, Taskset("demo", [task]), dry_run=False)


def test_read_only_project_allows_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _run_sync(monkeypatch, tmp_path, Taskset("demo", []), dry_run=True)


def test_project_override_does_not_pin_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = Task(env="example", id="solve", slug="one")
    local = Taskset("demo", [task])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sync_module, "require_api_key", lambda _: None)
    monkeypatch.setattr(
        sync_module.PlatformClient,
        "from_settings",
        lambda: _WritablePlatform(),
    )
    monkeypatch.setattr(sync_module, "_load_local_taskset", lambda *args, **kwargs: local)
    monkeypatch.setattr(
        sync_module,
        "_fetch_remote_taskset",
        lambda *args, **kwargs: Taskset("demo", []),
    )

    def upload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["project_id"] == "22222222-2222-4222-8222-222222222222"
        return {"taskset_id": "taskset-1", "tasks_created": 1}

    monkeypatch.setattr(sync_module, "upload_taskset", upload)

    sync_module.sync_tasks_command(
        taskset="demo",
        source=".",
        taskset_id=None,
        project="browser-evals",
        task_filter=None,
        exclude=None,
        yes=True,
        dry_run=False,
        force=False,
        export=None,
    )

    config = json.loads((tmp_path / ".hud" / "config.json").read_text())
    assert config == {"tasksetId": "taskset-1"}


@pytest.mark.parametrize("status_code", [400, 403, 500])
def test_rejected_upload_exits_with_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status_code: int
) -> None:
    project_id = "22222222-2222-4222-8222-222222222222"
    detail = "Taskset belongs to another Project" if status_code == 400 else "Upload rejected"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.hud_api_url", "https://api.example")
    (tmp_path / "tasks.json").write_text(
        json.dumps([{"env": "example", "id": "solve", "slug": "one"}])
    )

    def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if method == "GET" and url == f"https://api.example/v2/projects/{project_id}":
            return {"id": project_id, "name": "browser-evals", "capabilities": {"create": True}}
        assert method == "POST" and url == "https://api.example/v2/tasks/upload"
        assert kwargs["json"]["project_id"] == project_id
        assert len(kwargs["json"]["tasks"]) == 1
        raise HudRequestError(detail, status_code=status_code, response_json={"detail": detail})

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)

    result = CliRunner().invoke(
        sync_module.sync_app,
        ["tasks", "demo", "tasks.json", "--project", project_id, "--force", "--yes"],
    )

    assert result.exit_code == 1
    assert detail in result.output
    assert "Sync complete" not in result.output
    assert not (tmp_path / ".hud" / "config.json").exists()
