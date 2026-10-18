"""Fleet-wide test-harness self-healing: `.uv-workspace-siblings/agent-connector-sdk`.

`pyproject.toml`'s `[tool.uv.sources]` resolves the `agent-connector-sdk`
dependency through the gitignored `.uv-workspace-siblings/agent-connector-sdk`
symlink (it must be gitignored — it's a machine-local absolute path). Only
the canonical checkout has it hand-created; a freshly created `git worktree`
never gets one, so the FIRST `uv sync` in a new worktree fails outright:

    error: Distribution not found at:
    file://.../.uv-workspace-siblings/agent-connector-sdk

Three lanes hit this independently on 2026-08-27/28 and each fixed it by
hand. This mirrors the self-healing pattern agent-webui's `pnpm-build`
pre-commit hook already uses for its own missing `node_modules`
(`test -d node_modules || pnpm install`): provision the missing, gitignored,
worktree-local artifact idempotently instead of requiring every lane to
remember to hand-create it.

LIMITATION: this only self-heals a venv that has ALREADY been synced at
least once (e.g. the symlink was later deleted by `git clean`) — once
synced, uv's editable-install finder points at the resolved absolute path,
not the symlink, so pytest runs fine without it from then on. It CANNOT fix
the very first `uv sync` in a brand-new worktree, because pytest itself is
not installed until that sync succeeds. For that case, run this file
directly, once, before your first sync:

    python tests/conftest.py && uv sync --extra test --extra mcp
"""

import pathlib
import subprocess
import sys

import pytest


def _ensure_agent_connector_sdk_sibling() -> "pathlib.Path | None":
    here = pathlib.Path(__file__).resolve().parent  # .../tests
    repo_root = here.parent
    link = repo_root / ".uv-workspace-siblings" / "agent-connector-sdk"
    if link.is_symlink() or link.exists():
        return link

    # Derive the canonical (non-worktree) checkout root from git's common
    # dir — a worktree's own directory name/location tells us nothing about
    # where its sibling `agent-connector-sdk` repo lives (this program's other
    # fleet-wide defect: deriving anything from the worktree basename).
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return None
    common_dir = pathlib.Path(result.stdout.strip())
    canonical_root = common_dir.parent  # .../agent-packages/agents/<this-repo>
    packages_root = canonical_root.parent.parent  # .../agent-packages
    target = packages_root / "agent-connector-sdk"
    if not (target / "pyproject.toml").exists():
        return None

    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except FileExistsError:
        pass
    return link


_ensure_agent_connector_sdk_sibling()

if __name__ == "__main__":
    # Allows `python tests/conftest.py` to bootstrap a fresh worktree BEFORE
    # `uv sync`, without importing anything from this repo's own package or
    # its third-party deps — none of which exist yet in a pre-sync venv.
    raise SystemExit(0)




"""Shared test fixtures for Github Agent."""


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("GITHUB_URL", "https://test.example.com")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-12345")
