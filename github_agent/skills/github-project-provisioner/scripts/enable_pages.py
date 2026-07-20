import os
import re
import sys

from github_agent.github_http import github_json_request

REPOSITORY_PART_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def enable_pages(owner, repo):
    if REPOSITORY_PART_RE.fullmatch(owner) is None or REPOSITORY_PART_RE.fullmatch(repo) is None:
        print("Invalid repository identifier.", file=sys.stderr)
        raise SystemExit(2)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "Error: GITHUB_TOKEN environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure pages to build from GitHub Actions workflow
    try:
        status, _ = github_json_request(
            "POST",
            f"/repos/{owner}/{repo}/pages",
            token,
            {"build_type": "workflow"},
        )
        if 200 <= status < 300:
            print(f"Successfully enabled GitHub Pages for {owner}/{repo}.")
            return
        if status == 409:
            print(
                "GitHub Pages might already be enabled or updating."
            )
        else:
            print(
                f"Failed to enable GitHub Pages: HTTP {status}",
                file=sys.stderr,
            )
            raise SystemExit(1)
    except RuntimeError:
        print("GitHub Pages request failed.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python enable_pages.py <owner> <repo_name>")
        sys.exit(1)

    enable_pages(sys.argv[1], sys.argv[2])
