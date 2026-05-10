from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.cicd.metadata import (
    json_dumps,
    load_manifests,
    select_ci_services,
    select_deploy_services,
    select_live_smoke_services,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", default="HEAD")
    parser.add_argument("--event-name", required=True)
    args = parser.parse_args()

    changed_files = _changed_files(
        base_sha=args.base_sha.strip(),
        head_sha=args.head_sha.strip(),
        event_name=args.event_name,
    )
    manifests = load_manifests()
    ci_services = select_ci_services(manifests, changed_files)
    gcp_services = select_deploy_services(
        manifests,
        changed_files,
        backend="gcp-cloud-run",
    )
    live_smoke_services = select_live_smoke_services(manifests)

    _write_output(
        "changed_files_json",
        json_dumps(changed_files),
    )
    _write_output(
        "ci_matrix",
        json_dumps([service.to_ci_matrix_entry() for service in ci_services]),
    )
    _write_output(
        "has_ci_services",
        str(bool(ci_services)).lower(),
    )
    _write_output(
        "gcp_deploy_services",
        json_dumps([service.service_name for service in gcp_services]),
    )
    _write_output(
        "has_gcp_deploy_services",
        str(bool(gcp_services)).lower(),
    )
    _write_output(
        "live_smoke_matrix",
        json_dumps([service.to_live_smoke_entry() for service in live_smoke_services]),
    )


def _changed_files(*, base_sha: str, head_sha: str, event_name: str) -> list[str]:
    if not base_sha:
        if event_name == "pull_request":
            raise ValueError("base SHA is required for pull_request discovery")
        return _git_diff_tree(head_sha or "HEAD")
    return _git_diff_names([base_sha, head_sha])


def _git_diff_names(revisions: list[str]) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", *revisions],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _git_diff_tree(revision: str) -> list[str]:
    completed = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", revision],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _write_output(name: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    output_path = Path(github_output) if github_output else Path.cwd() / ".discover-output"
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


if __name__ == "__main__":
    main()
