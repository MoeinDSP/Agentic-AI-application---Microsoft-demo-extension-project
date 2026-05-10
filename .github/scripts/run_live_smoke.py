from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.cicd.metadata import load_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", required=True)
    args = parser.parse_args()

    manifest = load_manifests()[args.service]
    command = manifest.ci.get("live_smoke_command")
    if not command:
        raise ValueError(f"{args.service} does not declare a live smoke command")

    env = os.environ.copy()
    for key, definition in manifest.ci.get("live_smoke_env", {}).items():
        source = definition["source"]
        if source == "literal":
            env[key] = str(definition["value"])
            continue
        source_name = definition["name"]
        value = os.environ.get(source_name, "").strip()
        if value:
            env[key] = value

    subprocess.run(
        command,
        check=True,
        cwd=REPO_ROOT / manifest.ci["working_directory"],
        env=env,
        shell=True,
    )


if __name__ == "__main__":
    main()
