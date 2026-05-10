from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.cicd.gcp_cloud_run import (
    build_image_uri,
    build_run_deploy_command,
    build_submit_command,
    fetch_service_url,
    resolve_env_vars,
    resolve_secret_bindings,
    smoke_check_service,
)
from tools.cicd.metadata import load_manifests, order_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--services", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", default=os.environ.get("GITHUB_SHA", "local")[:12])
    args = parser.parse_args()

    service_names = [name.strip() for name in args.services.split(",") if name.strip()]
    if not service_names:
        print("No GCP services selected for deployment.")
        return

    manifests = load_manifests()
    selected = [manifests[name] for name in service_names]
    ordered = order_manifests(manifests, selected)
    deployed_urls: dict[str, str] = {}

    for manifest in ordered:
        image_uri = build_image_uri(
            project_id=args.project_id,
            region=args.region,
            repository=args.repository,
            service_name=manifest.service_name,
            tag=args.tag,
        )
        _run(build_submit_command(manifest, image_uri=image_uri))
        env_vars = resolve_env_vars(
            manifest,
            deployed_urls=deployed_urls,
            region=args.region,
            allow_missing_service_url=True,
        )
        secret_bindings = resolve_secret_bindings(manifest)
        _run(
            build_run_deploy_command(
                manifest,
                image_uri=image_uri,
                region=args.region,
                env_vars=env_vars,
                secret_bindings=secret_bindings,
            )
        )
        service_url = fetch_service_url(
            manifest.deploy["cloud_run_service"],
            region=args.region,
        )
        deployed_urls[manifest.service_name] = service_url
        final_env_vars = resolve_env_vars(
            manifest,
            deployed_urls=deployed_urls,
            region=args.region,
            service_url=service_url,
        )
        if final_env_vars != env_vars:
            _run(
                build_run_deploy_command(
                    manifest,
                    image_uri=image_uri,
                    region=args.region,
                    env_vars=final_env_vars,
                    secret_bindings=secret_bindings,
                )
            )
            service_url = fetch_service_url(
                manifest.deploy["cloud_run_service"],
                region=args.region,
            )
            deployed_urls[manifest.service_name] = service_url
        smoke_check_service(
            manifest,
            service_url=service_url,
        )
        print(f"Deployed {manifest.service_name} to {service_url}")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
