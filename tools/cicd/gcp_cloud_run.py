from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tools.cicd.metadata import REPO_ROOT, ServiceManifest, load_manifests


def build_image_uri(
    *,
    project_id: str,
    region: str,
    repository: str,
    service_name: str,
    tag: str,
) -> str:
    return f"{region}-docker.pkg.dev/{project_id}/{repository}/{service_name}:{tag}"


def resolve_env_vars(
    manifest: ServiceManifest,
    *,
    deployed_urls: dict[str, str],
    region: str,
    service_url: str | None = None,
    allow_missing_service_url: bool = False,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, definition in manifest.deploy.get("required_env", {}).items():
        source = definition["source"]
        if source == "literal":
            resolved[key] = str(definition["value"])
            continue
        if source == "github_var":
            value = os.environ.get(definition["name"], "").strip()
            if not value:
                raise ValueError(f"missing GitHub variable: {definition['name']}")
            resolved[key] = value
            continue
        if source == "dependency_url":
            dependency = definition["service"]
            resolved[key] = deployed_urls.get(dependency) or fetch_service_url(
                _cloud_run_service_name(load_manifests()[dependency]),
                region=region,
            )
            continue
        if source == "service_url":
            if service_url:
                resolved[key] = service_url.rstrip("/")
                continue
            if allow_missing_service_url:
                continue
            resolved[key] = fetch_service_url(
                _cloud_run_service_name(manifest),
                region=region,
            ).rstrip("/")
            continue
        raise ValueError(f"unsupported env source: {source}")
    return resolved


def resolve_secret_bindings(manifest: ServiceManifest) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for key, definition in manifest.deploy.get("required_secrets", {}).items():
        bindings[key] = f"{definition['secret_name']}:{definition['version']}"
    return bindings


def resolve_runtime_service_account(manifest: ServiceManifest) -> str | None:
    definition = manifest.deploy.get("runtime_service_account")
    if not definition:
        return None
    source = definition["source"]
    if source != "github_var":
        raise ValueError(f"unsupported runtime service account source: {source}")
    value = os.environ.get(definition["name"], "").strip()
    if not value:
        raise ValueError(f"missing GitHub variable: {definition['name']}")
    return value


def build_run_deploy_command(
    manifest: ServiceManifest,
    *,
    image_uri: str,
    region: str,
    env_vars: dict[str, str],
    secret_bindings: dict[str, str],
    runtime_service_account: str | None = None,
) -> list[str]:
    command = [
        "gcloud",
        "run",
        "deploy",
        _cloud_run_service_name(manifest),
        "--image",
        image_uri,
        "--platform",
        "managed",
        "--region",
        region,
        "--port",
        str(manifest.runtime_port),
        "--quiet",
    ]
    if manifest.deploy.get("allow_unauthenticated", False):
        command.append("--allow-unauthenticated")
    else:
        command.append("--no-allow-unauthenticated")
    if runtime_service_account:
        command.extend(["--service-account", runtime_service_account])
    if env_vars:
        command.extend(["--set-env-vars", ",".join(f"{key}={value}" for key, value in env_vars.items())])
    if secret_bindings:
        command.extend(
            [
                "--set-secrets",
                ",".join(f"{key}={value}" for key, value in secret_bindings.items()),
            ]
        )
    return command


def fetch_service_url(service_name: str, *, region: str) -> str:
    completed = subprocess.run(
        [
            "gcloud",
            "run",
            "services",
            "describe",
            service_name,
            "--region",
            region,
            "--format=value(status.url)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def smoke_check_service(
    manifest: ServiceManifest,
    *,
    service_url: str,
) -> None:
    normalized_service_url = service_url.rstrip("/")
    headers = _cloud_run_auth_headers(normalized_service_url)
    _http_request("GET", f"{normalized_service_url}{manifest.healthcheck}", headers=headers)
    smoke_type = manifest.smoke_check_type
    if smoke_type == "route-estimate":
        payload = {
            "origin": {"lat": 41.8902, "lng": 12.4922},
            "destination": {"lat": 41.8986, "lng": 12.4769},
            "mode": "walking",
            "departure_time": "2026-05-10T09:00:00Z",
        }
        _http_request(
            "POST",
            f"{service_url.rstrip('/')}/v1/tools/route-estimate",
            payload,
            headers=headers,
        )
        return
    if smoke_type == "fasta2a-day-schedule":
        agent_card = _http_request(
            "GET",
            f"{normalized_service_url}/.well-known/agent-card.json",
            headers=headers,
        )
        advertised_url = str(agent_card.get("url", "")).rstrip("/")
        if advertised_url != normalized_service_url:
            raise ValueError(
                "Agent card URL does not match deployed service URL: "
                f"{advertised_url!r} != {normalized_service_url!r}"
            )
        task_id = _send_agent3_message(normalized_service_url, headers=headers)
        _poll_agent3_task(normalized_service_url, task_id, headers=headers)
        return
    raise ValueError(f"unsupported smoke check type: {smoke_type}")


def _send_agent3_message(
    service_url: str,
    *,
    headers: dict[str, str] | None = None,
) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": "smoke-schedule-1",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "smoke-message-1",
                "role": "user",
                "kind": "message",
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "day_start": "2026-05-10T09:00:00",
                            "day_end": "2026-05-10T11:00:00",
                            "food_budget_per_day": 0,
                            "preferences": [],
                            "acceptable_transport_modes": ["walking"],
                            "places": [
                                {
                                    "id": "pantheon",
                                    "name": "Pantheon",
                                    "location": {
                                        "latitude": 41.8986,
                                        "longitude": 12.4769,
                                    },
                                    "estimated_visit_duration_minutes": 45,
                                    "priority_score": 4,
                                }
                            ],
                        },
                    }
                ],
            },
            "configuration": {"acceptedOutputModes": ["application/json"]},
        },
    }
    response = _http_request("POST", service_url.rstrip("/") + "/", payload, headers=headers)
    task_id = response.get("result", {}).get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Agent 3 smoke request did not return a task id")
    return task_id


def _poll_agent3_task(
    service_url: str,
    task_id: str,
    *,
    headers: dict[str, str] | None = None,
) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        payload = {
            "jsonrpc": "2.0",
            "id": f"{task_id}-poll",
            "method": "tasks/get",
            "params": {"id": task_id},
        }
        response = _http_request(
            "POST",
            service_url.rstrip("/") + "/",
            payload,
            headers=headers,
        )
        state = response.get("result", {}).get("status", {}).get("state")
        if state == "completed":
            return
        if state in {"failed", "canceled"}:
            raise ValueError(f"Agent 3 smoke task ended in state {state}")
        time.sleep(1)
    raise TimeoutError("Agent 3 smoke task did not complete within 30 seconds")


def _http_request(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, method=method, data=data, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {exc.code}: {body}") from exc
    return json.loads(body) if body else {}


def _cloud_run_service_name(manifest: ServiceManifest) -> str:
    return str(manifest.deploy["cloud_run_service"])


def _cloud_run_auth_headers(service_url: str) -> dict[str, str]:
    token = _print_identity_token(service_url.rstrip("/"))
    return {"Authorization": f"Bearer {token}"}


def _print_identity_token(audience: str) -> str:
    command = [
        "gcloud",
        "auth",
        "print-identity-token",
        f"--audiences={audience}",
    ]
    impersonated_service_account = os.environ.get("GCP_SERVICE_ACCOUNT_EMAIL", "").strip()
    if impersonated_service_account:
        command.append(
            f"--impersonate-service-account={impersonated_service_account}"
        )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"Failed to mint Cloud Run identity token: {stderr}")
    token = completed.stdout.strip()
    if not token:
        raise ValueError("gcloud auth print-identity-token returned an empty token")
    return token


def build_submit_command(
    manifest: ServiceManifest,
    *,
    image_uri: str,
) -> list[str]:
    command = [
        "gcloud",
        "builds",
        "submit",
        str((REPO_ROOT / manifest.build_context).resolve()),
        "--tag",
        image_uri,
        "--async",
    ]
    dockerfile_path = (REPO_ROOT / manifest.dockerfile).resolve()
    expected_dockerfile = (REPO_ROOT / manifest.build_context / "Dockerfile").resolve()
    if dockerfile_path != expected_dockerfile:
        command.extend(["--file", str(dockerfile_path)])
    return command


def extract_build_id(output: str) -> str:
    match = re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        output,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Could not extract Cloud Build id from output: {output!r}")
    return match.group(0)


def wait_for_build(build_id: str, *, timeout_seconds: int = 1800) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        completed = subprocess.run(
            [
                "gcloud",
                "builds",
                "describe",
                build_id,
                "--format=json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        status = str(payload.get("status", "")).upper()
        if status == "SUCCESS":
            return
        if status in {"FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED", "EXPIRED"}:
            log_url = payload.get("logUrl", "")
            raise RuntimeError(
                f"Cloud Build {build_id} finished with status {status}. "
                f"Inspect logs at {log_url or 'Cloud Build history'}."
            )
        time.sleep(5)
    raise TimeoutError(f"Cloud Build {build_id} did not finish within {timeout_seconds} seconds")
