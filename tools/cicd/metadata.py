from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_INFRA_PREFIXES = (
    ".github/scripts/",
    ".github/workflows/",
    "tests/cicd/",
    "tools/cicd/",
)


@dataclass(frozen=True)
class ServiceManifest:
    service_name: str
    manifest_path: Path
    ownership: str
    backend: str
    deploy_enabled: bool
    validate_enabled: bool
    build_context: str
    dockerfile: str
    runtime_port: int
    healthcheck: str
    smoke_check_type: str
    depends_on: tuple[str, ...]
    ci: dict[str, Any]
    deploy: dict[str, Any]

    @property
    def service_root(self) -> str:
        return self.manifest_path.parent.relative_to(REPO_ROOT).as_posix()

    def to_ci_matrix_entry(self) -> dict[str, str]:
        return {
            "service_name": self.service_name,
            "working_directory": self.ci["working_directory"],
            "install_command": self.ci["install_command"],
            "lint_command": self.ci["lint_command"],
            "test_command": self.ci["test_command"],
        }

    def to_live_smoke_entry(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "working_directory": self.ci["working_directory"],
            "install_command": self.ci["install_command"],
            "command": self.ci["live_smoke_command"],
            "env": self.ci.get("live_smoke_env", {}),
        }


def load_manifests(repo_root: Path | None = None) -> dict[str, ServiceManifest]:
    root = repo_root or REPO_ROOT
    manifests: dict[str, ServiceManifest] = {}
    for manifest_path in sorted(root.glob("services/*/service.deploy.yaml")):
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest = ServiceManifest(
            service_name=data["service_name"],
            manifest_path=manifest_path,
            ownership=data["ownership"],
            backend=data["backend"],
            deploy_enabled=bool(data["deploy_enabled"]),
            validate_enabled=bool(data["validate_enabled"]),
            build_context=data["build_context"],
            dockerfile=data["dockerfile"],
            runtime_port=int(data["runtime_port"]),
            healthcheck=data["healthcheck"],
            smoke_check_type=data["smoke_check_type"],
            depends_on=tuple(data.get("depends_on", [])),
            ci=data["ci"],
            deploy=data["deploy"],
        )
        manifests[manifest.service_name] = manifest
    return manifests


def select_ci_services(
    manifests: dict[str, ServiceManifest],
    changed_files: list[str],
) -> list[ServiceManifest]:
    if _includes_ci_infra_change(changed_files):
        return [
            manifest
            for manifest in sorted(manifests.values(), key=lambda item: item.service_name)
            if manifest.validate_enabled and manifest.ci.get("enabled", False)
        ]
    return [
        manifests[name]
        for name in _changed_manifest_names(manifests, changed_files)
        if manifests[name].validate_enabled and manifests[name].ci.get("enabled", False)
    ]


def select_live_smoke_services(
    manifests: dict[str, ServiceManifest],
) -> list[ServiceManifest]:
    return [
        manifest
        for manifest in sorted(manifests.values(), key=lambda item: item.service_name)
        if manifest.ci.get("enabled", False) and manifest.ci.get("live_smoke_command")
    ]


def select_deploy_services(
    manifests: dict[str, ServiceManifest],
    changed_files: list[str],
    *,
    backend: str,
) -> list[ServiceManifest]:
    if _includes_ci_infra_change(changed_files):
        selected = [
            manifest
            for manifest in sorted(manifests.values(), key=lambda item: item.service_name)
            if manifest.ownership == "repo-deployed"
            and manifest.deploy_enabled
            and manifest.backend == backend
        ]
        return order_manifests(manifests, selected)

    changed_names = _changed_manifest_names(manifests, changed_files)
    selected = [
        manifests[name]
        for name in changed_names
        if manifests[name].ownership == "repo-deployed"
        and manifests[name].deploy_enabled
        and manifests[name].backend == backend
    ]
    return order_manifests(manifests, selected)


def order_manifests(
    manifests: dict[str, ServiceManifest],
    selected: list[ServiceManifest],
) -> list[ServiceManifest]:
    selected_names = {manifest.service_name for manifest in selected}
    ordered: list[ServiceManifest] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(service_name: str) -> None:
        if service_name in visited:
            return
        if service_name in visiting:
            raise ValueError(f"cyclic dependency detected at {service_name}")
        visiting.add(service_name)
        for dependency in manifests[service_name].depends_on:
            if dependency in selected_names:
                visit(dependency)
        visiting.remove(service_name)
        visited.add(service_name)
        ordered.append(manifests[service_name])

    for manifest in sorted(selected, key=lambda item: item.service_name):
        visit(manifest.service_name)
    return ordered


def json_dumps(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"))


def _changed_manifest_names(
    manifests: dict[str, ServiceManifest],
    changed_files: list[str],
) -> list[str]:
    changed_names = set()
    for manifest in manifests.values():
        prefix = f"{manifest.service_root}/"
        if any(path.startswith(prefix) for path in changed_files):
            changed_names.add(manifest.service_name)
    return sorted(changed_names)


def _includes_ci_infra_change(changed_files: list[str]) -> bool:
    return any(
        any(path.startswith(prefix) for prefix in CI_INFRA_PREFIXES)
        for path in changed_files
    )
