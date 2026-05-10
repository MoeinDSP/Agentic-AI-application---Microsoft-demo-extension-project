from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tools.cicd.gcp_cloud_run import (
    build_image_uri,
    build_run_deploy_command,
    resolve_env_vars,
    resolve_secret_bindings,
    smoke_check_service,
)
from tools.cicd.metadata import load_manifests


class GcpCloudRunTests(unittest.TestCase):
    def test_build_image_uri_uses_expected_registry_shape(self) -> None:
        self.assertEqual(
            build_image_uri(
                project_id="cloud-computing-course-495606",
                region="europe-west1",
                repository="agent-services",
                service_name="agent-3",
                tag="abc123",
            ),
            "europe-west1-docker.pkg.dev/cloud-computing-course-495606/agent-services/agent-3:abc123",
        )

    def test_resolve_env_vars_uses_dependency_url_and_github_var(self) -> None:
        manifests = load_manifests()
        os.environ["AGENT3_AGENT4_BASE_URL"] = "https://external-agent4.example.com"
        self.addCleanup(os.environ.pop, "AGENT3_AGENT4_BASE_URL", None)

        env_vars = resolve_env_vars(
            manifests["agent-3"],
            deployed_urls={"agent-3-mcp": "https://agent-3-mcp.example.com"},
            region="europe-west1",
            allow_missing_service_url=True,
        )

        self.assertEqual(env_vars["AGENT3_MCP_BASE_URL"], "https://agent-3-mcp.example.com")
        self.assertEqual(
            env_vars["AGENT3_AGENT4_BASE_URL"],
            "https://external-agent4.example.com",
        )

    def test_build_run_deploy_command_includes_env_and_secret_bindings(self) -> None:
        manifest = load_manifests()["agent-3-mcp"]
        command = build_run_deploy_command(
            manifest,
            image_uri="europe-west1-docker.pkg.dev/project/repo/agent-3-mcp:tag",
            region="europe-west1",
            env_vars={
                "AGENT3_MCP_ENVIRONMENT": "production",
                "AGENT3_MCP_GOOGLE_ROUTES_TIMEOUT_SECONDS": "5.0",
            },
            secret_bindings=resolve_secret_bindings(manifest),
        )

        self.assertIn("--set-env-vars", command)
        self.assertIn("--set-secrets", command)
        self.assertTrue(
            any(
                "AGENT3_MCP_GOOGLE_MAPS_API_KEY=agent3-mcp-google-maps-api-key:latest"
                in item
                for item in command
            )
        )

    def test_resolve_env_vars_fetches_dependency_url_when_not_deployed_in_run(self) -> None:
        manifests = load_manifests()
        os.environ["AGENT3_AGENT4_BASE_URL"] = "https://external-agent4.example.com"
        self.addCleanup(os.environ.pop, "AGENT3_AGENT4_BASE_URL", None)

        with patch(
            "tools.cicd.gcp_cloud_run.fetch_service_url",
            return_value="https://existing-agent-3-mcp.example.com",
        ):
            env_vars = resolve_env_vars(
                manifests["agent-3"],
                deployed_urls={},
                region="europe-west1",
                allow_missing_service_url=True,
            )

        self.assertEqual(
            env_vars["AGENT3_MCP_BASE_URL"],
            "https://existing-agent-3-mcp.example.com",
        )

    def test_resolve_env_vars_includes_service_url_for_self_advertisement(self) -> None:
        manifests = load_manifests()
        os.environ["AGENT3_AGENT4_BASE_URL"] = "https://external-agent4.example.com"
        self.addCleanup(os.environ.pop, "AGENT3_AGENT4_BASE_URL", None)

        env_vars = resolve_env_vars(
            manifests["agent-3"],
            deployed_urls={"agent-3-mcp": "https://agent-3-mcp.example.com"},
            region="europe-west1",
            service_url="https://agent-3.example.com",
        )

        self.assertEqual(
            env_vars["AGENT3_PUBLIC_BASE_URL"],
            "https://agent-3.example.com",
        )

    def test_smoke_check_service_rejects_agent_card_url_mismatch(self) -> None:
        manifest = load_manifests()["agent-3"]
        with patch(
            "tools.cicd.gcp_cloud_run._http_request",
            return_value={"url": "https://wrong.example.com"},
        ):
            with self.assertRaisesRegex(ValueError, "Agent card URL does not match"):
                smoke_check_service(
                    manifest,
                    service_url="https://agent-3.example.com",
                )
