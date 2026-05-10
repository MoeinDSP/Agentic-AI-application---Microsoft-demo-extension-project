from __future__ import annotations

import os
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

from tools.cicd.gcp_cloud_run import (
    build_image_uri,
    build_run_deploy_command,
    build_submit_command,
    extract_build_id,
    _print_identity_token,
    resolve_runtime_service_account,
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

    def test_build_submit_command_uses_async_mode(self) -> None:
        manifest = load_manifests()["agent-3-mcp"]

        command = build_submit_command(
            manifest,
            image_uri="europe-west1-docker.pkg.dev/project/repo/agent-3-mcp:tag",
        )

        self.assertIn("--async", command)

    def test_extract_build_id_parses_gcloud_submit_output(self) -> None:
        output = (
            "Created [https://cloudbuild.googleapis.com/v1/projects/project/builds/"
            "e4aa618e-9e2c-4970-80d5-e7906d560ddb].\n"
        )

        build_id = extract_build_id(output)

        self.assertEqual(build_id, "e4aa618e-9e2c-4970-80d5-e7906d560ddb")

    def test_resolve_env_vars_uses_dependency_url_and_github_var(self) -> None:
        manifests = load_manifests()

        env_vars = resolve_env_vars(
            manifests["agent-3"],
            deployed_urls={"agent-3-mcp": "https://agent-3-mcp.example.com"},
            region="europe-west1",
            allow_missing_service_url=True,
        )

        self.assertEqual(env_vars["AGENT3_MCP_BASE_URL"], "https://agent-3-mcp.example.com")
        self.assertNotIn("AGENT3_AGENT4_BASE_URL", env_vars)

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
            runtime_service_account="agent-3-mcp-runtime@example.iam.gserviceaccount.com",
        )

        self.assertIn("--set-env-vars", command)
        self.assertIn("--set-secrets", command)
        self.assertIn("--service-account", command)
        self.assertIn("--no-allow-unauthenticated", command)
        self.assertTrue(
            any(
                "AGENT3_MCP_GOOGLE_MAPS_API_KEY=agent3-mcp-google-maps-api-key:latest"
                in item
                for item in command
            )
        )

    def test_resolve_env_vars_fetches_dependency_url_when_not_deployed_in_run(self) -> None:
        manifests = load_manifests()

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

    def test_resolve_env_vars_does_not_require_agent4_url_for_deploy(self) -> None:
        manifests = load_manifests()

        env_vars = resolve_env_vars(
            manifests["agent-3"],
            deployed_urls={"agent-3-mcp": "https://agent-3-mcp.example.com"},
            region="europe-west1",
            allow_missing_service_url=True,
        )

        self.assertNotIn("AGENT3_AGENT4_BASE_URL", env_vars)

    def test_resolve_runtime_service_account_reads_github_var(self) -> None:
        manifests = load_manifests()
        os.environ["GCP_AGENT3_RUNTIME_SERVICE_ACCOUNT_EMAIL"] = (
            "agent-3-runtime@example.iam.gserviceaccount.com"
        )
        self.addCleanup(os.environ.pop, "GCP_AGENT3_RUNTIME_SERVICE_ACCOUNT_EMAIL", None)

        service_account = resolve_runtime_service_account(manifests["agent-3"])

        self.assertEqual(
            service_account,
            "agent-3-runtime@example.iam.gserviceaccount.com",
        )

    def test_smoke_check_service_rejects_agent_card_url_mismatch(self) -> None:
        manifest = load_manifests()["agent-3"]
        with patch(
            "tools.cicd.gcp_cloud_run._print_identity_token",
            return_value="mock-token",
        ), patch(
            "tools.cicd.gcp_cloud_run._http_request",
            return_value={"url": "https://wrong.example.com"},
        ):
            with self.assertRaisesRegex(ValueError, "Agent card URL does not match"):
                smoke_check_service(
                    manifest,
                    service_url="https://agent-3.example.com",
                )

    def test_smoke_check_service_adds_auth_header_for_private_cloud_run(self) -> None:
        manifest = load_manifests()["agent-3-mcp"]
        calls: list[tuple[str, str, dict[str, object] | None, dict[str, str] | None]] = []

        def _capture_request(
            method: str,
            url: str,
            payload: dict[str, object] | None = None,
            *,
            headers: dict[str, str] | None = None,
        ) -> dict[str, object]:
            calls.append((method, url, payload, headers))
            return {}

        with patch(
            "tools.cicd.gcp_cloud_run._print_identity_token",
            return_value="mock-token",
        ), patch(
            "tools.cicd.gcp_cloud_run._http_request",
            side_effect=_capture_request,
        ):
            smoke_check_service(
                manifest,
                service_url="https://agent-3-mcp.example.com",
            )

        assert calls[0][3] == {"Authorization": "Bearer mock-token"}
        assert calls[1][3] == {"Authorization": "Bearer mock-token"}
        assert calls[1][2] == {
            "origin": {"lat": 41.8902, "lng": 12.4922},
            "destination": {"lat": 41.8986, "lng": 12.4769},
            "mode": "walking",
        }

    def test_print_identity_token_uses_impersonation_when_configured(self) -> None:
        os.environ["GCP_SERVICE_ACCOUNT_EMAIL"] = "github-deployer@example.iam.gserviceaccount.com"
        self.addCleanup(os.environ.pop, "GCP_SERVICE_ACCOUNT_EMAIL", None)

        with patch(
            "tools.cicd.gcp_cloud_run.subprocess.run",
            return_value=CompletedProcess(args=[], returncode=0, stdout="token\n", stderr=""),
        ) as mock_run:
            token = _print_identity_token("https://agent-3-mcp.example.com")

        self.assertEqual(token, "token")
        command = mock_run.call_args.args[0]
        self.assertIn(
            "--impersonate-service-account=github-deployer@example.iam.gserviceaccount.com",
            command,
        )

    def test_print_identity_token_surfaces_gcloud_error_output(self) -> None:
        with patch(
            "tools.cicd.gcp_cloud_run.subprocess.run",
            return_value=CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr="permission denied",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "permission denied"):
                _print_identity_token("https://agent-3-mcp.example.com")
