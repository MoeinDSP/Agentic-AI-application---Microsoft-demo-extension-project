from __future__ import annotations

import unittest

from tools.cicd.metadata import load_manifests, select_ci_services, select_deploy_services


class MetadataTests(unittest.TestCase):
    def test_load_manifests_includes_repo_deployed_services(self) -> None:
        manifests = load_manifests()

        self.assertIn("agent-3", manifests)
        self.assertIn("agent-3-mcp", manifests)
        self.assertEqual(manifests["agent-3"].ownership, "repo-deployed")
        self.assertEqual(manifests["agent-3-mcp"].backend, "gcp-cloud-run")

    def test_select_ci_services_for_service_change(self) -> None:
        manifests = load_manifests()

        selected = select_ci_services(
            manifests,
            ["services/agent-3/src/agent3/services/planner.py"],
        )

        self.assertEqual([manifest.service_name for manifest in selected], ["agent-3"])

    def test_select_ci_services_for_ci_infra_change(self) -> None:
        manifests = load_manifests()

        selected = select_ci_services(
            manifests,
            [".github/workflows/ci.yml"],
        )

        selected_names = {manifest.service_name for manifest in selected}

        self.assertIn("agent-3", selected_names)
        self.assertIn("agent-3-mcp", selected_names)

    def test_select_deploy_services_orders_dependencies(self) -> None:
        manifests = load_manifests()

        ordered = select_deploy_services(
            manifests,
            [
                "services/agent-3/src/agent3/services/planner.py",
                "services/agent-3-mcp/src/agent3_mcp/services/tools.py",
            ],
            backend="gcp-cloud-run",
        )

        self.assertEqual(
            [manifest.service_name for manifest in ordered],
            ["agent-3-mcp", "agent-3"],
        )
