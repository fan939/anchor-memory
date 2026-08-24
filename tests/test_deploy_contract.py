from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DeployContractTests(unittest.TestCase):
    def test_compose_reuses_named_external_production_volumes(self):
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

        self.assertIn("container_name: anchor-memory", compose)
        self.assertIn("127.0.0.1:${ANCHOR_LOCAL_PORT:-8100}:8000", compose)
        self.assertIn("external: true", compose)
        self.assertIn("${ANCHOR_DATA_VOLUME:-anchor-memory-data}", compose)
        self.assertIn(
            "${ANCHOR_MODEL_CACHE_VOLUME:-anchor-memory-model-cache}", compose
        )
        self.assertNotIn("caddy:", compose)
        self.assertNotIn('"80:80"', compose)
        self.assertNotIn('"443:443"', compose)

    def test_production_deploy_requires_backup_and_has_rollback_trap(self):
        script = (ROOT / "scripts" / "deploy_production.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('"$data_volume:/data/anchor:ro"', script)
        self.assertIn('"$backup_dir:/backups"', script)
        self.assertIn("trap 'restore_previous' EXIT INT TERM HUP", script)
        self.assertIn('docker rename "$container_name" "$rollback_name"', script)
        self.assertIn('docker rename "$rollback_name" "$container_name"', script)
        self.assertNotIn("docker-compose", script)


if __name__ == "__main__":
    unittest.main()
