"""Settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES = ROOT / "fixtures" / "replay.json"


@dataclass(frozen=True)
class Settings:
    db_path: Path
    host: str
    port: int
    coalesce_s: float
    model: str
    replay_forced: bool
    replay_scale: float
    fixtures: Path
    record_path: Path | None
    lease_s: float
    max_concurrent_jobs: int

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        record = env.get("DTA_RECORD")
        return cls(
            db_path=Path(env.get("DTA_DB", str(ROOT / "data" / "demo.db"))),
            host="127.0.0.1",
            port=int(env.get("PORT", "4106")),
            coalesce_s=float(env.get("DTA_COALESCE_S", "3")),
            model=env.get("DTA_MODEL", "claude-sonnet-5-5"),
            replay_forced=env.get("DEMO_REPLAY", "0") == "1",
            replay_scale=float(env.get("DTA_REPLAY_SCALE", "1")),
            fixtures=Path(env.get("DTA_FIXTURES", str(DEFAULT_FIXTURES))),
            record_path=Path(record) if record else None,
            lease_s=float(env.get("DTA_LEASE_S", "20")),
            max_concurrent_jobs=int(env.get("DTA_MAX_JOBS", "4")),
        )

    def worker_env(self, mode: str) -> dict[str, str]:
        """App settings handed to the worker; it reads them, then scrubs its env."""
        env = {
            "DTA_DB": str(self.db_path),
            "DTA_MODE": mode,
            "DTA_MODEL": self.model,
            "DTA_REPLAY_SCALE": str(self.replay_scale),
            "DTA_FIXTURES": str(self.fixtures),
            "DTA_LEASE_S": str(self.lease_s),
            "DTA_MAX_JOBS": str(self.max_concurrent_jobs),
            "DTA_COALESCE_S": str(self.coalesce_s),
        }
        if self.record_path:
            env["DTA_RECORD"] = str(self.record_path)
        return env
