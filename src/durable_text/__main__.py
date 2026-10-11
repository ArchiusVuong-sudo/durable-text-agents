"""Run the demo: ``python -m durable_text`` (127.0.0.1, PORT or 4106)."""

from __future__ import annotations

import logging
import os

import uvicorn

from .config import Settings
from .web import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    uds = os.environ.get("DTA_UDS")  # tests serve on a Unix socket, not a TCP port
    config = uvicorn.Config(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        uds=uds,
        log_level="warning",
        access_log=False,
    )
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
