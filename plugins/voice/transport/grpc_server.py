"""gRPC Transport Server for Voice Pipeline (placeholder)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GrpcTransportServer:
    """gRPC server for bidirectional voice communication (not yet implemented)."""

    def __init__(
        self,
        config: dict[str, Any],
        stt_engine: Any = None,
        tts_engine: Any = None,
        wakeword_engine: Any = None,
    ):
        self.config = config
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.wakeword_engine = wakeword_engine
        self._running = False

    async def start(self) -> None:
        """Start the gRPC server."""
        if self._running:
            return

        logger.warning("gRPC transport not yet implemented")
        self._running = True

    async def stop(self) -> None:
        """Stop the gRPC server."""
        self._running = False
        logger.info("gRPC transport server stopped")