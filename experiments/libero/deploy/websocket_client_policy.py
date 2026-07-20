"""WebSocket policy client. Sync counterpart to websocket_policy_server.py.

Speaks the same msgpack-over-websocket protocol as WebsocketPolicyServer:
on connect the server sends packed `metadata` once, then the client sends a
packed obs dict per `infer()` call and receives a packed action dict back.
"""

from __future__ import annotations

import logging
import time

import websockets.sync.client

from .msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket."""

    def __init__(self, host: str = "0.0.0.0", port: int | None = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self):
        logger.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._ws.close()
