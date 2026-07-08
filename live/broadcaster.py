"""WebSocket stats broadcaster.

Runs a small asyncio ``websockets`` server on its own thread with its own event
loop. The synchronous CV processing loop (a different thread) hands finished
per-frame stat dicts to :meth:`broadcast`, which marshals them onto the server's
loop via ``call_soon_threadsafe`` so inference never blocks on socket I/O.
Clients connect and receive one JSON message per processed frame.
"""

import asyncio
import json
import threading


class StatsBroadcaster:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port

        self._clients = set()
        self._loop = None
        self._thread = None
        self._server = None
        self._ready = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait until the loop + server are actually up before returning.
        self._ready.wait(timeout=10.0)
        return self

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._ready.set()
        self._loop.run_forever()

    async def _start_server(self):
        import websockets
        self._server = await websockets.serve(self._handler, self.host, self.port)

    async def _handler(self, websocket, *args):
        # websockets>=11 calls the handler with just (websocket); older
        # versions pass (websocket, path). Accept both via *args.
        self._clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)

    def broadcast(self, stats):
        """Thread-safe: called from the sync CV loop. Serialises ``stats`` and
        schedules the send on the server's event loop."""
        if self._loop is None:
            return
        message = json.dumps(stats)
        self._loop.call_soon_threadsafe(self._send, message)

    def _send(self, message):
        # Runs on the event loop thread.
        import websockets
        if self._clients:
            websockets.broadcast(self._clients, message)

    def stop(self):
        if self._loop is None:
            return

        async def _shutdown():
            if self._server is not None:
                self._server.close()
                # websockets' asyncio server exposes an awaitable wait_closed().
                wait_closed = getattr(self._server, "wait_closed", None)
                if wait_closed is not None:
                    try:
                        await wait_closed()
                    except Exception:
                        pass
            self._loop.stop()

        self._loop.call_soon_threadsafe(
            lambda: self._loop.create_task(_shutdown()))
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
