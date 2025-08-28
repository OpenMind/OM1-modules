import asyncio
import logging
import threading
from queue import Empty, Queue
from typing import Callable, Optional, Union

# Prefer the synchronous API if available (websockets>=12)
try:
    from websockets.sync.client import connect as ws_sync_connect  # type: ignore
    HAS_SYNC = True
except Exception:
    HAS_SYNC = False
    import websockets  # async API

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class Client:
    """
    WebSocket client that supports both sync (websockets>=12) and async fallbacks.
    Public API is thread-friendly:
      - start()
      - send_message(str|bytes)
      - register_message_callback(callable)
      - is_connected()
      - stop()
    """

    def __init__(self, url: str = "ws://localhost:6789"):
        self.url = url
        self.running: bool = True
        self.is_policy_violation: bool = False
        self.connected: bool = False

        # Will hold either a sync client (v12) or an async WebSocketClientProtocol
        self.websocket = None

        self.message_callback: Optional[Callable] = None
        self.message_queue: Queue = Queue()

        # Threads
        self.client_thread: Optional[threading.Thread] = None
        self.sender_thread: Optional[threading.Thread] = None
        self.receiver_thread: Optional[threading.Thread] = None

        # Async-only members
        self._async_mode = not HAS_SYNC
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        # Reconnect guard
        self._connect_lock = threading.Lock()

    # -------------------------
    # Lifecycle
    # -------------------------
    def start(self):
        """Start the connection manager."""
        self.client_thread = threading.Thread(target=self._run_client, daemon=True)
        self.client_thread.start()
        logger.info("WebSocket client thread started")

    def _run_client(self):
        """Keep the connection alive; reconnect if dropped."""
        while self.running:
            if not self.connected and not self.is_policy_violation:
                if self.connect():
                    logger.info("Connection established")
                else:
                    logger.info("Connection failed, retrying in 5 seconds")
                    threading.Event().wait(5)
            else:
                threading.Event().wait(1)

    def connect(self) -> bool:
        """Establish the connection (sync or async)."""
        with self._connect_lock:
            try:
                if self._async_mode:
                    # Start loop thread if needed
                    if self._loop is None:
                        self._loop = asyncio.new_event_loop()
                        self._loop_thread = threading.Thread(
                            target=self._loop.run_forever, daemon=True
                        )
                        self._loop_thread.start()

                    fut = asyncio.run_coroutine_threadsafe(self._async_connect(), self._loop)
                    fut.result(timeout=15)  # raise if connection fails
                    self.connected = True

                    # Sender runs in a normal thread; receiver runs inside the loop
                    if not self.sender_thread or not self.sender_thread.is_alive():
                        self.sender_thread = threading.Thread(
                            target=self._send_messages, daemon=True
                        )
                        self.sender_thread.start()

                else:
                    # Sync API: blocking connect returns a protocol with .send/.recv/.close
                    self.websocket = ws_sync_connect(self.url)
                    self.connected = True

                    # Start sender/receiver threads
                    if not self.receiver_thread or not self.receiver_thread.is_alive():
                        self.receiver_thread = threading.Thread(
                            target=self._receive_messages_sync, daemon=True
                        )
                        self.receiver_thread.start()

                    if not self.sender_thread or not self.sender_thread.is_alive():
                        self.sender_thread = threading.Thread(
                            target=self._send_messages, daemon=True
                        )
                        self.sender_thread.start()

                logger.info(f"Connected to {self.url}")
                return True
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.connected = False
                return False

    # -------------------------
    # Send / Receive
    # -------------------------
    def _send_messages(self):
        """Drain the queue and send out messages (sync or schedule async)."""
        while self.running:
            try:
                message = self.message_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                if not self.connected or self.websocket is None:
                    # Put back and let the manager reconnect
                    self.message_queue.put(message)
                    continue

                if self._async_mode:
                    # schedule coroutine send on the loop
                    async def _send(ws, data):
                        await ws.send(data)

                    fut = asyncio.run_coroutine_threadsafe(
                        _send(self.websocket, message), self._loop
                    )
                    fut.result(timeout=10)
                else:
                    # sync send
                    self.websocket.send(message)

                logger.debug(f"Sent WS Message to {self.url}")
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                # Put back to retry later
                self.message_queue.put(message)
                self.connected = False

    # ----- Sync receive loop (websockets.sync) -----
    def _receive_messages_sync(self):
        while self.running and self.connected and self.websocket is not None:
            try:
                message = self.websocket.recv()  # blocking
                if self.message_callback:
                    self.message_callback(message)
            except Exception as e:
                # websockets.sync raises ConnectionClosed on normal close too
                logger.info(f"WebSocket sync receive ended: {e}")
                self.connected = False
                break

    # ----- Async connect / receive (fallback) -----
    async def _async_connect(self):
        import websockets  # async
        self.websocket = await websockets.connect(self.url)
        # Start async receive task inside the loop
        asyncio.create_task(self._async_recv_loop())

    async def _async_recv_loop(self):
        import websockets  # async
        try:
            async for message in self.websocket:
                if self.message_callback:
                    try:
                        self.message_callback(message)
                    except Exception as cb_err:
                        logger.error(f"Message callback error: {cb_err}")
        except websockets.ConnectionClosed as e:
            if getattr(e, "code", None) == 1008:
                self.is_policy_violation = True
                logger.error("\n\n----- Policy Violation -----")
                logger.error(f"Policy violation: {getattr(e, 'reason', '')}")
                logger.error("----- Policy Violation -----\n\n")
            logger.info("WebSocket connection closed (async)")
            self.connected = False
        except Exception as e:
            logger.error(f"Async receive error: {e}")
            self.connected = False

    # -------------------------
    # Public API
    # -------------------------
    def send_message(self, message: Union[str, bytes]):
        """Queue a message for sending."""
        if self.running:
            self.message_queue.put(message)

    def register_message_callback(self, callback: Callable):
        self.message_callback = callback
        logger.info("Registered message callback")

    def is_connected(self) -> bool:
        return self.connected

    def stop(self):
        """Stop client and close connections cleanly."""
        self.running = False

        # drain queue
        try:
            while True:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
        except Empty:
            pass

        # Close websocket
        try:
            if self.websocket is not None:
                if self._async_mode:
                    fut = asyncio.run_coroutine_threadsafe(self.websocket.close(), self._loop)
                    try:
                        fut.result(timeout=5)
                    except Exception:
                        pass
                else:
                    self.websocket.close()
        except Exception:
            pass

        # Stop loop if async
        if self._async_mode and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

        self.connected = False
        logger.info("WebSocket client stopped")
