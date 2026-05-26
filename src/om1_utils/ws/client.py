import logging
import multiprocessing as mp
import threading
import time
from queue import Empty
from typing import Callable, Optional, Union

import websockets
from websockets.sync.client import ClientConnection, connect

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


def _ws_worker_process(
    url: str,
    outbound_queue: mp.Queue,
    inbound_queue: mp.Queue,
    state_queue: mp.Queue,
    control_queue: mp.Queue,
):
    websocket: Optional[ClientConnection] = None
    connected = False
    is_policy_violation = False
    running = True

    while running:
        try:
            while True:
                command = control_queue.get_nowait()
                if command == "stop":
                    running = False
        except Empty:
            pass

        if not running:
            break

        if not connected and not is_policy_violation:
            try:
                websocket = connect(url)
                connected = True
                state_queue.put(("connected", True))
            except Exception as e:
                state_queue.put(("connection_error", str(e)))
                time.sleep(5)
                continue

        if not connected or websocket is None:
            time.sleep(0.1)
            continue

        outbound_message: Optional[str | bytes] = None
        try:
            outbound_message = outbound_queue.get_nowait()
        except Empty:
            pass

        if outbound_message is not None:
            try:
                websocket.send(outbound_message)
            except Exception as e:
                state_queue.put(("send_error", str(e)))
                outbound_queue.put(outbound_message)
                try:
                    websocket.close()
                except Exception:
                    pass
                connected = False
                websocket = None
                state_queue.put(("connected", False))
                continue

        try:
            message = websocket.recv(timeout=0.1)
            inbound_queue.put(("message", message))
        except TimeoutError:
            pass
        except websockets.ConnectionClosed as e:
            if e.code == 1008:
                is_policy_violation = True
                state_queue.put(("policy_violation", e.reason))
            connected = False
            state_queue.put(("connected", False))
            try:
                websocket.close()
            except Exception:
                pass
            websocket = None
        except Exception as e:
            state_queue.put(("receive_error", str(e)))
            connected = False
            state_queue.put(("connected", False))
            try:
                websocket.close()
            except Exception:
                pass
            websocket = None

    if websocket is not None:
        try:
            websocket.close()
        except Exception:
            pass
    state_queue.put(("connected", False))
    state_queue.put(("stopped", True))


class Client:
    """
    A WebSocket client implementation with support for asynchronous message handling.

    This class uses a worker process to manage the WebSocket connection, message
    sending, and message receiving. The main process only dispatches callbacks and
    tracks connection state.

    Parameters
    ----------
    url : str, optional
        The WebSocket server URL to connect to, by default "ws://localhost:6789"
    """

    def __init__(self, url: str = "ws://localhost:6789"):
        self.url = url
        self.running: bool = True
        self.is_policy_violation: bool = False
        self.connected: bool = False
        self.websocket: Optional[ClientConnection] = None
        self.message_callback: Optional[Callable] = None

        self.message_queue: mp.Queue = mp.Queue()
        self._incoming_queue: mp.Queue = mp.Queue()
        self._state_queue: mp.Queue = mp.Queue()
        self._control_queue: mp.Queue = mp.Queue()

        self._worker_process: Optional[mp.Process] = None
        self._event_thread: Optional[threading.Thread] = None

        self.receiver_thread: Optional[threading.Thread] = None
        self.sender_thread: Optional[threading.Thread] = None
        self.client_thread: Optional[threading.Thread] = None

    def _drain_state_queue(self):
        while True:
            try:
                state, payload = self._state_queue.get_nowait()
            except Empty:
                break

            if state == "connected":
                self.connected = bool(payload)
                if self.connected:
                    logger.info(f"Connected to {self.url}")
                else:
                    logger.info("WebSocket connection closed")
            elif state == "policy_violation":
                self.is_policy_violation = True
                logger.error("\n\n")
                logger.error("----- Policy Violation -----")
                logger.error(f"Policy violation: {payload}")
                logger.error("----- Policy Violation -----\n\n")
            elif state == "connection_error":
                logger.error(f"Connection error: {payload}")
            elif state == "send_error":
                logger.error(f"Failed to send message: {payload}")
            elif state == "receive_error":
                logger.error(f"Error in message processing: {payload}")

    def _drain_incoming_queue(self):
        while True:
            try:
                kind, payload = self._incoming_queue.get_nowait()
            except Empty:
                break

            if kind != "message":
                continue

            formatted_msg = self.format_message(payload)
            logger.debug(f"Received WS Message: {formatted_msg}")
            if self.message_callback:
                self.message_callback(payload)

    def _monitor_worker(self):
        while self.running:
            self._drain_state_queue()
            self._drain_incoming_queue()

            if (
                self._worker_process
                and not self._worker_process.is_alive()
                and self.running
                and not self.is_policy_violation
            ):
                logger.info("Worker process stopped, attempting reconnect")
                self._start_worker_process()

            time.sleep(0.05)

    def _start_worker_process(self):
        self._worker_process = mp.Process(
            target=_ws_worker_process,
            args=(
                self.url,
                self.message_queue,
                self._incoming_queue,
                self._state_queue,
                self._control_queue,
            ),
            daemon=True,
        )
        self._worker_process.start()

    def _start_event_thread(self):
        if self._event_thread and self._event_thread.is_alive():
            return
        self._event_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self._event_thread.start()
        self.client_thread = self._event_thread

    def connect(self) -> bool:
        """
        Establish a connection to the WebSocket server.

        Starts the worker process and waits briefly for connection state updates.

        Returns
        -------
        bool
            True if connection was successful, False otherwise
        """
        if self._worker_process and self._worker_process.is_alive():
            self._drain_state_queue()
            return self.connected

        self._start_worker_process()
        self._start_event_thread()

        deadline = time.time() + 5
        while time.time() < deadline:
            self._drain_state_queue()
            if self.connected or self.is_policy_violation:
                break
            if self._worker_process and not self._worker_process.is_alive():
                break
            time.sleep(0.05)

        if self.connected:
            return True

        logger.info("Connection failed, retrying in background")
        return False

    def send_message(self, message: str | bytes):
        """
        Queue a message to be sent through the WebSocket connection.

        Parameters
        ----------
        message : Union[str, bytes]
            The message to send, either as a string or bytes
        """
        if self.connected and self.running:
            self.message_queue.put(message)

    def _run_client(self):
        """
        Legacy no-op compatibility shim for old thread-based implementation.
        """
        self._start_event_thread()

    def start(self):
        """
        Start the WebSocket client.

        Starts a worker process that manages the WebSocket connection.
        """
        if self._worker_process and self._worker_process.is_alive():
            logger.warning("WebSocket client process is already running")
            return

        self.running = True
        self._start_worker_process()
        self._start_event_thread()
        logger.info("WebSocket client started")

    def register_message_callback(self, callback: Callable):
        """
        Register a callback function for handling received messages.

        Parameters
        ----------
        callback : Callable[[Union[str, bytes]], Any]
            Function to be called when a message is received. Should accept
            either string or bytes as input.
        """
        self.message_callback = callback
        logger.info("Registered message callback")

    def format_message(self, msg: Union[str, bytes], max_length: int = 200) -> str:
        """
        Format a message for logging purposes, truncating if necessary.

        Parameters
        ----------
        msg : Union[str, bytes]
            The message to format
        max_length : int, optional
            Maximum length of the formatted message, by default 200

        Returns
        -------
        str
            The formatted message string
        """
        try:
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8")
            if len(msg) <= max_length:
                return msg
            preview_size = max_length // 2 - 20
            return f"{msg[:preview_size]}...{msg[-preview_size:]}"
        except Exception as e:
            return f"<Error formatting message: {e}>"

    def is_connected(self) -> bool:
        """
        Check if the client is currently connected.

        Returns
        -------
        bool
            True if connected to the WebSocket server, False otherwise
        """
        return self.connected

    def stop(self):
        """
        Stop the WebSocket client.

        Stops worker process and event thread, then cleans up resources.
        """
        self.running = False

        try:
            self._control_queue.put_nowait("stop")
        except Exception:
            pass

        if self._worker_process and self._worker_process.is_alive():
            self._worker_process.join(timeout=2.0)
            if self._worker_process.is_alive():
                self._worker_process.terminate()
                self._worker_process.join(timeout=1.0)

        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=2.0)
            if self._event_thread.is_alive():
                logger.warning("Client event thread did not terminate gracefully")
            else:
                logger.info("Client event thread stopped")

        self.client_thread = None
        self._event_thread = None
        self._worker_process = None
        self.receiver_thread = None
        self.sender_thread = None

        try:
            while True:
                self.message_queue.get_nowait()
        except Empty:
            pass

        try:
            while True:
                self._incoming_queue.get_nowait()
        except Empty:
            pass

        try:
            while True:
                self._state_queue.get_nowait()
        except Empty:
            pass

        try:
            while True:
                self._control_queue.get_nowait()
        except Empty:
            pass

        self.connected = False
        logger.info("WebSocket client stopped")
