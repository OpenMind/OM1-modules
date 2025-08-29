# -*- coding: utf-8 -*-
"""
Design goals
------------
- Prefer the sync client (websockets 12+) so send/recv are simple blocking calls.
- Fall back to the async client (older websockets) by running an asyncio loop
  in a background thread and scheduling async send/receive tasks there.
- Maintain a message queue and two loops (send/receive) so the app thread never blocks.
- Auto-reconnect unless there was a policy violation.
"""
import logging
import threading
from queue import Empty, Queue
from typing import Callable, Optional, Union

# ---- websockets 版本兼容：优先使用同步 API；否则回退到 asyncio API ----
try:
    # websockets 12+ 同步客户端
    from websockets.sync.client import connect as ws_connect  # type: ignore
    from websockets.sync.client import ClientConnection as WSConnSync  # type: ignore
    SYNC_WS = True
except Exception:
    SYNC_WS = False
    import websockets  # asyncio 版整包
    try:
        # websockets 12+ asyncio 客户端
        from websockets.asyncio.client import connect as ws_connect  # type: ignore
    except Exception:
        try:
            # websockets 11/10
            from websockets.client import connect as ws_connect  # type: ignore
        except Exception:
            # 最保守兜底（11/10 legacy 命名空间）
            from websockets.legacy.client import connect as ws_connect  # type: ignore

# 统一异常类型（不同版本在 exceptions 命名空间）
try:
    if SYNC_WS:
        from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError  # type: ignore
    else:
        from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError  # type: ignore
except Exception:  # 极端兜底
    class ConnectionClosedOK(Exception): ...
    class ConnectionClosedError(Exception): ...

# asyncio 仅在需要时导入，避免无谓依赖
if not SYNC_WS:
    import asyncio

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class Client:
    """
    A WebSocket client implementation with support for asynchronous message handling.

    This class provides a threaded WebSocket client that can maintain a persistent
    connection, automatically reconnect, and handle message sending and receiving
    asynchronously.

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

        # 同步 / 异步统一：同步时是 WSConnSync；异步时是 websockets 的协议实例
        self.websocket: Optional[Union["WSConnSync", object]] = None  # type: ignore

        self.message_callback: Optional[Callable] = None
        self.message_queue: Queue = Queue()
        self.receiver_thread: Optional[threading.Thread] = None
        self.sender_thread: Optional[threading.Thread] = None

        # asyncio 回退模式用
        self._loop: Optional["asyncio.AbstractEventLoop"] = None
        self._async_recv_task = None
        self._async_send_task = None

    # -------------------------
    # 同步收发（优先使用）
    # -------------------------
    def _receive_messages_sync(self):
        """
        Internal method to handle receiving messages from the WebSocket connection.

        Continuously receives messages and processes them through the registered callback
        if one exists. Runs in a separate thread.
        """
        while self.running and self.connected and self.websocket is not None:
            try:
                message = self.websocket.recv()  # blocking
                formatted_msg = self.format_message(message)
                logger.debug(f"Received WS Message: {formatted_msg}")
                if self.message_callback:
                    self.message_callback(message)
            except (ConnectionClosedOK, ConnectionClosedError) as e:
                logger.info(f"WebSocket connection closed: {e}")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                self.connected = False
                break

    def _send_messages_sync(self):
        """
        Internal method to handle sending messages through the WebSocket connection.

        Continuously processes messages from the message queue and sends them through
        the WebSocket connection. Runs in a separate thread.
        """
        while self.running:
            try:
                if self.connected and self.websocket:
                    message = self.message_queue.get_nowait()
                    try:
                        self.websocket.send(message)  # blocking
                        formatted_msg = self.format_message(message)
                        logger.debug(f"Sent WS Message: {formatted_msg} to {self.url}")
                    except Exception as e:
                        logger.error(f"Failed to send message: {e}")
                        # 发送失败时把消息放回队列，避免丢失
                        self.message_queue.put(message)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in send queue processing: {e}")
                self.connected = False

    # -------------------------
    # asyncio 回退模式（仅在无 sync API 时使用）
    # -------------------------
    async def _async_connect(self):
        ws = await ws_connect(self.url)  # type: ignore
        self.websocket = ws
        self.connected = True
        logger.info(f"Connected to {self.url} (async mode)")
        return ws

    async def _async_receive_loop(self):
        assert self.websocket is not None
        ws = self.websocket
        try:
            async for message in ws:  # type: ignore
                formatted_msg = self.format_message(message)
                logger.debug(f"Received WS Message: {formatted_msg}")
                if self.message_callback:
                    try:
                        self.message_callback(message)
                    except Exception as cb_e:
                        logger.error(f"Message callback error: {cb_e}")
        except (ConnectionClosedOK, ConnectionClosedError) as e:
            logger.info(f"WebSocket connection closed (async): {e}")
        except Exception as e:
            logger.error(f"Error in async receive: {e}")
        finally:
            self.connected = False

    async def _async_send_loop(self):
        assert self.websocket is not None
        ws = self.websocket
        while self.running:
            try:
                msg = await self._loop.run_in_executor(None, self.message_queue.get)  # type: ignore
                try:
                    await ws.send(msg)  # type: ignore
                    formatted_msg = self.format_message(msg)
                    logger.debug(f"Sent WS Message: {formatted_msg} to {self.url}")
                except Exception as e:
                    logger.error(f"Failed to send message (async): {e}")
                    # 放回队列重试
                    self.message_queue.put(msg)
                    await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"Error in async send loop: {e}")
                self.connected = False
                await asyncio.sleep(0.5)

    # -------------------------
    # connect to lifecycle
    # -------------------------
    def connect(self) -> bool:
        """
        Establish a connection to the WebSocket server.

        Attempts to connect to the WebSocket server and starts the receiver and sender
        threads if the connection is successful.

        Returns
        -------
        bool
            True if connection was successful, False otherwise
        """
        try:
            if SYNC_WS:
                # 同步直连
                self.websocket = ws_connect(self.url)  # type: ignore
                self.connected = True

                # Start receiver and sender threads
                if not self.receiver_thread or not self.receiver_thread.is_alive():
                    self.receiver_thread = threading.Thread(
                        target=self._receive_messages_sync, daemon=True
                    )
                    self.receiver_thread.start()

                if not self.sender_thread or not self.sender_thread.is_alive():
                    self.sender_thread = threading.Thread(
                        target=self._send_messages_sync, daemon=True
                    )
                    self.sender_thread.start()

                logger.info(f"Connected to {self.url}")
                return True
            else:
                # asyncio 回退：单独的事件循环线程
                if self._loop is None:
                    self._loop = asyncio.new_event_loop()
                    t = threading.Thread(
                        target=lambda: (asyncio.set_event_loop(self._loop), self._loop.run_forever()),
                        daemon=True,
                    )
                    t.start()

                fut = asyncio.run_coroutine_threadsafe(self._async_connect(), self._loop)
                fut.result(timeout=10)

                # 启动异步收发任务
                self._async_recv_task = asyncio.run_coroutine_threadsafe(
                    self._async_receive_loop(), self._loop
                )
                self._async_send_task = asyncio.run_coroutine_threadsafe(
                    self._async_send_loop(), self._loop
                )
                return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False

    def send_message(self, message: Union[str, bytes]):
        """
        Queue a message to be sent through the WebSocket connection.

        Parameters
        ----------
        message : Union[str, bytes]
            The message to send, either as a string or bytes
        """
        if self.connected:
            self.message_queue.put(message)

    def _run_client(self):
        """
        Internal method to manage the WebSocket client lifecycle.

        Continuously attempts to maintain a connection to the WebSocket server,
        implementing automatic reconnection with a delay between attempts.
        """
        while self.running:
            if not self.connected and not self.is_policy_violation:
                if self.connect():
                    logger.info("Connection established")
                else:
                    logger.info("Connection failed, retrying in 5 seconds")
                    threading.Event().wait(5)  # Wait 5 seconds before retrying
            else:
                threading.Event().wait(1)

    def start(self):
        """
        Start the WebSocket client.

        Initializes and starts the main client thread that manages the WebSocket
        connection.
        """
        self.client_thread = threading.Thread(target=self._run_client, daemon=True)
        self.client_thread.start()
        logger.info("WebSocket client thread started")

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
            s = msg if isinstance(msg, str) else (msg.decode("utf-8", "ignore") if isinstance(msg, (bytes, bytearray)) else str(msg))
            if len(s) <= max_length:
                return s
            preview_size = max_length // 2 - 20
            return f"{s[:preview_size]}...{s[-preview_size:]}"
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

        Closes the WebSocket connection, stops all threads, and cleans up resources.
        """
        self.running = False

        # 关闭连接
        try:
            if self.websocket:
                if SYNC_WS:
                    self.websocket.close()  # type: ignore
                else:
                    if self._loop and not self._loop.is_closed():
                        asyncio.run_coroutine_threadsafe(self.websocket.close(), self._loop)  # type: ignore
        except Exception:
            pass

        # 清空队列
        try:
            while True:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
        except Empty:
            pass

        self.connected = False

        # 关闭事件循环（async 回退）
        if not SYNC_WS and self._loop and not self._loop.is_closed():
            def _stop_loop(loop):
                loop.stop()
            self._loop.call_soon_threadsafe(_stop_loop, self._loop)

        logger.info("WebSocket client stopped")
