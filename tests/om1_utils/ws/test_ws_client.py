import multiprocessing as mp
from unittest.mock import Mock

import pytest

from om1_utils.ws import Client


@pytest.fixture
def mock_websocket():
    """Fixture to create a mock websocket connection"""
    mock_ws = Mock()
    # Mock the required websocket methods
    mock_ws.recv.return_value = "test message"
    mock_ws.send.return_value = None
    mock_ws.close.return_value = None
    # Prevent actual socket operations
    mock_ws.socket = Mock()
    return mock_ws


@pytest.fixture
def client():
    """Fixture to create a client instance"""
    return Client("ws://test.com")


def test_client_initialization(client):
    """Test client initialization with default values"""
    assert client.url == "ws://test.com"
    assert client.running is True
    assert client.connected is False
    assert client.websocket is None
    assert client.message_callback is None
    assert isinstance(client.message_queue, mp.queues.Queue)
    assert client.client_thread is None


def test_register_message_callback(client):
    """Test callback registration"""

    def callback(message):
        pass

    client.register_message_callback(callback)
    assert client.message_callback == callback


def test_send_message_when_connected(client):
    """Test message sending when client is connected"""
    client.connected = True
    test_message = "Hello, WebSocket!"

    client.send_message(test_message)

    assert client.message_queue.get(timeout=0.5) == test_message


def test_send_message_when_disconnected(client):
    """Test message sending when client is disconnected"""
    client.connected = False
    test_message = "Hello, WebSocket!"

    client.send_message(test_message)

    assert client.message_queue.empty()


def test_format_message_short(client):
    """Test message formatting with short message"""
    short_message = "Short test message"
    formatted = client.format_message(short_message)
    assert formatted == short_message


def test_format_message_long(client):
    """Test message formatting with long message"""
    long_message = "x" * 300
    formatted = client.format_message(long_message, max_length=100)
    assert len(formatted) <= 100
    assert "..." in formatted


def test_stop_client(client):
    """Test client stop functionality"""
    client.connected = True

    client.stop()

    assert client.running is False
    assert client.connected is False
    assert client.message_queue.empty()
    assert client.client_thread is None


def test_drain_incoming_messages_with_callback(client):
    """Test callback dispatch from incoming queue"""
    received_messages = []

    def callback(message):
        received_messages.append(message)

    client.register_message_callback(callback)
    client._incoming_queue.put(("message", "test message"))

    client._drain_incoming_queue()

    assert len(received_messages) > 0
    assert received_messages[0] == "test message"
