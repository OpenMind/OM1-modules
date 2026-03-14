import socket
import time
from contextlib import closing

import pytest
import requests

from om1_utils.healthcheck import HealthCheckServer


def find_free_port() -> int:
    """Find a free port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return sock.getsockname()[1]


@pytest.fixture
def server():
    """Create and manage a HealthCheckServer for testing."""
    port = find_free_port()
    srv = HealthCheckServer(port=port)
    srv.start()
    time.sleep(0.3)
    yield srv, port
    srv.stop()
    time.sleep(0.3)


class TestHealthCheckServer:
    """Tests for the HealthCheckServer class."""

    def test_default_port(self):
        """Test that default port is 9999."""
        srv = HealthCheckServer()
        assert srv.port == 9999

    def test_custom_port(self):
        """Test initialization with a custom port."""
        srv = HealthCheckServer(port=8080)
        assert srv.port == 8080

    def test_initial_state(self):
        """Test that server starts in non-running state."""
        srv = HealthCheckServer()
        assert srv.is_running() is False
        assert srv.server is None
        assert srv.server_thread is None

    def test_start_sets_running(self, server):
        """Test that start sets running to True."""
        srv, _ = server
        assert srv.is_running() is True

    def test_stop_sets_not_running(self, server):
        """Test that stop sets running to False."""
        srv, _ = server
        srv.stop()
        assert srv.is_running() is False

    def test_double_start_does_not_crash(self, server):
        """Test that calling start twice does not raise an error."""
        srv, _ = server
        srv.start()
        assert srv.is_running() is True

    def test_get_returns_200(self, server):
        """Test that GET request returns 200 OK."""
        _, port = server
        response = requests.get(f"http://127.0.0.1:{port}", timeout=5)
        assert response.status_code == 200
        assert response.text == "OK"

    def test_stop_without_start(self):
        """Test that stop on a never-started server does not crash."""
        srv = HealthCheckServer()
        srv.stop()
        assert srv.is_running() is False
