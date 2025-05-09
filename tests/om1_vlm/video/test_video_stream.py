import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from om1_vlm import VideoStream


class MockVideoCapture:
    def __init__(self, device_index):
        self.device_index = device_index
        self.is_opened = True
        self.frame_count = 0
        self.release_called = False
        self._mock_stream = None
        self._properties = {
            3: 640,
            4: 480,
            5: 30,
        }

    def isOpened(self):
        is_open = self.is_opened
        return is_open

    def getBackendName(self):
        return "Mock"

    def read(self):
        if not self.is_opened:
            return False, None

        if self.frame_count < 10:  # Generate 10 test frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.frame_count += 1
            return True, frame
        return False, None

    def release(self):
        self.is_opened = False
        self.release_called = True

    def set(self, prop_id, value):
        self._properties[prop_id] = value

    def get(self, prop_id):
        return self._properties.get(prop_id, 100)


@pytest.fixture
def mock_cv2():
    with patch("cv2.VideoCapture", MockVideoCapture) as mock:
        with patch("cv2.imencode") as mock_imencode:
            # Mock imencode to return a simple base64 string
            mock_imencode.return_value = (True, b"fake_image_data")
            yield mock


@pytest.fixture
def mock_camera():
    class MockVideoCapture:
        def __init__(self, device_index):
            print(f"Creating mock camera for device: {device_index}")
            self.is_opened = True
            self.frame_count = 0

        def isOpened(self):
            return self.is_opened

        def getBackendName(self):
            return "Mock"

        def read(self):
            if not self.is_opened:
                return False, None

            self.frame_count += 1
            if self.frame_count <= 10:  # Generate 10 test frames
                # Create a simple test pattern
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Add some visible pattern
                frame[200:300, 200:300] = 255  # White square
                return True, frame
            return False, None

        def release(self):
            self.is_opened = False

    return MockVideoCapture


@pytest.fixture(autouse=True)
def mock_video_dependencies(mock_camera):
    with (
        patch("cv2.VideoCapture", mock_camera),
        patch("cv2.imencode", return_value=(True, b"fake_frame_data")),
        patch(
            "om1_vlm.video.video_stream.enumerate_video_devices",
            return_value=[(0, "Mock Camera")],
        ),
        patch("platform.system", return_value="Linux"),
    ):
        yield


@pytest.fixture
def mock_platform():
    with patch("platform.system") as mock:
        mock.return_value = "Linux"
        yield mock


@pytest.fixture
def mock_enumerate_devices():
    with patch("om1_vlm.enumerate_video_devices") as mock:
        mock.return_value = [(0, "Test Camera")]
        yield mock


def test_video_stream_initialization():
    callback = Mock()
    stream = VideoStream(frame_callback=callback)
    assert len(stream.frame_callbacks) == 1
    assert stream.frame_callbacks[0] == callback
    assert stream.running
    assert stream._video_thread is None


@pytest.mark.usefixtures("mock_cv2", "mock_platform", "mock_enumerate_devices")
def test_video_stream_start_stop():
    callback = Mock()
    stream = VideoStream(frame_callback=callback)

    # Start the video stream
    stream.start()
    assert stream._video_thread is not None
    assert stream._video_thread.is_alive()

    # Let it run briefly
    import time

    time.sleep(0.1)

    # Stop the video stream
    stream.stop()
    assert not stream.running


def test_frame_callback():
    received_frames = []

    def callback(frame_data):
        received_frames.append(frame_data)

    # Set up all mocks before creating VideoStream
    mock_devices = [(0, "Mock Camera")]

    with (
        patch(
            "om1_vlm.video.video_stream.enumerate_video_devices",
            return_value=mock_devices,
        ),
        patch("platform.system", return_value="Linux"),
    ):
        # Create mock camera AFTER platform and devices are mocked
        mock_cap = MockVideoCapture("/dev/video0")

        with (
            patch("cv2.VideoCapture", return_value=mock_cap) as mock_capture,
            patch("cv2.imencode", return_value=(True, b"fake_image_data")),
        ):
            # Create and start the video stream
            stream = VideoStream(frame_callback=callback)
            stream.start()

            # Wait for frames
            timeout = 2.0
            start_time = time.time()
            while len(received_frames) == 0 and time.time() - start_time < timeout:
                time.sleep(0.5)

            # Stop the stream
            stream.stop()

            # Verify results
            assert mock_capture.called, "VideoCapture was never created"
            assert mock_cap.release_called, "Camera release was never called"
            assert len(received_frames) > 0, "No frames were received"
            assert isinstance(received_frames[0], str), "Frame data is not a string"


def test_frame_callback_coroutine():
    import asyncio

    received_frames = []

    async def async_frame_callback(frame_data):
        # Simulate asynchronous processing before handling the frame.
        await asyncio.sleep(1)
        received_frames.append(frame_data)

    # Set up all mocks before creating VideoStream
    mock_devices = [(0, "Mock Camera")]

    with (
        patch(
            "om1_vlm.video.video_stream.enumerate_video_devices",
            return_value=mock_devices,
        ),
        patch("platform.system", return_value="Linux"),
    ):
        # Create mock camera AFTER platform and devices are mocked
        mock_cap = MockVideoCapture("/dev/video0")

        with (
            patch("cv2.VideoCapture", return_value=mock_cap) as mock_capture,
            patch("cv2.imencode", return_value=(True, b"fake_image_data")),
        ):
            # Create and start the video stream
            stream = VideoStream(frame_callback=async_frame_callback)
            stream.start()

            # Wait for frames
            timeout = 2.0
            start_time = time.time()
            # The async callback should be called in parallel, so it should be higher than 2 (with 1s sleep)
            while len(received_frames) <= 2 and time.time() - start_time < timeout:
                time.sleep(0.5)

            # Stop the stream
            stream.stop()

            # Verify results
            assert mock_capture.called, "VideoCapture was never created"
            assert mock_cap.release_called, "Camera release was never called"
            assert len(received_frames) > 2, "Less than 2 frames were received"
            assert isinstance(received_frames[0], str), "Frame data is not a string"


@pytest.mark.usefixtures("mock_cv2", "mock_platform", "mock_enumerate_devices")
def test_camera_device_selection():
    stream = VideoStream()

    # Test Linux camera selection
    with patch("platform.system", return_value="Linux"):
        stream.start()
        assert stream._video_thread is not None
    stream.stop()

    # Test macOS camera selection
    with patch("platform.system", return_value="Darwin"):
        stream.start()
        assert stream._video_thread is not None
    stream.stop()


def test_register_frame_callback():
    stream = VideoStream()
    callback = Mock()
    stream.register_frame_callback(callback)
    assert len(stream.frame_callbacks) == 1
    assert stream.frame_callbacks[0] == callback


def test_register_multiple_frame_callbacks():
    stream = VideoStream()
    callback1 = Mock()
    callback2 = Mock()
    stream.register_frame_callback(callback1)
    stream.register_frame_callback(callback2)
    assert len(stream.frame_callbacks) == 2
    assert stream.frame_callbacks[0] == callback1
    assert stream.frame_callbacks[1] == callback2


@pytest.mark.usefixtures("mock_cv2", "mock_platform", "mock_enumerate_devices")
def test_error_handling():
    # Test with a mock that simulates camera not found
    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap.return_value.isOpened.return_value = False

        stream = VideoStream()
        stream.start()

        # Let it try to initialize
        import time

        time.sleep(0.1)

        # Verify error handling
        stream.stop()
        assert not stream._video_thread.is_alive()
