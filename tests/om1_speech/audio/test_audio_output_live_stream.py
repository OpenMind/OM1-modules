import base64
import json
import subprocess
import threading
import time
from queue import Queue
from unittest.mock import Mock, patch

import pytest

from om1_speech.audio.audio_output_live_stream import (
    AudioOutputLiveStream,
    is_installed,
)


@pytest.fixture
def mock_openai():
    """Mock OpenAI client and streaming response"""
    with patch("openai.OpenAI") as mock:
        # Create mock client
        mock_client = Mock()

        # Create mock streaming response context manager
        mock_stream_response = Mock()
        mock_stream_response.__enter__ = Mock(return_value=mock_stream_response)
        mock_stream_response.__exit__ = Mock(return_value=None)
        mock_stream_response.iter_bytes = Mock(
            return_value=[b"chunk1", b"chunk2", b"chunk3"]
        )

        # Chain the mocks: client.audio.speech.with_streaming_response.create
        mock_client.audio.speech.with_streaming_response.create = Mock(
            return_value=mock_stream_response
        )

        mock.return_value = mock_client
        yield mock


@pytest.fixture
def mock_zenoh():
    """Mock Zenoh session and messaging"""
    with patch(
        "om1_speech.audio.audio_output_live_stream.open_zenoh_session"
    ) as mock_session:
        mock_sess = Mock()
        mock_pub = Mock()
        mock_sub = Mock()

        mock_sess.declare_publisher.return_value = mock_pub
        mock_sess.declare_subscriber.return_value = mock_sub
        mock_sess.close.return_value = None

        mock_session.return_value = mock_sess
        yield mock_sess


@pytest.fixture
def mock_ffplay():
    """Mock ffplay subprocess"""
    with patch("subprocess.Popen") as mock:
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.stdin = Mock()
        mock_process.stdin.write = Mock()
        mock_process.stdin.flush = Mock()
        mock_process.stdin.close = Mock()
        mock_process.wait = Mock()
        mock_process.terminate = Mock()
        mock_process.kill = Mock()

        mock.return_value = mock_process
        yield mock


@pytest.fixture
def mock_is_installed():
    """Mock is_installed function"""
    with patch("om1_speech.audio.audio_output_live_stream.is_installed") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def audio_stream(mock_openai, mock_zenoh, mock_ffplay, mock_is_installed):
    """Create AudioOutputLiveStream instance with mocked dependencies"""
    stream = AudioOutputLiveStream(
        url="http://test-tts-server/v1",
        tts_model="test-model",
        tts_voice="test-voice",
        rate=24000,
    )
    yield stream
    stream.stop()


def test_initialization(audio_stream):
    """Test AudioOutputLiveStream initialization with default parameters"""
    assert audio_stream._url == "http://test-tts-server/v1"
    assert audio_stream._tts_model == "test-model"
    assert audio_stream._tts_voice == "test-voice"
    assert audio_stream._rate == 24000
    assert audio_stream._response_format == "pcm"
    assert audio_stream.running is True
    assert isinstance(audio_stream._pending_requests, Queue)
    assert audio_stream._ffplay_proc is None
    assert audio_stream._ffplay_initialized is False


def test_initialization_with_custom_params(mock_openai, mock_zenoh):
    """Test initialization with custom parameters"""

    def custom_callback(is_active):
        pass

    stream = AudioOutputLiveStream(
        url="http://custom-server/v1",
        tts_model="custom-model",
        tts_voice="custom-voice",
        response_format="wav",
        rate=16000,
        api_key="test-api-key",
        tts_state_callback=custom_callback,
        enable_tts_interrupt=True,
    )

    assert stream._url == "http://custom-server/v1"
    assert stream._tts_model == "custom-model"
    assert stream._tts_voice == "custom-voice"
    assert stream._response_format == "wav"
    assert stream._rate == 16000
    assert stream._api_key == "test-api-key"
    assert stream._tts_state_callback == custom_callback
    assert stream._enable_tts_interrupt is True

    stream.stop()


def test_add_request(audio_stream):
    """Test adding requests to the queue"""
    request1 = {"text": "Hello world"}
    request2 = {"text": "Goodbye world"}

    audio_stream.add_request(request1)
    audio_stream.add_request(request2)

    assert audio_stream._pending_requests.qsize() == 2
    assert audio_stream._pending_requests.get() == request1
    assert audio_stream._pending_requests.get() == request2


def test_set_tts_state_callback(audio_stream):
    """Test setting TTS state callback"""
    callback_called = []

    def callback(is_active):
        callback_called.append(is_active)

    audio_stream.set_tts_state_callback(callback)
    audio_stream._tts_callback(True)
    audio_stream._tts_callback(False)

    assert callback_called == [True, False]


def test_initialize_ffplay_success(audio_stream, mock_ffplay, mock_is_installed):
    """Test successful ffplay initialization"""
    result = audio_stream._initialize_ffplay()

    assert result is True
    assert audio_stream._ffplay_proc is not None
    assert audio_stream._ffplay_initialized is True

    call_args = mock_ffplay.call_args
    args = call_args[1]["args"]
    assert "ffplay" in args
    assert "-f" in args
    assert "s16le" in args
    assert "-ar" in args
    assert str(audio_stream._rate) in args
    assert "-nodisp" in args
    assert "-autoexit" in args


def test_initialize_ffplay_already_running(audio_stream, mock_ffplay):
    """Test ffplay initialization when already running"""
    audio_stream._initialize_ffplay()
    first_proc = audio_stream._ffplay_proc

    result = audio_stream._initialize_ffplay()

    assert result is True
    assert audio_stream._ffplay_proc is first_proc  # Same process


def test_initialize_ffplay_not_installed(audio_stream, mock_is_installed):
    """Test ffplay initialization when ffplay is not installed"""
    mock_is_installed.return_value = False

    result = audio_stream._initialize_ffplay()

    assert result is False
    assert audio_stream._ffplay_proc is None
    assert audio_stream._ffplay_initialized is False


def test_stream_audio_chunk(audio_stream, mock_ffplay):
    """Test streaming audio chunks to ffplay"""
    audio_stream._initialize_ffplay()

    test_chunk = b"test_audio_data"
    audio_stream._stream_audio_chunk(test_chunk)

    audio_stream._ffplay_proc.stdin.write.assert_called_with(test_chunk)
    audio_stream._ffplay_proc.stdin.flush.assert_called()


def test_stream_audio_chunk_broken_pipe(audio_stream, mock_ffplay):
    """Test handling of broken pipe during streaming"""
    audio_stream._initialize_ffplay()

    mock_process = audio_stream._ffplay_proc
    mock_process.stdin.close.return_value = None
    mock_process.terminate.return_value = None
    mock_process.wait.return_value = None  # Simulate successful wait

    audio_stream._ffplay_proc.stdin.write.side_effect = BrokenPipeError()

    test_chunk = b"test_audio_data"
    audio_stream._stream_audio_chunk(test_chunk)

    assert audio_stream._ffplay_proc is None


def test_finish_audio_playback(audio_stream, mock_ffplay):
    """Test finishing audio playback"""
    audio_stream._initialize_ffplay()

    mock_process = audio_stream._ffplay_proc

    audio_stream._finish_audio_playback()

    mock_process.stdin.close.assert_called_once()
    mock_process.wait.assert_called_once()

    assert audio_stream._ffplay_proc is None


def test_finish_audio_playback_timeout(audio_stream, mock_ffplay):
    """Test finishing audio playback with timeout"""
    audio_stream._initialize_ffplay()

    mock_process = audio_stream._ffplay_proc

    mock_process.wait.side_effect = subprocess.TimeoutExpired("ffplay", 10)

    audio_stream._finish_audio_playback()

    mock_process.kill.assert_called_once()

    assert audio_stream._ffplay_proc is None


def test_cleanup_ffplay(audio_stream, mock_ffplay):
    """Test cleaning up ffplay process"""
    audio_stream._initialize_ffplay()

    mock_process = audio_stream._ffplay_proc

    audio_stream._cleanup_ffplay()

    mock_process.stdin.close.assert_called_once()
    mock_process.terminate.assert_called_once()
    assert audio_stream._ffplay_proc is None
    assert audio_stream._ffplay_initialized is False


def test_cleanup_ffplay_force_kill(audio_stream, mock_ffplay):
    """Test force killing ffplay process on timeout"""
    audio_stream._initialize_ffplay()

    mock_process = audio_stream._ffplay_proc

    mock_process.wait.side_effect = subprocess.TimeoutExpired("ffplay", 2)

    audio_stream._cleanup_ffplay()

    mock_process.kill.assert_called_once()

    assert audio_stream._ffplay_proc is None
    assert audio_stream._ffplay_initialized is False


def test_update_audio_status(audio_stream, mock_zenoh):
    """Test updating and publishing audio status"""
    with patch("om1_speech.audio.audio_output_live_stream.AudioStatus") as mock_status:
        with patch("om1_speech.audio.audio_output_live_stream.prepare_header"):
            mock_state = Mock()
            mock_state.serialize.return_value = b"serialized_data"
            mock_status.return_value = mock_state

            audio_stream._update_audio_status(1)

            audio_stream.pub.put.assert_called_with(b"serialized_data")


def test_create_silence_audio(audio_stream):
    """Test creating silence audio"""
    silence = audio_stream._create_silence_audio(100)

    assert isinstance(silence, bytes)

    decoded = base64.b64decode(silence)
    assert all(b == 0 for b in decoded)


def test_write_audio(audio_stream, mock_ffplay):
    """Test writing audio data"""
    audio_stream._initialize_ffplay()

    test_audio = base64.b64encode(b"test_audio_data")
    audio_stream._write_audio_bytes(test_audio)

    audio_stream._ffplay_proc.stdin.write.assert_called()


def test_write_audio_raw_data(audio_stream, mock_ffplay):
    """Test writing raw audio data (not base64)"""
    audio_stream._initialize_ffplay()

    test_audio = b"raw_audio_data"
    audio_stream._write_audio_bytes(test_audio)

    audio_stream._ffplay_proc.stdin.write.assert_called()


def test_process_audio(audio_stream, mock_openai, mock_ffplay):
    """Test audio processing flow"""
    callback_states = []

    def callback(is_active):
        callback_states.append(is_active)

    audio_stream.set_tts_state_callback(callback)
    audio_stream.add_request({"text": "Test message"})

    thread = threading.Thread(target=audio_stream._process_audio)
    thread.daemon = True
    thread.start()

    time.sleep(0.2)

    audio_stream.stop()
    thread.join(timeout=1)

    mock_openai.return_value.audio.speech.with_streaming_response.create.assert_called()


def test_process_audio_error_handling(audio_stream, mock_openai):
    """Test error handling in audio processing"""
    mock_openai.return_value.audio.speech.with_streaming_response.create.side_effect = (
        Exception("Test error")
    )

    audio_stream.add_request({"text": "Test message"})

    thread = threading.Thread(target=audio_stream._process_audio)
    thread.daemon = True
    thread.start()

    time.sleep(0.2)

    audio_stream.stop()
    thread.join(timeout=1)

    assert audio_stream.running is False


def test_zenoh_audio_message(audio_stream):
    """Test handling Zenoh audio messages"""
    with patch("om1_speech.audio.audio_output_live_stream.AudioStatus") as mock_status:
        # Create mock audio status
        mock_audio_status = Mock()
        mock_audio_status.sentence_to_speak.data = json.dumps({"text": "Test"})
        mock_audio_status.status_speaker = 1  # ACTIVE

        mock_status.deserialize.return_value = mock_audio_status
        mock_status.STATUS_SPEAKER.ACTIVE.value = 1

        mock_sample = Mock()
        mock_sample.payload.to_bytes.return_value = b"test_data"

        audio_stream.zenoh_audio_message(mock_sample)

        assert audio_stream._pending_requests.qsize() == 1


def test_on_asr_text_interrupt(audio_stream):
    """Test ASR text interrupt functionality"""
    audio_stream._enable_tts_interrupt = True

    with patch("om1_speech.audio.audio_output_live_stream.AudioStatus") as mock_status:
        # Setup audio status to indicate active speaker
        audio_stream.audio_status = Mock()
        audio_stream.audio_status.status_speaker = 1  # ACTIVE
        mock_status.STATUS_SPEAKER.ACTIVE.value = 1

        audio_stream.add_request({"text": "Request 1"})
        audio_stream.add_request({"text": "Request 2"})

        mock_sample = Mock()
        mock_sample.payload.to_bytes.return_value = b"speech detected"

        audio_stream._on_asr_text(mock_sample)

        assert audio_stream._pending_requests.qsize() == 0


def test_on_asr_text_no_interrupt_when_disabled(audio_stream):
    """Test that ASR doesn't interrupt when disabled"""
    audio_stream._enable_tts_interrupt = False

    audio_stream.add_request({"text": "Request 1"})
    audio_stream.add_request({"text": "Request 2"})

    mock_sample = Mock()
    mock_sample.payload.to_bytes.return_value = b"speech detected"

    audio_stream._on_asr_text(mock_sample)

    assert audio_stream._pending_requests.qsize() == 2


def test_start(audio_stream):
    """Test starting the audio stream"""
    with patch("threading.Thread") as mock_thread:
        audio_stream.start()

        assert mock_thread.call_count == 2


def test_stop(audio_stream, mock_ffplay):
    """Test stopping the audio stream"""
    audio_stream._initialize_ffplay()
    audio_stream.stop()

    assert audio_stream.running is False
    # Cleanup should be called
    assert audio_stream._ffplay_proc is None


def test_keepalive_worker(audio_stream):
    """Test keepalive worker functionality"""
    audio_stream._last_audio_time = time.time() - 65  # 65 seconds ago

    # Mock write_audio_bytes
    with patch.object(audio_stream, "_write_audio_bytes") as mock_write:
        thread = threading.Thread(target=audio_stream._keepalive_worker)
        thread.daemon = True
        thread.start()

        time.sleep(0.2)

        audio_stream.stop()
        thread.join(timeout=1)

        mock_write.assert_called()


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 44100])
def test_different_sample_rates(mock_openai, mock_zenoh, rate):
    """Test initialization with different sample rates"""
    stream = AudioOutputLiveStream(
        url="http://test-server/v1",
        tts_model="test",
        tts_voice="test",
        rate=rate,
    )

    assert stream._rate == rate
    stream.stop()


@pytest.mark.parametrize("response_format", ["pcm", "wav", "mp3"])
def test_different_response_formats(mock_openai, mock_zenoh, response_format):
    """Test initialization with different response formats"""
    stream = AudioOutputLiveStream(
        url="http://test-server/v1",
        tts_model="test",
        tts_voice="test",
        response_format=response_format,
    )

    assert stream._response_format == response_format
    stream.stop()


def test_is_installed_function():
    """Test the is_installed utility function"""

    assert is_installed("ls") is True or is_installed("dir") is True
    assert is_installed("nonexistent_command_12345") is False


def test_openai_client_creation(mock_openai, mock_zenoh):
    """Test OpenAI client is created correctly"""
    stream = AudioOutputLiveStream(
        url="http://test-server/v1",
        tts_model="test",
        tts_voice="test",
        api_key="test-key",
    )

    mock_openai.assert_called_once_with(
        base_url="http://test-server/v1",
        api_key="test-key",
    )

    stream.stop()


def test_openai_client_no_api_key(mock_openai, mock_zenoh):
    """Test OpenAI client creation without API key"""
    stream = AudioOutputLiveStream(
        url="http://test-server/v1",
        tts_model="test",
        tts_voice="test",
    )

    mock_openai.assert_called_once_with(
        base_url="http://test-server/v1",
        api_key="no-need-api-key",
    )

    stream.stop()


def test_zenoh_session_failure(mock_openai):
    """Test handling of Zenoh session initialization failure"""
    with patch(
        "om1_speech.audio.audio_output_live_stream.open_zenoh_session"
    ) as mock_session:
        mock_session.side_effect = Exception("Zenoh connection failed")

        stream = AudioOutputLiveStream(
            url="http://test-server/v1",
            tts_model="test",
            tts_voice="test",
        )

        assert stream.session is None
        stream.stop()


def test_write_audio_bytes(audio_stream, mock_ffplay):
    """Test writing audio bytes directly"""
    audio_stream._initialize_ffplay()

    test_data = base64.b64encode(b"test_audio")
    audio_stream._write_audio_bytes(test_data)

    audio_stream._ffplay_proc.stdin.write.assert_called()


def test_ffplay_process_reinitialization(audio_stream, mock_ffplay):
    """Test ffplay process reinitialization after failure"""
    audio_stream._initialize_ffplay()

    audio_stream._ffplay_proc.poll.return_value = 1  # Process exited
    audio_stream._stream_audio_chunk(b"test")

    assert audio_stream._ffplay_proc is None
