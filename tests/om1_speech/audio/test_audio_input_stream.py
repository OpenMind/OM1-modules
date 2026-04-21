import base64
import json
import queue
import threading
from unittest.mock import MagicMock, Mock, patch

import pyaudio
import pytest

from om1_speech import AudioInputStream


@pytest.fixture
def mock_pyaudio():
    with patch("pyaudio.PyAudio") as mock:
        # Setup default device info
        default_device = {
            "index": 0,
            "name": "Default Test Device",
            "maxInputChannels": 1,
            "defaultSampleRate": 16000,
        }
        mock.return_value.get_default_input_device_info.return_value = default_device
        mock.return_value.get_device_info_by_index.return_value = default_device

        mock_stream = MagicMock()
        mock_stream.stop_stream = Mock()
        mock_stream.close = Mock()
        mock.return_value.open.return_value = mock_stream

        yield mock


@pytest.fixture
def audio_stream(mock_pyaudio):
    stream = AudioInputStream(rate=16000, chunk=4048, device=None)
    yield stream
    stream.stop()


def test_initialization(mock_pyaudio):
    """Test AudioInputStream initialization with default parameters"""
    stream = AudioInputStream()

    assert stream._rate == 16000
    assert stream._chunk == 3200
    assert stream._device == 0
    assert stream.running is True
    assert stream._is_tts_active is False
    assert isinstance(stream._buff, queue.Queue)
    assert stream._audio_interface is not None
    assert stream._audio_stream is None
    assert stream._audio_thread is None
    assert stream._language_code == "en-US"
    assert stream._alternative_language_codes == []

    stream.stop()


def test_start_with_default_device(audio_stream, mock_pyaudio):
    """Test starting AudioInputStream with default device"""
    audio_stream.start()

    mock_pyaudio.assert_called_once()
    mock_pyaudio.return_value.get_default_input_device_info.assert_called_once()
    mock_pyaudio.return_value.open.assert_called_once_with(
        format=pyaudio.paInt16,
        input_device_index=0,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4048,
        stream_callback=audio_stream._fill_buffer,
    )

    audio_stream.stop()


def test_start_with_specific_device(mock_pyaudio):
    """Test starting AudioInputStream with a specific device"""
    stream = AudioInputStream(device=1)
    stream.start()

    mock_pyaudio.return_value.get_device_info_by_index.assert_called_once_with(1)
    stream.stop()


def test_audio_data_callbacks(mock_pyaudio):
    """Test audio data callbacks"""
    stream = AudioInputStream(device=1)
    stream.register_audio_data_callback(lambda data: None)

    assert len(stream._audio_data_callbacks) == 1

    stream.stop()


@pytest.mark.parametrize("is_active", [True, False])
def test_tts_state_change(audio_stream, is_active):
    """Test TTS state changes"""
    audio_stream.on_tts_state_change(is_active)
    assert audio_stream._is_tts_active == is_active


def test_fill_buffer_with_tts_inactive(audio_stream):
    """Test buffer filling when TTS is inactive"""
    test_data = b"test_audio_data"
    audio_stream._fill_buffer(test_data, 1024, {}, 0)

    assert audio_stream._buff.get() == test_data


def test_fill_buffer_with_tts_active(audio_stream):
    """Test buffer filling when TTS is active"""
    audio_stream.on_tts_state_change(True)
    test_data = b"test_audio_data"
    audio_stream._fill_buffer(test_data, 1024, {}, 0)

    with pytest.raises(queue.Empty):
        audio_stream._buff.get_nowait()


def test_generator(audio_stream):
    """Test audio data generation"""
    audio_stream.running = True
    test_chunks = [b"chunk1", b"chunk2", b"chunk3"]

    for chunk in test_chunks:
        audio_stream._buff.put(chunk)

    collected_chunks = []

    def collect_data():
        for data in audio_stream.generator():
            collected_chunks.append(data)
            if len(collected_chunks) >= len(test_chunks):
                break

    collection_thread = threading.Thread(target=collect_data)
    collection_thread.daemon = True
    collection_thread.start()

    collection_thread.join(timeout=1.0)

    audio_stream.running = False
    audio_stream._buff.put(None)

    assert len(collected_chunks) > 0
    assert all(isinstance(data, dict) for data in collected_chunks)


def test_stop(audio_stream, mock_pyaudio):
    """Test stopping the audio stream"""
    audio_stream.start()
    audio_stream.stop()

    # Verify stream was stopped
    assert audio_stream.running is False
    assert audio_stream._audio_thread is None


def test_audio_callback(mock_pyaudio):
    """Test audio data callback functionality"""
    callback_data = None
    fixed_time = 1234567890

    def test_callback(data):
        nonlocal callback_data
        callback_data = data

    with patch("om1_speech.audio.audio_input_stream.time") as mock_time:
        mock_time.time.return_value = fixed_time

        stream = AudioInputStream(audio_data_callback=test_callback)
        stream.start()

        # Simulate receiving audio data
        test_data = b"test_audio_data"
        stream._buff.put(test_data)
        stream._buff.put(None)

        # Process one chunk through generator
        next(stream.generator())

    # Verify callback was called with correct data
    assert callback_data == json.dumps(
        {
            "audio": base64.b64encode(test_data).decode("utf-8"),
            "rate": 16000,
            "language_code": "en-US",
            "alternative_language_codes": [],
            "timestamp": fixed_time,
        }
    )

    stream.stop()


def test_error_handling(mock_pyaudio):
    """Test error handling during stream initialization"""
    mock_pyaudio.return_value.open.side_effect = Exception("Test error")

    stream = AudioInputStream()
    with pytest.raises(Exception, match="Test error"):
        stream.start()

    assert stream._audio_stream is None

    stream.stop()


@pytest.mark.parametrize(
    "rate,chunk,device", [(8000, 2048, None), (44100, 1024, 1), (48000, 8192, 2)]
)
def test_different_configurations(mock_pyaudio, rate, chunk, device):
    """Test AudioInputStream with different configurations"""
    stream = AudioInputStream(rate=rate, chunk=chunk, device=device)
    stream.start()

    # Verify stream was opened with correct parameters
    mock_pyaudio.return_value.open.assert_called_once_with(
        format=pyaudio.paInt16,
        input_device_index=device if device is not None else 0,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
        stream_callback=stream._fill_buffer,
    )

    stream.stop()


def test_initialization_with_alternative_languages(mock_pyaudio):
    """Test AudioInputStream initialization with alternative language codes"""
    alt_langs = ["es-ES", "fr-FR", "de-DE"]
    stream = AudioInputStream(
        language_code="en-US", alternative_language_codes=alt_langs
    )

    assert stream._language_code == "en-US"
    assert stream._alternative_language_codes == alt_langs

    stream.stop()


def test_generator_with_alternative_languages(mock_pyaudio):
    """Test generator output includes alternative language codes"""
    alt_langs = ["es-ES", "fr-FR"]
    stream = AudioInputStream(
        language_code="en-GB", alternative_language_codes=alt_langs
    )
    stream.start()

    test_data = b"test_audio_data"
    stream._buff.put(test_data)
    stream._buff.put(None)

    generated = next(stream.generator())

    assert generated["language_code"] == "en-GB"
    assert generated["alternative_language_codes"] == alt_langs
    assert generated["audio"] == base64.b64encode(test_data).decode("utf-8")
    assert generated["rate"] == 16000

    stream.stop()


def test_fill_buffer_remote_with_alternative_languages(mock_pyaudio):
    """Test remote buffer filling with alternative language codes"""
    stream = AudioInputStream(remote_input=True)
    stream.start()

    test_audio = b"remote_audio_data"
    remote_data = json.dumps(
        {
            "audio": base64.b64encode(test_audio).decode("utf-8"),
            "rate": 48000,
            "language_code": "ja-JP",
            "alternative_language_codes": ["zh-CN", "ko-KR"],
        }
    )

    stream.fill_buffer_remote(remote_data)

    buffered_data = stream._buff.get()
    assert buffered_data == test_audio
    assert stream._rate == 48000
    assert stream._language_code == "ja-JP"

    stream.stop()


def test_audio_callback_with_alternative_languages(mock_pyaudio):
    """Test audio data callback with alternative language codes"""
    callback_data = None
    alt_langs = ["it-IT", "pt-BR"]

    def test_callback(data):
        nonlocal callback_data
        callback_data = data

    fixed_time = 1234567890

    with patch("om1_speech.audio.audio_input_stream.time") as mock_time:
        mock_time.time.return_value = fixed_time

        stream = AudioInputStream(
            audio_data_callback=test_callback,
            language_code="es-MX",
            alternative_language_codes=alt_langs,
        )
        stream.start()

        test_data = b"test_audio_data"
        stream._buff.put(test_data)
        stream._buff.put(None)

        next(stream.generator())

    expected_data = json.dumps(
        {
            "audio": base64.b64encode(test_data).decode("utf-8"),
            "rate": 16000,
            "language_code": "es-MX",
            "alternative_language_codes": alt_langs,
            "timestamp": fixed_time,
        }
    )
    assert callback_data == expected_data

    stream.stop()
