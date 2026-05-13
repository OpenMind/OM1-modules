from prometheus_client import Gauge, Histogram

om1_tts_latency = Histogram(
    "om1_tts_latency_seconds", "Latency of TTS processing in seconds", ["model"]
)

om1_tts_latency_last = Gauge(
    "om1_tts_latency_last_seconds",
    "Latency of the last TTS processing in seconds",
    ["model"],
)
