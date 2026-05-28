from prometheus_client import Gauge, Histogram

om1_http_request_duration_seconds = Histogram(
    "om1_http_request_duration_seconds",
    "Total HTTP request duration (client-side) in seconds",
    ["host", "path", "method", "status_code"],
)

om1_http_upstream_total_seconds = Histogram(
    "om1_http_upstream_total_seconds",
    "Upstream total time in seconds (from x-upstream-total-ms header)",
    ["host", "path", "method", "status_code"],
)

om1_http_upstream_ttfb_seconds = Histogram(
    "om1_http_upstream_ttfb_seconds",
    "Upstream TTFB in seconds (from x-upstream-ttfb-ms header)",
    ["host", "path", "method", "status_code"],
)

om1_http_proxy_total_seconds = Histogram(
    "om1_http_proxy_total_seconds",
    "Proxy total time in seconds (from x-proxy-total-ms header)",
    ["host", "path", "method", "status_code"],
)

om1_http_request_duration_last_seconds = Gauge(
    "om1_http_request_duration_last_seconds",
    "Most recent HTTP request duration (client-side) in seconds",
    ["host", "path", "method", "status_code"],
)

om1_http_upstream_total_last_seconds = Gauge(
    "om1_http_upstream_total_last_seconds",
    "Most recent upstream total time in seconds",
    ["host", "path", "method", "status_code"],
)

om1_http_upstream_ttfb_last_seconds = Gauge(
    "om1_http_upstream_ttfb_last_seconds",
    "Most recent upstream TTFB in seconds",
    ["host", "path", "method", "status_code"],
)

om1_http_proxy_total_last_seconds = Gauge(
    "om1_http_proxy_total_last_seconds",
    "Most recent proxy total time in seconds",
    ["host", "path", "method", "status_code"],
)
