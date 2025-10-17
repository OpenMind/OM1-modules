#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# teach_face.sh — helper to talk to run.py’s HTTP API
#
# What it does
#   - who:     query who has been seen recently (presence snapshot)
#   - config:  get / set live runtime settings (e.g., conf, nms, blur_mode…)
#   - selfie:  enroll from the latest clean frame (aligned only, immediate embed)
#   - upload:  copy a raw image to gallery/<name>/raw then refresh (align+embed)
#
# Requirements
#   - curl, jq
#
# Endpoint
#   - FACE_HTTP (env) → defaults to http://127.0.0.1:6791
#     export FACE_HTTP="http://<host>:<port>"
#
# Quick usage
#   # default recent_sec = 2
#   ./teach_face.sh who
#
#   # 1-second window
#   ./teach_face.sh who 1
#
#   # change a setting (e.g., detector confidence)
#   ./teach_face.sh config set conf=0.7
#
#   # check current config
#   ./teach_face.sh config get
#
#   # take a selfie for “wendy” (temporarily disables blur, waits for exactly 1 face)
#   ./teach_face.sh selfie wendy
#
#   # upload an existing image to raw, then refresh/align/embed
#   ./teach_face.sh upload wendy /abs/path/to/image.jpg
#
# Notes
#   - Selfie temporarily sets blur=false and restores it afterward.
#   - `who` aggregates presence over the last N seconds (recent_sec; default 2).
#   - `config set` infers value types: true/false/number are unquoted; strings are quoted.
#   - The script calls: /ping, /who, /config, /selfie, /gallery/add_raw, /gallery/refresh.
#   - Ensure run.py is running with matching --http-host / --http-port.
# ---------------------------------------------------------------------------

set -eE -o pipefail
IFS=$'\n\t'

FACE_HTTP="${FACE_HTTP:-http://127.0.0.1:6793}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERR] '$1' is required"; exit 1; }; }
need_cmd jq
need_cmd curl

_curl() { curl -sS --max-time 5 -H 'Content-Type: application/json' "$@"; }
post_json()       { _curl -f -d "$1" "$FACE_HTTP$2"; }          # fail on HTTP errors
post_json_soft()  { _curl    -d "$1" "$FACE_HTTP$2" || true; }  # ignore HTTP errors (polling)

pretty() {
  local s
  if [ $# -gt 0 ]; then
    s="$1"
  else
    # read from stdin if no argument
    s="$(cat)"
  fi

  if [ -z "${s:-}" ]; then
    echo "[WARN] empty response"
    return 0
  fi

  if jq . >/dev/null 2>&1 <<<"$s"; then
    jq . <<<"$s"
  else
    printf "%s\n" "$s"
  fi
}

usage() {
  cat <<EOF
Usage:
  $(basename "$0") selfie <name>
  $(basename "$0") upload <name> <abs_path>
  $(basename "$0") delete <name>
  $(basename "$0") who [recent_sec]     # default 2
  $(basename "$0") config get
  $(basename "$0") config set key=value [key=value] ...
  $(basename "$0") list                   # list identities in the gallery
  $(basename "$0") identities             # alias of 'list'

Env:
  FACE_HTTP   default: $FACE_HTTP

Notes:
  - selfie: saves directly to gallery/<name>/aligned and embeds immediately.
  - upload: copies to gallery/<name>/raw, then refreshes (align+embed) right away.
  - delete: removes gallery/<name>, rebuilds embeddings, refreshes in-memory means.
EOF
}

get_config() {
  post_json '{"get":true}' '/config' | jq '.config'
}

restore_blur() {
  # $1: "true" | "false"
  local want="${1:-true}"
  local v="false"; [ "$want" = "true" ] && v="true"
  post_json "{\"set\":{\"blur\":$v}}" '/config' >/dev/null || true
}

wait_single_face() {
  # $1: timeout seconds (default 15)
  local timeout="${1:-15}"
  local poll_ms=200
  local tries=$(( (timeout * 1000) / poll_ms ))
  echo "[INFO] Waiting for a single face (timeout: ${timeout}s)…"
  for ((i=0; i<tries; i++)); do
    local resp faces
    resp="$(post_json_soft '{"recent_sec":1}' '/who')"
    if [ -n "${resp:-}" ]; then
      faces="$(echo "$resp" | jq -r '((.now // []) | length) + (.unknown_now // 0)' 2>/dev/null || echo 0)"
      if [ "$faces" = "1" ]; then
        echo "[INFO] OK, current face(s):"
        pretty "$resp"
        return 0
      fi
    fi
    sleep "$(awk -v ms=$poll_ms 'BEGIN{print ms/1000.0}')"
  done
  echo "[ERR] Timeout waiting for exactly one face."
  return 1
}

cmd="${1-}"
[ -z "$cmd" ] && { usage; exit 1; }

case "$cmd" in
  selfie)
    name="${2-}"
    [ -z "$name" ] && { echo "[ERR] selfie requires <name>"; usage; exit 1; }
    # health check
    post_json '{}' '/ping' >/dev/null 2>&1 || { echo "[ERR] Cannot reach $FACE_HTTP/ping — is run.py running with --http-host/--http-port?"; exit 1; }

    echo "[INFO] Fetching current config…"
    cfg="$(get_config)"
    orig_blur="$(echo "$cfg" | jq -r '(.blur // true)')"

    echo "[INFO] Temporarily disabling blur to capture selfie…"
    post_json '{"set":{"blur":false}}' '/config' >/dev/null || true
    trap 'echo "[INFO] Restoring blur='"$orig_blur"'…"; restore_blur '"$orig_blur"'' EXIT

    wait_single_face 15

    echo "[INFO] Requesting selfie save (aligned) for '\''"$name"'\''…"
    resp="$(post_json "$(jq -n --arg id "$name" '{id:$id}')" '/selfie')"
    pretty "$resp"
    ok="$(echo "$resp" | jq -r '.ok // false')"
    [ "$ok" = "true" ] || { echo "[ERR] Selfie failed."; exit 2; }
    echo "[OK] Selfie for '$name' saved to aligned and embedded."
    ;;

  upload)
    name="${2-}"
    img="${3-}"
    [ -z "$name" ] || [ -z "$img" ] && { echo "[ERR] upload requires <name> <abs_path>"; usage; exit 1; }
    [ -f "$img" ] || { echo "[ERR] file not found: $img"; exit 1; }
    post_json '{}' '/ping' >/dev/null 2>&1 || { echo "[ERR] Cannot reach $FACE_HTTP/ping — is run.py running?"; exit 1; }

    echo "[INFO] Uploading image to raw for '$name'…"
    resp_add="$(post_json "$(jq -n --arg id "$name" --arg image_path "$img" '{id:$id, image_path:$image_path}')" '/gallery/add_raw')"
    pretty "$resp_add"
    ok="$(echo "$resp_add" | jq -r '.ok // false')"
    [ "$ok" = "true" ] || { echo "[ERR] Upload failed."; exit 2; }

    echo "[INFO] Align+embed refresh after upload…"
    resp_ref="$(post_json '{}' '/gallery/refresh')"
    pretty "$resp_ref"
    echo "[OK] Raw uploaded, aligned cropped and embeddings updated."
    ;;

  delete)
    name="${2-}"
    [ -z "$name" ] && { echo "[ERR] delete requires <name>"; usage; exit 1; }
    post_json '{}' '/ping' >/dev/null 2>&1 || { echo "[ERR] Cannot reach $FACE_HTTP/ping — is run.py running?"; exit 1; }

    echo "[INFO] Deleting identity '$name' and rebuilding embeddings…"
    resp_del="$(post_json "$(jq -n --arg id "$name" '{id:$id}')" '/gallery/delete')"
    pretty "$resp_del"
    ok="$(echo "$resp_del" | jq -r '.ok // false')"
    [ "$ok" = "true" ] || { echo "[ERR] Delete failed."; exit 2; }
    echo "[OK] Deleted '$name' and refreshed embeddings."
    ;;

  who)
    # quick ping so we don’t silently fail
    post_json '{}' '/ping' >/dev/null 2>&1 || { echo "[ERR] Cannot reach $FACE_HTTP/ping — is run.py running?"; exit 1; }
    sec="${2:-2}"
    # if non-numeric, fall back to 2
    if ! printf '%s' "$sec" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
      sec=2
    fi
    body=$(printf '{"recent_sec":%s}' "$sec")
    resp="$(post_json_soft "$body" '/who')"
    if [ -z "${resp:-}" ]; then
      echo "[ERR] Empty response from /who"
      exit 1
    fi
    pretty "$resp"
    ;;

  list|identities)
    post_json '{}' '/gallery/identities' | pretty
    ;;

  config)
    sub="${2-}"
    case "$sub" in
      get)
        get_config | pretty
        ;;
      set)
        shift 2
        [ $# -lt 1 ] && { echo "[ERR] config set needs key=value ..."; exit 1; }
        payload='{ "set": {'
        for kv in "$@"; do
          key="${kv%%=*}"; val="${kv#*=}"
          if printf '%s' "$val" | grep -Eq '^(true|false|[0-9]+([.][0-9]+)?)$'; then
            payload="$payload\"$key\":$val,"
          else
            esc_val="$(printf '%s' "$val" | jq -R '.')"
            payload="$payload\"$key\":$esc_val,"
          fi
        done
        payload="${payload%,} } }"
        post_json "$payload" '/config' | pretty
        ;;
      *) usage; exit 1 ;;
    esac
    ;;

  *)
    usage; exit 1 ;;
esac
