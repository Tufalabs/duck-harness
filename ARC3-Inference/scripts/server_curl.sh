#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <api-key-file> [curl args...]" >&2
  exit 2
fi

api_key_file="$1"
shift

if [[ -s "${api_key_file}" ]]; then
  curl_config="${api_key_file}.curlrc"
  api_key="$(head -n 1 "${api_key_file}")"
  umask 077
  printf 'header = "Authorization: Bearer %s"\n' "${api_key}" > "${curl_config}"
  chmod 600 "${curl_config}" 2>/dev/null || true
  exec curl --config "${curl_config}" "$@"
fi

exec curl "$@"
