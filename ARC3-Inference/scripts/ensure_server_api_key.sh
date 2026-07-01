#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <api-key-file>" >&2
  exit 2
fi

api_key_file="$1"
api_key="${SERVER_API_KEY:-}"

if [[ -z "${api_key}" && -f "${api_key_file}" ]]; then
  api_key="$(head -n 1 "${api_key_file}")"
fi

if [[ -z "${api_key}" ]]; then
  api_key="$("${PYTHON:-python3}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
fi

mkdir -p "$(dirname "${api_key_file}")"
umask 077
printf '%s\n' "${api_key}" > "${api_key_file}"
chmod 600 "${api_key_file}" 2>/dev/null || true
printf '%s\n' "${api_key}"
