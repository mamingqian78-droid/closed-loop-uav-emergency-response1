#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/aiderv2"
mkdir -p "${DATA_DIR}"

if [[ -z "${AIDERV2_URL:-}" ]]; then
  echo "Please set AIDERV2_URL to the official AIDERv2 archive URL before running this script."
  echo "Example: AIDERV2_URL=https://.../aiderv2.zip bash src/perception/aiderv2_download.sh"
  exit 1
fi

TMP_ZIP="${DATA_DIR}/aiderv2.zip"
curl -L "${AIDERV2_URL}" -o "${TMP_ZIP}"
unzip -o "${TMP_ZIP}" -d "${DATA_DIR}"
echo "AIDERv2 downloaded to ${DATA_DIR}"
