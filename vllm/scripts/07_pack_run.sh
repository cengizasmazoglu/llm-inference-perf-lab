#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <RUN_DIR>"
  exit 1
fi

RUN_DIR="${1%/}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: Run directory does not exist:"
  echo "$RUN_DIR"
  exit 1
fi

RUN_ID="$(basename "$RUN_DIR")"
PARENT_DIR="$(dirname "$RUN_DIR")"

ARCHIVE="${RUN_ID}.tar.gz"
CHECKSUM="${ARCHIVE}.sha256"

tar \
  -C "$PARENT_DIR" \
  -czf "$ARCHIVE" \
  "$RUN_ID"

sha256sum "$ARCHIVE" \
  | sed "s#  .*#  ${ARCHIVE}#" \
  > "$CHECKSUM"

echo
echo "Created:"
echo "$ARCHIVE"
echo "$CHECKSUM"
echo
echo "Verify later with:"
echo "sha256sum -c $CHECKSUM"