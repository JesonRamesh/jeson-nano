#!/usr/bin/env bash
# Make the two GLBs from a raw Scaniverse export (spec §6.2 step 4). Needs Node 22 (npx).
#
#   tools/prepare_scan.sh maps/<name>
#
# reads   maps/<name>/raw/scan_raw.glb   (gitignored)
# writes  maps/<name>/scan.glb   web: meshopt + webp textures <= 2048 px   (D's R3F view; must be < 50 MB)
#         maps/<name>/sim.glb    analysis: simplified, UNcompressed (trimesh can't read meshopt)
#
# Env overrides: RAW=<path>  TEX=2048  RATIO=0.25  ERROR=0.001  GLTF_TRANSFORM_VERSION=4
set -euo pipefail

MAP_DIR="${1:?usage: $0 maps/<name>}"
RAW="${RAW:-$MAP_DIR/raw/scan_raw.glb}"
TEX="${TEX:-2048}"
RATIO="${RATIO:-0.25}"
ERROR="${ERROR:-0.001}"
GT="npx --yes @gltf-transform/cli@${GLTF_TRANSFORM_VERSION:-4}"

[[ -f "$RAW" ]] || { echo "missing $RAW (export the Scaniverse GLB there)" >&2; exit 1; }

# Web: an exact copy of the original when it fits the 50 MB limit (spec §1.2.6); otherwise
# compress (meshopt + webp textures). COMPRESS=1 forces compression.
raw_size=$(wc -c < "$RAW")
if [[ "${COMPRESS:-0}" != "1" ]] && (( raw_size <= 50000000 )); then
  echo "== web: $MAP_DIR/scan.glb = exact copy of the original ($((raw_size / 1000000)) MB)"
  cp "$RAW" "$MAP_DIR/scan.glb"
else
  echo "== web: $MAP_DIR/scan.glb (compressed: original is $((raw_size / 1000000)) MB)"
  $GT optimize "$RAW" "$MAP_DIR/scan.glb" --compress meshopt --texture-compress webp --texture-size "$TEX"
fi

echo "== sim: $MAP_DIR/sim.glb"
$GT simplify "$RAW" "$MAP_DIR/sim.glb" --ratio "$RATIO" --error "$ERROR"
$GT resize "$MAP_DIR/sim.glb" "$MAP_DIR/sim.glb" --width "$TEX" --height "$TEX"

ls -lh "$RAW" "$MAP_DIR/scan.glb" "$MAP_DIR/sim.glb"
size=$(wc -c < "$MAP_DIR/scan.glb")
if (( size > 50000000 )); then
  echo "WARNING: scan.glb is > 50 MB. Retry with TEX=1024, or RATIO=0.5 + simplify before optimize (spec §6.4)." >&2
fi
echo "next: python tools/scan_to_grid.py $MAP_DIR/sim.glb --entrance 0 0"
