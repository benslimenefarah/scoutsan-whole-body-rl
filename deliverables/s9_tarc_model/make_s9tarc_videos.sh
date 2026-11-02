#!/usr/bin/env bash
# Encode the s9_tarc demo frames (3 cameras) into this folder. Frames come from repo-root
# Recordings/frame_sequence/s9_tarc/<cameraName>/ (set each recorder's OutputDirectory to that).
set -eu   #? no pipefail (find|sort closes pipes early)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
ROOT="${ROOT:-$REPO/Recordings/frame_sequence/s9_tarc}"
FPS="${FPS:-30}"
[ -d "$ROOT" ] || { echo "ERROR: no frames at $ROOT (record first)"; exit 1; }
mapfile -t DIRS < <(find "$ROOT" -type f -name 'frame_*.jpg' -printf '%h\n' | sort -u)
[ "${#DIRS[@]}" -gt 0 ] || { echo "ERROR: no frame_*.jpg under $ROOT"; exit 1; }
for d in "${DIRS[@]}"; do
  cam=$(basename "$d"); out="$HERE/s9tarc_${cam}.mp4"
  n=$(ls "$d"/frame_*.jpg 2>/dev/null | wc -l)
  echo "[$cam] $n frames -> $out"
  ffmpeg -y -framerate "$FPS" -i "$d/frame_%05d.jpg" -c:v libx264 -preset slow -crf 18 \
    -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "$out" </dev/null >/dev/null 2>&1 \
    && echo "   done ($(du -h "$out"|cut -f1))" || echo "   FAILED"
done
ls -1 "$HERE"/s9tarc_*.mp4 2>/dev/null
