#!/usr/bin/env bash
# Encode the arm_6dof demo frames (3 cameras) into this folder. Frames come from repo-root
# Recordings/frame_sequence/arm_6dof/<cameraName>/ (set each recorder's OutputDirectory to that).
set -eu   #? no pipefail (find|sort closes pipes early)
REPO="/home/nyctophilia/UnityProjects/RL_Scoutsan"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$REPO/Recordings/frame_sequence/arm_6dof}"
FPS="${FPS:-30}"
[ -d "$ROOT" ] || { echo "ERROR: no frames at $ROOT (record first)"; exit 1; }
mapfile -t DIRS < <(find "$ROOT" -type f -name 'frame_*.jpg' -printf '%h\n' | sort -u)
[ "${#DIRS[@]}" -gt 0 ] || { echo "ERROR: no frame_*.jpg under $ROOT"; exit 1; }
for d in "${DIRS[@]}"; do
  cam=$(basename "$d"); out="$HERE/arm6dof_${cam}.mp4"
  n=$(ls "$d"/frame_*.jpg 2>/dev/null | wc -l)
  echo "[$cam] $n frames -> $out"
  ffmpeg -y -framerate "$FPS" -i "$d/frame_%05d.jpg" -c:v libx264 -preset slow -crf 18 \
    -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "$out" </dev/null >/dev/null 2>&1 \
    && echo "   done ($(du -h "$out"|cut -f1))" || echo "   FAILED"
done
ls -1 "$HERE"/arm6dof_*.mp4 2>/dev/null
