#!/usr/bin/env bash
# =====================================================================================================================
# make_s9np_videos.sh  --  encode the s9_np demo frames (3 cameras) into s9_np_model/.
#
# Prereq: 3-camera rig (FollowCam / upcam / sidecam), each with a FrameSequenceRecorder whose OutputDirectory is under
#         Recordings/frame_sequence/s9_np (AppendCameraName ON). This script finds EVERY subfolder holding frame_*.jpg
#         (any nesting depth, e.g. s9_np/fixed_target/FollowCam) and encodes one mp4 per camera into s9_np_model/.
#
# Usage:  bash s9_np_model/make_s9np_videos.sh          # 30 fps, default frame root
#         FPS=30 ROOT="Recordings/frame_sequence/s9_np" bash s9_np_model/make_s9np_videos.sh
# =====================================================================================================================
set -eu   #? no pipefail: find|sort closes pipes early (SIGPIPE)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   #? write next to this script wherever it lives
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
ROOT="${ROOT:-$REPO/Recordings/frame_sequence/s9_np}"
FPS="${FPS:-30}"
OUTDIR="${OUTDIR:-$HERE}"

if [ ! -d "$ROOT" ]; then
  echo "ERROR: frame root not found: $ROOT"; exit 1
fi
mapfile -t DIRS < <(find "$ROOT" -type f -name 'frame_*.jpg' -printf '%h\n' | sort -u)
if [ "${#DIRS[@]}" -eq 0 ]; then
  echo "ERROR: no frame_*.jpg under $ROOT -- did you record this session?"; exit 1
fi

for d in "${DIRS[@]}"; do
  cam=$(basename "$d")
  nf=$(ls "$d"/frame_*.jpg 2>/dev/null | wc -l)
  out="$OUTDIR/s9np_${cam}.mp4"
  echo "[$cam] $nf frames -> $out"
  ffmpeg -y -framerate "$FPS" -i "$d/frame_%05d.jpg" \
    -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "$out" </dev/null >/dev/null 2>&1 \
    && echo "   done ($(du -h "$out" | cut -f1))" || echo "   FAILED (run ffmpeg on $d manually to see why)"
done
echo "all videos -> $OUTDIR"
ls -1 "$OUTDIR"/s9np_*.mp4 2>/dev/null