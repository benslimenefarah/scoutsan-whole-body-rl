#!/usr/bin/env bash
# =====================================================================================================================
# make_video.sh  --  encode the base-model demo frames (captured by FrameSequenceRecorder) into base_model/.
#
# Prereq: in Assets/Scenes/base.unity, on the recording camera (GameObject named "base_model") add both
#         FrameSequenceRecorder (RecordOnStart=ON, 1920x1080, 30 fps) and EpisodeFramingCamera (wire target +
#         baseRobot), press PLAY, run eval_base_flu.py at ARM_TIME_SCALE=1, then stop Play.
#         Frames are written to  Recordings/frame_sequence/base_model/frame_XXXXX.jpg .
#
# Usage:  bash base_model/make_video.sh            # uses the default frame dir + 30 fps
#         FRAME_DIR="Recordings/frame_sequence/Main Camera" FPS=30 bash base_model/make_video.sh
# =====================================================================================================================
set -eu   #? no pipefail: `find | sort | head` legitimately closes the pipe early (SIGPIPE)
REPO="/home/nyctophilia/UnityProjects/RL_Scoutsan"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   #? write next to this script wherever it lives
ROOT="${FRAME_DIR:-$REPO/Recordings/frame_sequence/base_model}"
FPS="${FPS:-30}"
OUT="${OUT:-$HERE/base_model_demo.mp4}"

if [ ! -d "$ROOT" ]; then
  echo "ERROR: frame dir not found: $ROOT"
  echo "  Attach FrameSequenceRecorder to the recording camera in base.unity, press PLAY, run eval_base_flu.py, then retry."
  exit 1
fi

# FrameSequenceRecorder appends the camera GameObject name (AppendCameraName) as a subfolder,
# so frames may sit in ROOT itself OR one level down. Find the dir that actually holds frame_*.jpg
# (pick the one with the most-recent frame if several).
if ls "$ROOT"/frame_*.jpg >/dev/null 2>&1; then
  FRAME_DIR="$ROOT"
else
  FRAME_DIR=$(find "$ROOT" -type f -name 'frame_*.jpg' -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
fi
if [ -z "${FRAME_DIR:-}" ] || ! ls "$FRAME_DIR"/frame_*.jpg >/dev/null 2>&1; then
  echo "ERROR: no frame_*.jpg found under $ROOT (or its subfolders)."
  echo "  Did you record this session? Frames go to Recordings/frame_sequence/<cameraName>/ ."
  exit 1
fi
NF=$(ls "$FRAME_DIR"/frame_*.jpg | wc -l)
echo "encoding $NF frames from: $FRAME_DIR  @ ${FPS} fps  ->  $OUT"

# yuv420p + even dims -> plays everywhere; crf 18 = visually lossless at 1080p
ffmpeg -y -framerate "$FPS" -i "$FRAME_DIR/frame_%05d.jpg" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "$OUT"
echo "done -> $OUT"
