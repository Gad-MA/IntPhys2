#!/usr/bin/env bash
# group_viz.sh — group PSI context + prediction GIFs into per-window subfolders
#
# Usage:
#   bash group_viz.sh <viz_dir> [frame_step]
#
# Arguments:
#   viz_dir     Root visualization directory (contains per-video subfolders).
#   frame_step  Raw-frame gap between the last context frame and the first
#               predicted frame.  Must match viz_frame_step in the config.
#               Default: 10.
#
# Structure BEFORE:
#   <viz_dir>/<video_name>/
#     context_frame00030_ctx02.gif
#     pred_frame00040_ctx02_L1_0.0831.gif
#     context_frame00050_ctx02.gif
#     pred_frame00060_ctx02_L1_0.0912.gif
#     ...
#
# Structure AFTER:
#   <viz_dir>/<video_name>/
#     window_f00040_ctx02/
#       context_frame00030_ctx02.gif
#       pred_frame00040_ctx02_L1_0.0831.gif
#     window_f00060_ctx02/
#       context_frame00050_ctx02.gif
#       pred_frame00060_ctx02_L1_0.0912.gif
#     ...
#
# The subfolder is named after the FIRST PREDICTED FRAME index and context
# length, making it easy to sort windows chronologically within a video.
#
# Safe to re-run: already-grouped files live in subdirs and are not touched.

set -euo pipefail

VIZ_DIR="${1:?Error: viz_dir argument required.  Usage: $0 <viz_dir> [frame_step]}"
FRAME_STEP="${2:-10}"

if [[ ! -d "$VIZ_DIR" ]]; then
    echo "Error: '$VIZ_DIR' is not a directory." >&2
    exit 1
fi

n_matched=0
n_missing_ctx=0
n_videos=0

for video_dir in "$VIZ_DIR"/*/; do
    [[ -d "$video_dir" ]] || continue
    video_name="$(basename "$video_dir")"
    (( n_videos++ )) || true

    # Only look at prediction GIFs directly in this folder (not in subdirs).
    shopt -s nullglob
    pred_gifs=("$video_dir"pred_frame*_ctx*.gif)
    shopt -u nullglob

    if [[ ${#pred_gifs[@]} -eq 0 ]]; then
        echo "  [$video_name] no ungrouped prediction GIFs found — skipping."
        continue
    fi

    echo "  [$video_name] found ${#pred_gifs[@]} prediction GIF(s)..."

    for pred_gif in "${pred_gifs[@]}"; do
        filename="$(basename "$pred_gif")"

        # Parse pred_frame number, ctx number, and L1 score.
        # Filename pattern: pred_frame{NNNNN}_ctx{KK}_L1_{score}.gif
        if [[ "$filename" =~ pred_frame([0-9]+)_ctx([0-9]+)_L1_([0-9]+\.[0-9]+) ]]; then
            pred_frame="${BASH_REMATCH[1]}"   # e.g. "00040"
            ctx="${BASH_REMATCH[2]}"          # e.g. "02"
            l1="${BASH_REMATCH[3]}"           # e.g. "0.0831"
        else
            echo "    WARNING: cannot parse filename '$filename' — skipping." >&2
            continue
        fi

        # Compute the context GIF's frame number.
        # raw_ctx_frame = raw_tgt_start - frame_step
        # Use 10# prefix to force base-10 (avoids octal for zero-padded numbers).
        ctx_frame_num=$(( 10#$pred_frame - FRAME_STEP ))
        ctx_frame_str="$(printf '%05d' "$ctx_frame_num")"

        ctx_gif="${video_dir}context_frame${ctx_frame_str}_ctx${ctx}.gif"

        if [[ ! -f "$ctx_gif" ]]; then
            echo "    WARNING: no context GIF for '$filename'" >&2
            echo "             expected: context_frame${ctx_frame_str}_ctx${ctx}.gif" >&2
            (( n_missing_ctx++ )) || true
            continue
        fi

        # Create the window subfolder and move both GIFs into it.
        window_dir="${video_dir}window_f${pred_frame}_ctx${ctx}_L1_${l1}"
        mkdir -p "$window_dir"
        mv "$pred_gif" "$window_dir/"
        mv "$ctx_gif"  "$window_dir/"

        (( n_matched++ )) || true
    done
done

echo ""
echo "Done."
echo "  Videos scanned : $n_videos"
echo "  Windows grouped: $n_matched"
if (( n_missing_ctx > 0 )); then
    echo "  Missing context: $n_missing_ctx  (prediction GIFs left ungrouped)"
fi
