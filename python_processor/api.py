# api.py
from flask import Flask, request, jsonify
import logging
import os
import tempfile
import requests
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
# If you keep them separate:
from .shotput_stage1_script import main as run_stage1
from .shotput_stage2_script import main as run_stage2
# etc.

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### UTILS ###

def download_video(url: str) -> str:
    """Download remote URL to a local temp file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
        local_path = tmp_file.name
        logger.info(f"Downloaded video to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download video: {str(e)}")
        raise

def cut_video_opencv(full_video_path: str, start_sec: float, end_sec: float) -> str:
    """Use OpenCV to cut a [start_sec, end_sec] subclip."""
    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    subclip_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    subclip_path = subclip_file.name
    subclip_file.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(subclip_path, fourcc, fps, (frame_width, frame_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    logger.info(f"Subclip saved to: {subclip_path}")
    return subclip_path

### STAGE ANALYSIS ###

def analyze_stage(video_path: str, stage: Dict) -> Dict:
    """
    stage = {"name": "stage1", "startTime": 0.0, "endTime": 3.5}
    Actually calls your shotput_stageN_script based on stage name
    """
    name_lower = stage["name"].lower()
    start_sec = float(stage["startTime"])
    end_sec = float(stage["endTime"])

    # 1) Cut subclip
    subclip_path = cut_video_opencv(video_path, start_sec, end_sec)
    try:
        # 2) Decide which stage script to call
        if name_lower == "stage1":
            return run_stage1(subclip_path)
        elif name_lower == "stage2":
            return run_stage2(subclip_path)
        # ...
        else:
            return {
                "error": f"Unknown stage: {stage['name']}"
            }
    finally:
        if os.path.exists(subclip_path):
            os.remove(subclip_path)

### ENDPOINTS ###

@app.route("/analyze", methods=["POST"])
def analyze_video():
    """Main entry: receives JSON with videoUrl, exercise, stages, etc."""
    try:
        data = request.json
        video_url = data["videoUrl"]
        stages = data.get("stages", [])
        # optional
        exercise = data.get("exercise", "shotput")
        processing_id = data.get("processingId", "unknown")

        # 1) Download the entire video
        local_video_path = download_video(video_url)

        try:
            result = {
                "processingId": processing_id,
                "exercise": exercise,
                "stageAnalysis": {},
                "metrics": {"overall_score": 0.0, "confidence": 0.0}
            }

            total_score = 0.0
            stage_count = 0
            total_confidence = 0.0

            # 2) For each stage, run analysis
            for stage in stages:
                stage_name = stage["name"]
                stage_result = analyze_stage(local_video_path, stage)
                result["stageAnalysis"][stage_name] = stage_result

                # If the stage_result has "predicted_score" or "score"
                maybe_score = stage_result.get("predicted_score") or stage_result.get("score")
                maybe_confidence = stage_result.get("confidence", 0.0)
                
                if maybe_score is not None:
                    total_score += float(maybe_score)
                    total_confidence += float(maybe_confidence)
                    stage_count += 1

            # 3) Compute overall average
            if stage_count > 0:
                result["metrics"]["overall_score"] = total_score / stage_count
                result["metrics"]["confidence"] = total_confidence / stage_count

            return jsonify(result)
        finally:
            # cleanup the big video
            if os.path.exists(local_video_path):
                os.remove(local_video_path)

    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}")
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.environ.get("VERSION", "dev")
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
