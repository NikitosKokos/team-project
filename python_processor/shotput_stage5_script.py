import cv2
import mediapipe as mp
import os
import json
import numpy as np
import logging
from tensorflow.keras.models import load_model
from .utils import get_local_path

# Constants
MODEL_SUBPATH = "models/2_shotput/shotput_stage5.keras"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_distance(point1, point2):
    """Calculate distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def extract_keypoints(video_path):
    """Extract keypoints and detect release frame."""
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    distances = []
    release_frame = None
    
    if not cap.isOpened():
        raise Exception("Could not open video file")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            neck = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

            distance = calculate_distance(wrist, neck)
            distances.append(distance)
            keypoints.append({"wrist": wrist, "neck": neck, "shoulder": shoulder})

    # Detect release frame
    for i in range(1, len(distances)):
        if distances[i] > distances[i - 1] * 1.5:
            release_frame = i
            break

    cap.release()
    return keypoints, release_frame

def extract_features(keypoints, release_frame):
    """Extract features from keypoints."""
    if not release_frame:
        return np.array([])

    features = []
    for i in range(len(keypoints)):
        frame_features = {}
        wrist = keypoints[i]["wrist"]
        neck = keypoints[i]["neck"]
        frame_features["shot_neck_distance"] = calculate_distance(wrist, neck)

        if i == release_frame and i > 0:
            prev_wrist = keypoints[i - 1]["wrist"]
            release_angle = np.degrees(np.arctan2(
                wrist[1] - prev_wrist[1], wrist[0] - prev_wrist[0]
            ))
            frame_features["release_angle"] = release_angle
        else:
            frame_features["release_angle"] = 0

        features.append(list(frame_features.values()))

    return np.array(features, dtype=np.float32)

def classify_prediction(predictions):
    """Map prediction to class labels."""
    class_map = {0: 0, 1: 0.5, 2: 1}
    predicted_class = np.argmax(predictions)
    return class_map[predicted_class]

def main(video_path):
    """Main function to process video and return results."""
    try:
        local_path = get_local_path(video_path)
        keypoints, release_frame = extract_keypoints(local_path)
        
        if not keypoints:
            return {
                "video": video_path,
                "error": "No keypoints extracted"
            }

        features = extract_features(keypoints, release_frame)
        if features.size == 0:
            return {
                "video": video_path,
                "error": "No features extracted"
            }

        features = features.reshape(1, features.shape[0], features.shape[1])
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, MODEL_SUBPATH)
        model = load_model(model_path)
        
        predictions = model.predict(features)
        classified_score = classify_prediction(predictions)

        return {
            "video": video_path,
            "predicted_scores": predictions.tolist(),
            "classified_score": float(classified_score),
            "release_frame": release_frame
        }

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return {
            "video": video_path,
            "error": str(e)
        }
    finally:
        if video_path.startswith(("http://", "https://")) and os.path.exists(local_path):
            try:
                os.unlink(local_path)
            except Exception as e:
                logging.error(f"Error cleaning up temp file: {str(e)}")

if __name__ == "__main__":
    test_video = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_videos", "stage5.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))
