import cv2
import mediapipe as mp
import os
import json
import numpy as np
import logging
from tensorflow.keras.models import load_model
from .utils import get_local_path

# Constants
MODEL_SUBPATH = "models/2_shotput/shotput_stage4.keras"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(video_path):
    """Extract keypoints from video."""
    keypoints = []
    cap = cv2.VideoCapture(video_path)
    
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
            keypoints.append({
                "push_leg": {
                    "hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
                    "knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
                    "ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                },
                "torso": {
                    "right_shoulder": [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                    "left_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                },
                "arms": {
                    "right_elbow": [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                    "right_wrist": [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                }
            })

    cap.release()
    return keypoints

def extract_features(keypoints):
    """Extract features from keypoints."""
    features = []
    for i in range(1, len(keypoints)):
        frame_features = {}

        prev_hip = keypoints[i - 1]["push_leg"]["hip"]
        curr_hip = keypoints[i]["push_leg"]["hip"]
        frame_features["push_leg_velocity_x"] = curr_hip[0] - prev_hip[0]
        frame_features["push_leg_velocity_y"] = curr_hip[1] - prev_hip[1]

        knee = keypoints[i]["push_leg"]["knee"]
        ankle = keypoints[i]["push_leg"]["ankle"]
        frame_features["knee_ankle_distance"] = np.linalg.norm(np.array(knee) - np.array(ankle))

        right_shoulder = keypoints[i]["torso"]["right_shoulder"]
        left_shoulder = keypoints[i]["torso"]["left_shoulder"]
        frame_features["shoulder_angle"] = np.arctan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        )

        right_elbow = keypoints[i]["arms"]["right_elbow"]
        right_wrist = keypoints[i]["arms"]["right_wrist"]
        frame_features["right_arm_angle"] = np.arctan2(
            right_wrist[1] - right_elbow[1],
            right_wrist[0] - right_elbow[0]
        )

        features.append(list(frame_features.values()))

    return np.array(features)

def classify_prediction(predictions):
    """Map prediction to class labels."""
    class_map = {0: 0, 1: 0.5, 2: 1}
    predicted_class = np.argmax(predictions)
    return class_map[predicted_class]

def main(video_path):
    """Main function to process video and return results."""
    try:
        local_path = get_local_path(video_path)
        keypoints = extract_keypoints(local_path)
        
        if not keypoints:
            return {
                "video": video_path,
                "error": "No keypoints extracted"
            }

        features = extract_features(keypoints)
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
            "classified_score": float(classified_score)
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
    test_video = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_videos", "stage4.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))
