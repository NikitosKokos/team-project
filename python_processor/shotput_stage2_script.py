import cv2
import mediapipe as mp
import os
import json
import numpy as np
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .utils import get_local_path

# Constants
MODEL_SUBPATH = "models/2_shotput/shotput_stage2.keras"
MAX_SEQUENCE_LENGTH = 76

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def calculate_distance(a, b):
    """Calculate distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def classify_score(prediction):
    """Classify prediction into 0, 0.5, or 1."""
    if prediction >= 0.90:
        return 1.0
    elif prediction >= 0.75:
        return 0.5
    else:
        return 0.0

def extract_features_from_video(video_path):
    """Extract features from video."""
    angles = []
    distances = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

            angle = calculate_angle(right_hip, right_knee, right_ankle)
            distance = calculate_distance(right_hip, right_knee)

            angles.append(angle)
            distances.append(distance)

    cap.release()
    return np.stack([angles, distances], axis=1) if angles else np.array([])

def main(video_path):
    """Main function to process video and return results."""
    try:
        # Get local video path (downloading if necessary)
        local_path = get_local_path(video_path)
        
        # Extract features
        features = extract_features_from_video(local_path)
        if features.size == 0:
            return {
                "video": video_path,
                "error": "No features extracted"
            }

        # Pad sequences
        if features.shape[0] < MAX_SEQUENCE_LENGTH:
            padding = np.zeros((MAX_SEQUENCE_LENGTH - features.shape[0], 2))
            features_padded = np.vstack((features, padding))
        else:
            features_padded = features[:MAX_SEQUENCE_LENGTH, :]

        features_padded = np.expand_dims(features_padded, axis=0)

        # Load model and predict
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, MODEL_SUBPATH)
        model = load_model(model_path)
        prediction = model.predict(features_padded)[0][0]
        classified_score = classify_score(prediction)

        return {
            "video": video_path,
            "predicted_score": float(prediction),
            "classified_score": float(classified_score)
        }

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return {
            "video": video_path,
            "error": str(e)
        }
    finally:
        # Clean up temp file if it was created
        if video_path.startswith(("http://", "https://")) and os.path.exists(local_path):
            try:
                os.unlink(local_path)
            except Exception as e:
                logging.error(f"Error cleaning up temp file: {str(e)}")

if __name__ == "__main__":
    test_video = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_videos", "stage2.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))