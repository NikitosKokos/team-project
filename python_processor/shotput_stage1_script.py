import cv2
import mediapipe as mp
import os
import json
import numpy as np
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
from .utils import get_local_path

# Constants
MODEL_SUBPATH = "models/2_shotput/shotput_stage1.keras"
MAX_SEQUENCE_LENGTH = 170

@register_keras_serializable()
def weighted_mse(y_true, y_pred):
    """Weighted Mean Squared Error to prioritize true negatives."""
    weights = K.switch(y_true < 0.70, 2.0, 1.0)
    return K.mean(weights * K.square(y_true - y_pred))

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def extract_keypoints(video_path):
    """Extract keypoints and angles from a video."""
    keypoints = []
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
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            angle_left_leg = calculate_angle(left_hip, left_knee, left_ankle)
            keypoints.append(angle_left_leg)

    cap.release()
    return keypoints

def classify_score(prediction):
    """Classify the prediction into 0, 0.5, or 1 based on thresholds."""
    if prediction >= 0.85:
        return 1.0
    elif prediction >= 0.70:
        return 0.5
    else:
        return 0.0

def main(video_path):
    """Main function to process video and return results."""
    try:
        # Get local video path (downloading if necessary)
        local_path = get_local_path(video_path)
        
        # Extract Keypoints
        keypoints = extract_keypoints(local_path)
        
        # Build absolute path to model
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, MODEL_SUBPATH)

        # Pad Sequence
        keypoints_padded = pad_sequences([keypoints], maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')
        keypoints_padded = keypoints_padded[..., np.newaxis]

        # Load Model and Predict
        model = load_model(model_path, custom_objects={"weighted_mse": weighted_mse})
        prediction = model.predict(keypoints_padded)[0][0]
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
    test_video = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_videos", "stage1.mp4")
    result = main(test_video)
    print(json.dumps(result, indent=2))
