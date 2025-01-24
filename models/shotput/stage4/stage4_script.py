import cv2
import mediapipe as mp
import os
import json
import numpy as np
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = "/Users/cezar/Desktop/Team Project/AI/shotput/stage4/shotput_stage4.keras"
OUTPUT_JSON = "stage4_results.json"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract keypoints from a video
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []

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

# Function to extract features from keypoints
def extract_features(keypoints):
    features = []
    for i in range(1, len(keypoints)):
        frame_features = {}

        # Push leg extension features
        prev_hip = keypoints[i - 1]["push_leg"]["hip"]
        curr_hip = keypoints[i]["push_leg"]["hip"]
        frame_features["push_leg_velocity_x"] = curr_hip[0] - prev_hip[0]
        frame_features["push_leg_velocity_y"] = curr_hip[1] - prev_hip[1]

        knee = keypoints[i]["push_leg"]["knee"]
        ankle = keypoints[i]["push_leg"]["ankle"]
        frame_features["knee_ankle_distance"] = np.linalg.norm(np.array(knee) - np.array(ankle))

        # Torso rotation
        right_shoulder = keypoints[i]["torso"]["right_shoulder"]
        left_shoulder = keypoints[i]["torso"]["left_shoulder"]
        frame_features["shoulder_angle"] = np.arctan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        )

        # Right arm involvement
        right_elbow = keypoints[i]["arms"]["right_elbow"]
        right_wrist = keypoints[i]["arms"]["right_wrist"]
        frame_features["right_arm_angle"] = np.arctan2(
            right_wrist[1] - right_elbow[1],
            right_wrist[0] - right_elbow[0]
        )

        features.append(list(frame_features.values()))

    return np.array(features)

# Function to classify prediction
def classify_prediction(predictions):
    class_map = {0: 0, 1: 0.5, 2: 1}
    predicted_class = np.argmax(predictions)
    return class_map[predicted_class]

def main(video_path):
    print("Extracting keypoints from video...")
    keypoints = extract_keypoints(video_path)

    if not keypoints:
        print("No keypoints extracted. Check the video file.")
        return

    print("Extracting features from keypoints...")
    features = extract_features(keypoints)

    if features.size == 0:
        print("No features extracted. Ensure keypoints are processed correctly.")
        return

    print("Reshaping features for model input...")
    features = features.reshape(1, features.shape[0], features.shape[1])

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Predicting score...")
    predictions = model.predict(features)
    classified_score = classify_prediction(predictions)

    print("Saving results...")
    results = {
        "video": video_path,
        "predicted_scores": predictions.tolist(),
        "classified_score": classified_score
    }

    with open(OUTPUT_JSON, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    VIDEO_PATH = "/Users/cezar/Desktop/Team Project/AI/shotput/stage4/videos/1_user23.mp4"
    main(VIDEO_PATH)
