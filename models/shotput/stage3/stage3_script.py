import cv2
import mediapipe as mp
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MODEL_PATH = "/Users/cezar/Desktop/Team Project/AI/shotput/stage3/shotput_stage3.keras"
OUTPUT_JSON = "stage3_results.json"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Helper function to calculate velocity
def calculate_velocity(coord1, coord2, fps=30):
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    distance = np.sqrt(dx**2 + dy**2)
    return distance * fps

# Function to extract features from video keypoints
def extract_features(keypoints):
    features = []
    for i in range(1, len(keypoints)):
        prev_left_ankle = keypoints[i - 1]["left_leg"]["ankle"]
        curr_left_ankle = keypoints[i]["left_leg"]["ankle"]
        left_velocity = calculate_velocity(prev_left_ankle, curr_left_ankle)

        right_leg = keypoints[i]["right_leg"]
        knee_angle = calculate_angle(right_leg["hip"], right_leg["knee"], right_leg["ankle"])

        left_ankle = keypoints[i]["left_leg"]["ankle"]
        right_ankle = keypoints[i]["right_leg"]["ankle"]
        ankle_distance = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))

        features.append([left_velocity, knee_angle, ankle_distance])
    return np.array(features)

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
            left_leg = {
                "hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                "knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                "ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
            }
            right_leg = {
                "hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
                "knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
                "ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y],
            }
            keypoints.append({"left_leg": left_leg, "right_leg": right_leg})

    cap.release()
    return keypoints

def classify_score(predictions):
    """Map prediction to class labels."""
    score_mapping = {0: 0, 1: 0.5, 2: 1}
    return score_mapping[np.argmax(predictions)]

def main(video_path):
    print("Extracting keypoints from video...")
    keypoints = extract_keypoints(video_path)

    if not keypoints:
        print(f"No keypoints extracted. Check the video file: {video_path}")
        return

    print("Extracting features from keypoints...")
    features = extract_features(keypoints)

    if features.size == 0:
        print("No features extracted. Ensure the keypoints extraction is working correctly.")
        return

    print("Reshaping features for model input...")
    features = features.reshape(1, features.shape[0], features.shape[1])

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Predicting score...")
    predictions = model.predict(features)
    classified_score = classify_score(predictions)

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
    VIDEO_PATH = "/Users/cezar/Desktop/Team Project/AI/shotput/stage3/videos/0_user3.mp4"  # Replace with actual path
    main(VIDEO_PATH)
