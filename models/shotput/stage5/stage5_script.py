import cv2
import mediapipe as mp
import os
import json
import numpy as np
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = "/Users/cezar/Desktop/Team Project/AI/shotput/stage5/shotput_stage5.keras"
OUTPUT_JSON = "stage5_results.json"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Helper function to calculate distance
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Helper function to extract keypoints and detect release frame
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    distances = []
    release_frame = None

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

            # Compute wrist-to-neck distance
            distance = calculate_distance(wrist, neck)
            distances.append(distance)

            # Save keypoints for feature engineering
            keypoints.append({"wrist": wrist, "neck": neck, "shoulder": shoulder})

    # Detect release frame: sudden increase in wrist-to-neck distance
    for i in range(1, len(distances)):
        if distances[i] > distances[i - 1] * 1.5:  # Threshold for sudden increase
            release_frame = i
            break

    cap.release()
    return keypoints, release_frame

# Function to extract features from keypoints
def extract_features(keypoints, release_frame):
    features = []
    for i in range(len(keypoints)):
        frame_features = {}

        # Shot-neck proximity
        wrist = keypoints[i]["wrist"]
        neck = keypoints[i]["neck"]
        frame_features["shot_neck_distance"] = calculate_distance(wrist, neck)

        # Release angle (only at release frame)
        if i == release_frame and i > 0:
            prev_wrist = keypoints[i - 1]["wrist"]
            release_angle = np.degrees(np.arctan2(
                wrist[1] - prev_wrist[1], wrist[0] - prev_wrist[0]
            ))
            frame_features["release_angle"] = release_angle
        else:
            frame_features["release_angle"] = 0  # Default to 0 if not the release frame

        features.append(list(frame_features.values()))

    # Convert to numpy array and ensure all values are numeric
    return np.array(features, dtype=np.float32)


# Function to classify prediction
def classify_prediction(predictions):
    class_map = {0: 0, 1: 0.5, 2: 1}
    predicted_class = np.argmax(predictions)
    return class_map[predicted_class]

def main(video_path):
    print("Extracting keypoints from video...")
    keypoints, release_frame = extract_keypoints(video_path)

    if not keypoints:
        print("No keypoints extracted. Check the video file.")
        return

    print("Extracting features from keypoints...")
    features = extract_features(keypoints, release_frame)

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
        "classified_score": classified_score,
        "release_frame": release_frame
    }

    with open(OUTPUT_JSON, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    VIDEO_PATH = "/Users/cezar/Desktop/Team Project/AI/shotput/stage5/videos/0_user10.mp4"
    main(VIDEO_PATH)
