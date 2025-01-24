import cv2
import mediapipe as mp
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MODEL_PATH = "/Users/cezar/Desktop/Team Project/AI/shotput/stage2/shotput_stage2.keras"
OUTPUT_JSON = "stage2_results.json"
MAX_SEQUENCE_LENGTH = 76  # Update based on training max_len

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

# Function to calculate distance between two points
def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# Function to classify prediction into 0, 0.5, or 1
def classify_score(prediction):
    if prediction >= 0.90:
        return 1.0
    elif prediction >= 0.75:
        return 0.5
    else:
        return 0.0

def extract_features_from_video(video_path):
    angles = []
    distances = []

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Extract relevant keypoints for the right leg
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

            # Calculate angle and distance
            angle = calculate_angle(right_hip, right_knee, right_ankle)
            distance = calculate_distance(right_hip, right_knee)

            angles.append(angle)
            distances.append(distance)

    cap.release()

    # Combine features
    features = np.stack([angles, distances], axis=1)
    return features

def main(video_path):
    print("Extracting features from video...")
    features = extract_features_from_video(video_path)  # Shape: (sequence_length, 2)

    print("Padding sequences to match the model input shape...")
    # Manually pad or truncate to match the model's input shape (76, 2)
    if features.shape[0] < 76:
        # Pad with zeros if the sequence is shorter
        padding = np.zeros((76 - features.shape[0], 2))
        features_padded = np.vstack((features, padding))
    else:
        # Truncate if the sequence is longer
        features_padded = features[:76, :]

    # Add batch dimension to get shape (1, 76, 2)
    features_padded = np.expand_dims(features_padded, axis=0)

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Predicting score...")
    # Input shape: (1, 76, 2)
    prediction = model.predict(features_padded)[0][0]
    classified_score = classify_score(prediction)

    print("Saving results...")
    results = {
        "video": video_path,
        "predicted_score": float(prediction),
        "classified_score": float(classified_score)
    }

    with open(OUTPUT_JSON, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {OUTPUT_JSON}")



if __name__ == "__main__":
    # Replace with actual path to the new video
    VIDEO_PATH = "/Users/cezar/Desktop/Team Project/AI/shotput/stage2/keypoint_videos/0_user19.mp4"
    main(VIDEO_PATH)