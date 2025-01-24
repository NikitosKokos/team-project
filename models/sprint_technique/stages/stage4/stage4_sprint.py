import os
import numpy as np
import logging
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_PTH_PATH = "/Users/danyukezz/Desktop/2 year 1 semester/team project/danya_preprocessing_sports/sprint_technique/stages/stage4/models/stage4_sprint.pth"
VIDEO_DIR = "/Users/danyukezz/Desktop/2 year 1 semester/team project/danya_preprocessing_sports/sprint_technique/stages/stage4/test_videos"
SEQUENCE_LENGTH = 20
INPUT_SIZE = 513

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP and MediaPipe
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

# Check model path
if not os.path.exists(MODEL_PTH_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PTH_PATH}")

# LSTM Model definition
class TemporalModel(nn.Module):
    def __init__(self, input_size=513, hidden_size=128, num_layers=1, output_size=1):
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output, hidden

# Load model
def load_sprint_model():
    try:
        model = TemporalModel()
        state_dict = torch.load(MODEL_PTH_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)
    except Exception as e:
        logger.error(f"Error loading PyTorch model: {str(e)}")
        raise

# Frame interpolation function
def interpolate_frames(frames, target_frame_count):
    original_count = len(frames)
    if original_count >= target_frame_count:
        return frames[:target_frame_count]

    interpolated_frames = []
    for i in range(target_frame_count):
        index = (i / target_frame_count) * (original_count - 1)
        low_idx = int(np.floor(index))
        high_idx = min(low_idx + 1, original_count - 1)
        alpha = index - low_idx
        blended_frame = cv2.addWeighted(frames[low_idx], 1 - alpha, frames[high_idx], alpha, 0)
        interpolated_frames.append(blended_frame)

    return interpolated_frames

# Process video and predict
def process_and_diagnose(video_path, model, sequence_length):
    logger.info(f"Processing video: {os.path.basename(video_path)}")

    # Extract frames
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        count += 1
    cap.release()

    # Interpolate frames if needed
    if len(frames) < sequence_length:
        logger.info(f"Interpolating frames for {video_path} to reach {sequence_length}")
        frames = interpolate_frames(frames, sequence_length)

    # Extract keypoints
    keypoints = []
    for frame in frames:
        result = pose.process(frame)
        if result.pose_landmarks:
            keypoints.append([
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in result.pose_landmarks.landmark
            ])

    if not keypoints:
        logger.error("No keypoints detected, indicating potential preprocessing issues.")
        return {"error": "No keypoints detected"}

    # Check missing keypoints
    missing_keypoints = sum(1 for kp in keypoints if len(kp) < 33)
    logger.info(f"Missing keypoints in {missing_keypoints}/{len(frames)} frames")

    # Compute stride-based feature
    stride_scores = [keypoints[i][32]['y'] - keypoints[i-1][32]['y'] > 0 for i in range(1, len(keypoints))]

    # Extract CLIP embeddings
    clip_embeddings = []
    for frame in frames:
        image = Image.fromarray(frame)
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs).cpu().numpy().flatten()
            clip_embeddings.append(embedding)

    # Compare feature distribution
    if len(clip_embeddings) > 1:
        similarity = cosine_similarity([clip_embeddings[0]], [clip_embeddings[-1]])[0][0]
        logger.info(f"CLIP embedding similarity (first vs last frame): {similarity:.4f}")

    # Combine features
    combined_features = []
    for i, clip_embedding in enumerate(clip_embeddings):
        feature_vector = np.concatenate([
            clip_embedding,
            [stride_scores[i] if i < len(stride_scores) else 0]
        ])
        feature_vector = np.pad(feature_vector, (0, max(0, INPUT_SIZE - len(feature_vector))), 'constant')
        combined_features.append(feature_vector)

    # Normalize features
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    combined_features = torch.tensor(combined_features, dtype=torch.float32).to(device)
    if combined_features.shape[0] < sequence_length:
        padding = torch.zeros((sequence_length - combined_features.shape[0], INPUT_SIZE)).to(device)
        combined_features = torch.cat((combined_features, padding), dim=0)

    # Model inference
    model.eval()
    with torch.no_grad():
        combined_features = combined_features.unsqueeze(0)
        prediction, hidden_states = model(combined_features)
        prediction = prediction.squeeze().cpu().item()

    logger.info(f"Raw prediction value: {prediction:.4f}")

    # Diagnose hidden states
    hidden_variance = torch.var(hidden_states).cpu().item()
    logger.info(f"LSTM hidden state variance: {hidden_variance:.4f}")

    # Threshold predictions
    if prediction < 0.35:
        final_prediction = 0
    elif 0.35 <= prediction < 0.65:
        final_prediction = 0.5
    else:
        final_prediction = 1

    logger.info(f"Thresholded Prediction: {final_prediction}")

    # Diagnosis report
    diagnosis_report = {
        "raw_prediction": prediction,
        "thresholded_prediction": final_prediction,
        "clip_similarity": similarity,
        "missing_keypoints": missing_keypoints,
        "hidden_state_variance": hidden_variance
    }

    return diagnosis_report

# Run diagnosis
results = {}
model = load_sprint_model()
for video_file in os.listdir(VIDEO_DIR):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        diagnosis = process_and_diagnose(video_path, model, SEQUENCE_LENGTH)
        results[video_file] = diagnosis

logger.info("\nDiagnosis Report:")
for video, report in results.items():
    logger.info(f"{video}: {report}")