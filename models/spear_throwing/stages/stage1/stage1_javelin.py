import os
import cv2
import torch
import logging
import numpy as np
from PIL import Image
import torch.nn as nn
import mediapipe as mp
from transformers import CLIPProcessor, CLIPModel

def get_local_path(video_path):
    return video_path  # Simply return the video path as is

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PTH_PATH = "/Users/danyukezz/Desktop/2 year 1 semester/team project/preprocessing_sports/spear_throwing/stages/stage1/models/stage1_javelin.pth"
VIDEO_DIR = "/Users/danyukezz/Desktop/2 year 1 semester/team project/preprocessing_sports/spear_throwing/stages/stage1/test_videos"
SEQUENCE_LENGTH = 20
INPUT_SIZE = 513

# Initialize CLIP and MediaPipe
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

# Check if model file exists
if not os.path.exists(MODEL_PTH_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PTH_PATH}")

class TemporalModel(nn.Module):
    def __init__(self, input_size=513, hidden_size=128, num_layers=1, output_size=1):
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output

def load_javelin_model():
    try:
        logger.info(f"Loading model weights from: {MODEL_PTH_PATH}")
        model = TemporalModel()
        state_dict = torch.load(MODEL_PTH_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading PyTorch model: {str(e)}")
        raise

def process_and_predict(video_path, model, sequence_length):
    logger.info(f"Processing video: {os.path.basename(video_path)}")
    
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

    keypoints = []
    for frame in frames:
        result = pose.process(frame)
        if result.pose_landmarks:
            keypoints.append([
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in result.pose_landmarks.landmark
            ])

    clip_embeddings = []
    for frame in frames:
        image = Image.fromarray(frame)
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs).cpu().numpy().flatten()
            clip_embeddings.append(embedding)

    combined_features = []
    for i, clip_embedding in enumerate(clip_embeddings):
        feature_vector = np.concatenate([clip_embedding])
        if len(feature_vector) > INPUT_SIZE:
            feature_vector = feature_vector[:INPUT_SIZE]
        elif len(feature_vector) < INPUT_SIZE:
            feature_vector = np.concatenate([feature_vector, np.zeros(INPUT_SIZE - len(feature_vector))])
        combined_features.append(feature_vector)

    combined_features = torch.tensor(combined_features, dtype=torch.float32)

    if combined_features.shape[0] >= sequence_length:
        combined_features = combined_features[:sequence_length]
    else:
        padding = torch.zeros((sequence_length - combined_features.shape[0], combined_features.shape[1]))
        combined_features = torch.cat((combined_features, padding), dim=0)

    model.eval()
    with torch.no_grad():
        combined_features = combined_features.unsqueeze(0).to(device)
        prediction = model(combined_features).squeeze().cpu().item()

    prediction = 0 if prediction < 0.35 else 0.5 if prediction < 0.65 else 1
    logger.info(f"Thresholded Prediction: {prediction}")
    return prediction

results = {}
for video_file in os.listdir(VIDEO_DIR):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        prediction = process_and_predict(video_path, load_javelin_model(), SEQUENCE_LENGTH)
        results[video_file] = prediction

logger.info("\nPrediction Results:")
for video, prediction in results.items():
    logger.info(f"{video}: {prediction}")
