import os
import numpy as np
import logging
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_KERAS_PATH = "/Users/danyukezz/Desktop/2 year 1 semester/team project/danya_preprocessing_sports/sprint_technique/stages/stage5/models/stage5_sprint_keras_with_weights.keras"
VIDEO_DIR = "/Users/danyukezz/Desktop/2 year 1 semester/team project/danya_preprocessing_sports/sprint_technique/stages/stage5/test_videos"
SEQUENCE_LENGTH = 40
INPUT_SIZE = 513

# Load Keras model
keras_model = load_model(MODEL_KERAS_PATH)
logger.info("Keras model loaded successfully.")

# Load CLIP and MediaPipe
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

# Load scaler trained on original dataset
stats_file = "/Users/danyukezz/Desktop/2 year 1 semester/team project/danya_preprocessing_sports/sprint_technique/stages/stage5/weights/train_feature_statistics.npy"
if os.path.exists(stats_file):
    stats = np.load(stats_file)
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = stats[0], stats[1]
    logger.info("Feature scaler loaded successfully.")
else:
    logger.error(f"Feature statistics file not found: {stats_file}")
    exit(1)

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

    # Ensure stride_scores length matches frame length
    stride_scores = np.zeros(len(frames))

    # Extract CLIP embeddings
    clip_embeddings = []
    for frame in frames:
        image = Image.fromarray(frame)
        inputs = clip_processor(images=image, return_tensors="pt")
        with tf.device('/CPU:0'):
            embedding = clip_model.get_image_features(**inputs).detach().numpy().flatten()
            clip_embeddings.append(embedding)

    # Ensure equal lengths for concatenation
    min_length = min(len(clip_embeddings), len(stride_scores))
    clip_embeddings = np.array(clip_embeddings[:min_length])
    stride_scores = stride_scores[:min_length]

    # Combine features
    combined_features = np.hstack((clip_embeddings, stride_scores.reshape(-1, 1)))
    combined_features = scaler.transform(combined_features)
    combined_features = np.expand_dims(combined_features, axis=0)
    
    # Model inference
    prediction = model.predict(combined_features)[0][0]
    logger.info(f"Raw prediction value: {prediction:.4f}")

    # Adaptive thresholding
    final_prediction = 0 if prediction < 0.35 else (0.5 if prediction < 0.65 else 1)
    logger.info(f"Thresholded Prediction: {final_prediction}")

    return {"raw_prediction": prediction, "thresholded_prediction": final_prediction}

# Run diagnosis
results = {}
for video_file in os.listdir(VIDEO_DIR):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        diagnosis = process_and_diagnose(video_path, keras_model, SEQUENCE_LENGTH)
        results[video_file] = diagnosis

logger.info("\nDiagnosis Report:")
for video, report in results.items():
    logger.info(f"{video}: {report}")