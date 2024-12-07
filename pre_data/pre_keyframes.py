import os
import torch
import pickle
from transformers import SwinModel
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm


# Configuration
DATA = "FakeTT"
MAX_KEYFRAMES = 10  # Maximum number of keyframes to keep for each video
FEATURE_DIM = 768   # Feature dimension for Swin Transformer (based on model settings)
KEYFRAMES_DIR = f"../data/{DATA}/keyframes"  # Directory where keyframes are stored
OUTPUT_DIR = f"../data/{DATA}/keyframes_fea"  # Directory to save extracted features

# Load Swin Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinModel.from_pretrained('../swin-transformer').to(device)
model.eval()

# Image preprocessing
transform = T.Compose([
    T.Resize(224),               # Resize image to 224
    T.CenterCrop(224),           # Center crop to 224x224
    T.ToTensor(),                # Convert to Tensor format
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])


def extract_features_from_keyframes(keyframes_dir, output_dir, max_keyframes=15, feature_dim=768):
    """
      Extract features from keyframes and save them as .pkl files.

      :param keyframes_dir: Directory containing video keyframes
      :param output_dir: Directory to save the extracted features
      :param max_keyframes: Maximum number of keyframes to keep per video
      :param feature_dim: Dimensionality of the feature representation
      """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Traverse the keyframe directories of each video
    video_ids = [d for d in os.listdir(keyframes_dir) if os.path.isdir(os.path.join(keyframes_dir, d))]
    for video_id in tqdm(video_ids, desc="Extracting features"):
        video_path = os.path.join(keyframes_dir, video_id)

        # Read and sort keyframe files
        keyframe_files = sorted(os.listdir(video_path))[:max_keyframes]

        features = []
        for frame_file in keyframe_files:
            frame_path = os.path.join(video_path, frame_file)
            try:
                img = Image.open(frame_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)   # Add batch dimension
                with torch.no_grad():
                    feature = model(img_tensor).last_hidden_state.mean(dim=1)  # Extract features
                features.append(feature.cpu().squeeze(0))   # Remove batch dimension
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")

        # Pad with zeros if there are fewer than max_keyframes
        while len(features) < max_keyframes:
            features.append(torch.zeros(feature_dim))

        # Convert to Tensor and save
        features_tensor = torch.stack(features)  # [max_keyframes, feature_dim]
        output_path = os.path.join(output_dir, f"{video_id}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(features_tensor, f)

        print(f"Saved features: {output_path}")


if __name__ == "__main__":
    extract_features_from_keyframes(KEYFRAMES_DIR, OUTPUT_DIR, max_keyframes=MAX_KEYFRAMES, feature_dim=FEATURE_DIM)
