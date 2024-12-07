import os
import torch
import pickle
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import librosa
from tqdm import tqdm

if __name__ == '__main__':

    # CONFIG and MODEL SETUP
    model_name = '../ExHuBERT'
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("../hubert-base-ls960")
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True,
                                                            output_hidden_states=True,
                                                            revision="b158d45ed8578432468f3ab8d46cbe5974380812")

    # Freezing half of the encoder for further transfer learning
    model.freeze_og_encoder()

    sampling_rate = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Process all audio files in the 'audio' directory
    #audio_dir = "../../FakeSV_dataset/FakeSV/audio/"
    DATA="FakeTT"
    audio_dir = f"../data/{DATA}/audio/"

    # Directories for saving features
    hidden_stats_dir = audio_dir+"hidden_stats"
    emotions_dir = audio_dir+"emotions"

    # Ensure the directories exist
    os.makedirs(hidden_stats_dir, exist_ok=True)
    os.makedirs(emotions_dir, exist_ok=True)

    # List all audio files in the directory
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith((".mp3", ".wav"))]

    # Iterate over all files in the audio directory
    for filename in tqdm(audio_files, desc="Processing Audio Files", unit="file"):
            audio_path = os.path.join(audio_dir, filename)

            # Load the audio file
            waveform, sr_wav = librosa.load(audio_path, sr=sampling_rate)

            # Max Padding to 3 Seconds at 16k sampling rate for the best results
            waveform = feature_extractor(waveform, sampling_rate=sampling_rate, padding='max_length', max_length=48000)
            waveform = waveform['input_values'][0]
            waveform = waveform.reshape(1, -1)
            waveform = torch.from_numpy(waveform).to(device)

            # Forward pass through the model to get the output
            with torch.no_grad():
                output = model(waveform)

                # Extract hidden states and logits
                hidden_stats_mean = []
                hidden_stats = output.hidden_states
                for hs in hidden_stats:
                    hidden_stats_mean.append(hs.squeeze(0).mean(dim=0))  # Calculate mean across time
                hidden_stats_mean = torch.stack(hidden_stats_mean, dim=0)
                emotions = output.logits

                # Define filenames for saving the features
                base_filename = os.path.splitext(filename)[0]  # Remove file extension
                hidden_stats_filename = os.path.join(hidden_stats_dir, f"{base_filename}.pkl")
                emotions_filename = os.path.join(emotions_dir, f"{base_filename}.pkl")

            with open(hidden_stats_filename, 'wb') as f:
                pickle.dump(hidden_stats_mean.cpu(), f)

            with open(emotions_filename, 'wb') as f:
                pickle.dump(emotions.cpu(), f)

            print(f"Processed {filename}: saved hidden stats and logits.")

    print("Feature extraction complete.")
