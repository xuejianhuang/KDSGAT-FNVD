import os
import json
import torch
import pickle
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Configuration
DATA = 'FakeSV'  # dataset
DATA_PATH = f"../data/{DATA}/data.json"  # Path to the JSON data file
OUTPUT_DIR = f"../data/{DATA}/user_fea"  # Directory to save extracted features
MAX_TEXT_LENGTH = 50  # Maximum input length for BERT
LANGUAGE = "cn" if DATA == "FakeSV" else 'en'  # 根据数据集选择语言：中文（'cn'）或英文（'en'）

# Load BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("../bert-base-multilingual-uncased/")
model = BertModel.from_pretrained("../bert-base-multilingual-uncased/").to(device)
model.eval()


def extract_bert_features(text, max_length=50):
    """
    Extract text features using BERT.

    :param text: Input text
    :param max_length: Maximum sequence length
    :return: Text feature representation as a tensor
    """
    tokens = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:,0,:]
    return features.squeeze(0)  # Remove batch dimension


def get_verification_status(is_verified, language="ch"):
    """
    Get verification status in the appropriate language.

    :param is_verified: Verification status (0: not verified, 1: personal, 2: organization)
    :param language: Language type ('ch' or 'en')
    :return: Verification status as a string
    """
    if language == "en":
        statuses = {
            1: "Verification",
            0: "Not Verified",
        }
        return statuses.get(is_verified, "Unknown Verification Status")
    else:
        statuses = {
            1: "个人认证",
            2: "机构认证",
            0: "未认证",
        }
        return statuses.get(is_verified, "认证状态未知")


def process_data(data_path, output_dir, language="ch"):
    """
    Process JSON data and extract features.

    :param data_path: Path to the JSON data file
    :param output_dir: Directory to save extracted features
    :param language: Language type ('ch' or 'en')
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(data_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in tqdm(lines, desc="Processing User"):
        try:
            data = json.loads(line.strip())
            video_id = data["video_id"]
            author_intro = data.get("author_intro", "No introduction provided")
            is_author_verified = data.get("is_author_verified", -1)

            # Get verification status in the appropriate language
            verification_status = get_verification_status(is_author_verified, language)

            # Combine verification status and author introduction
            combined_text = f"{verification_status}: {author_intro}"

            # Extract features
            features = extract_bert_features(combined_text, max_length=MAX_TEXT_LENGTH)

            # Save features as a .pkl file
            output_path = os.path.join(output_dir, f"{video_id}.pkl")
            with open(output_path, "wb") as output_file:
                pickle.dump(features.cpu(), output_file)

        except Exception as e:
            print(f"Error processing video {data.get('video_id', 'unknown')}: {e}")


if __name__ == "__main__":
    process_data(DATA_PATH, OUTPUT_DIR, language=LANGUAGE)
