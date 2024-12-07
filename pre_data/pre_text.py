import os
import json
import torch
import pickle
import re
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Configuration parameters
DATA = 'FakeSV'  # Dataset name
DATA_PATH = f"../data/{DATA}/data.json"  # Path to the data.json file
OUTPUT_DIR = f"../data/{DATA}/text_fea"  # Directory to save features
LANGUAGE = "cn" if DATA == "FakeSV" else 'en'  # Select language based on the dataset: Chinese ('cn') or English ('en')
MAX_TEXT_LENGTH = 200 if LANGUAGE == 'cn' else 100  # Set max input length for BERT based on the language

# Load BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("../bert-base-multilingual-uncased/")
model = BertModel.from_pretrained("../bert-base-multilingual-uncased/").to(device)
model.eval()  # 设置模型为评估模式，禁用 dropout 等操作


def clean_text(text, language="cn"):
    """
      Clean text based on language:
      - Chinese: Keeps Chinese characters, numbers, spaces, common punctuation, as well as `#` and `@`.
      - English: Keeps letters, numbers, spaces, common punctuation, as well as `#` and `@`.

      :param text: The original text
      :param language: The language type ('cn' or 'en')
      :return: The cleaned text
      """
    if language == "cn":
        # Chinese: Keep Chinese characters, numbers, spaces, and common punctuation (including # and @)
        text = re.sub(r"[^\u4e00-\u9fa5\s\d.,!?;:'\"，。！？；：“”‘’#@]", "", text)
    elif language == "en":
        # English: Keep letters, numbers, spaces, and common punctuation (including # and @)
        text = re.sub(r"[^\w\s.,!?;:'\"#@-]", "", text)

    # Remove extra spaces, leaving only one space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_bert_features(text, max_length=512):
    """
        Extract text features using BERT.

        :param text: The input text
        :param max_length: The maximum sequence length
        :return: The feature representation of the text
        """
    tokens = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        feature = outputs.last_hidden_state  # Extract the last hidden state
    return feature.squeeze(0)   # Remove the batch dimension and return the feature for a single sample


def process_data(data_path, output_dir, language="cn"):
    """
     Process JSON data and extract text features.

     :param data_path: Path to the JSON file containing data
     :param output_dir: Directory to save the extracted features
     :param language: The language type of the text ('cn' or 'en')
     """
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the JSON file and process it
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing text", unit="file"):
        try:
            # 解析每一行 JSON 数据
            data = json.loads(line.strip())
            video_id = data["video_id"]
            raw_text = data["title"] + ":" + data["text"]  # Concatenate title and text

            # Clean the text
            clean_text_data = clean_text(raw_text, language=language)

            # Extract text features
            feature = extract_bert_features(clean_text_data, max_length=MAX_TEXT_LENGTH)

            # Save features as a .pkl file
            output_path = os.path.join(output_dir, f"{video_id}.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(feature.cpu(), f)  # 将特征保存为pkl格式

        except Exception as e:
            # Print warning message if an error occurs and continue to the next video
            print(f"Error processing video {data.get('video_id', 'Unknown')}: {e}")


if __name__ == "__main__":
    # Run data processing and feature extraction
    process_data(DATA_PATH, OUTPUT_DIR, language=LANGUAGE)
