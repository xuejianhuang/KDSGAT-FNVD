from torchvision import transforms as T
import numpy as np
import torch
import os,random
from PIL import Image
import argparse
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='HXJ')
    parser.add_argument('--dataset', type=str, default='FakeSV')
    parser.add_argument('--model', type=str, default='KDSGAT-FNVD')
    parser.add_argument("--mode", type=str, default='test')
    args = parser.parse_args()
    return args

def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def str2num(str_x):
    str_x = str(str_x)
    if str_x.isdigit():
        return float(str_x)
    elif 'w' in str_x:  # For numbers with 'w' (ten thousand)
        return float(str_x[:-1]) * 10000
    elif '亿' in str_x:  # For numbers with '亿' (hundred million)
        return float(str_x[:-1]) * 100000000
    else:
        return 0

def clean_text(text, dataset="FakeSV"):
    """
      Clean text based on language:
      - Chinese: Retain Chinese characters, numbers, spaces, common punctuation marks, and `#` and `@`.
      - English: Retain letters, numbers, spaces, common punctuation marks, and `#` and `@`.

      :param text: The original text
      :param dataset: The dataset type ('FakeSV' or 'FakeTT')
      :return: The cleaned text
      """
    if dataset == "FakeSV":
        # Chinese: Retain Chinese characters, numbers, spaces, and common punctuation marks (including # and @)
        text = re.sub(r"[^\u4e00-\u9fa5\s\d.,!?;:'\"，。！？；：“”‘’#@]", "", text)
    elif dataset == "FakeTT":
        # English: Retain letters, numbers, spaces, and common punctuation marks (including # and @)
        text = re.sub(r"[^\w\s.,!?;:'\"#@-]", "", text)

    # Remove extra spaces and retain only one space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_verification_status(is_verified, dataset="FakeSV"):
    """
    Get verification status in the appropriate language.

    :param is_verified: Verification status (0: not verified, 1: personal, 2: organization)
    :param language: Language type ('ch' or 'en')
    :return: Verification status as a string
    """
    if dataset == "FakeTT":
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

def load_img_pil(image_path):
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_transform_compose():
    return T.Compose([
        T.Resize(224),  # Resize to 256px
        T.CenterCrop(224),  # Center crop to 224px
        #T.Resize((224, 224)),
        T.ToTensor(),  # Convert image to tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with pre-defined ImageNet values
    ])

def pad_frame_sequence(seq_len, lst):
    """
    Pad the frame sequence to match the specified length.

    :param seq_len: The target sequence length
    :param lst: The list of input video frame sequences
    :return: The padded frame sequences and the corresponding attention masks
    """
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]

        if ori_len >= seq_len:
            # If the number of video frames is greater than or equal to the target length, sample proportionally
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones(seq_len)
        else:
            # If the number of video frames is less than the target length, pad with zeros
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))

        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)

    return torch.stack(result), torch.stack(attention_masks)

def pad_frame_by_seg(seq_len, lst, seg):
    result = []
    seg_indicators = []
    sampled_seg = []
    for i in range(len(lst)):
        video = lst[i]
        v_sampled_seg = []
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        seg_video = seg[i]
        seg_len = len(seg_video)
        if seg_len >= seq_len:
            gap = seg_len // seq_len
            seg_video = seg_video[::gap][:seq_len]
            sample_index = []
            sample_seg_indicator = []
            for j in range(len(seg_video)):
                v_sampled_seg.append(seg_video[j])
                if seg_video[j][0] == seg_video[j][1]:
                    sample_index.append(seg_video[j][0])
                else:
                    sample_index.append(np.random.randint(seg_video[j][0], seg_video[j][1]))
                sample_seg_indicator.append(j)
            video = video[sample_index]
            mask = sample_seg_indicator
        else:
            if ori_len < seq_len:
                video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)

                mask = []
                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    mask.extend([j] * (seg_video[j][1] - seg_video[j][0] + 1))
                mask.extend([-1] * (seq_len - len(mask)))

            else:

                sample_index = []
                sample_seg_indicator = []
                seg_len = [(x[1] - x[0]) + 1 for x in seg_video]
                sample_ratio = [seg_len[i] / sum(seg_len) for i in range(len(seg_len))]
                sample_len = [seq_len * sample_ratio[i] for i in range(len(seg_len))]
                sample_per_seg = [int(x) + 1 if x < 1 else int(x) for x in sample_len]

                sample_per_seg = [x if x <= seg_len[i] else seg_len[i] for i, x in enumerate(sample_per_seg)]
                additional_sample = sum(sample_per_seg) - seq_len
                if additional_sample > 0:
                    idx = 0
                    while additional_sample > 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if sample_per_seg[idx] > 1:
                            sample_per_seg[idx] = sample_per_seg[idx] - 1
                            additional_sample = additional_sample - 1
                        idx += 1

                elif additional_sample < 0:
                    idx = 0
                    while additional_sample < 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if seg_len[idx] - sample_per_seg[idx] >= 1:
                            sample_per_seg[idx] = sample_per_seg[idx] + 1
                            additional_sample = additional_sample + 1
                        idx += 1

                for seg_idx in range(len(sample_per_seg)):
                    sample_seg_indicator.extend([seg_idx] * sample_per_seg[seg_idx])

                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    if sample_per_seg[j] == seg_len[j]:
                        sample_index.extend(np.arange(seg_video[j][0], seg_video[j][1] + 1))

                    else:
                        sample_index.extend(
                            np.sort(np.random.randint(seg_video[j][0], seg_video[j][1] + 1, sample_per_seg[j])))

                sample_index = np.array(sample_index)
                sample_index = np.sort(sample_index)
                video = video[sample_index]
                batch_sample_seg_indicator = np.array(sample_seg_indicator)
                mask = batch_sample_seg_indicator
                v_sampled_seg.sort(key=lambda x: x[0])

        result.append(video)
        mask = torch.IntTensor(mask)
        sampled_seg.append(v_sampled_seg)
        seg_indicators.append(mask)
    return torch.stack(result), torch.stack(seg_indicators), sampled_seg

def pad_segment(seg_lst, target_len):
    """
     Pad the segment sequence to match the target length.

     :param seg_lst: The input list of segments
     :param target_len: The target length
     :return: The padded list of segments
     """
    for sl_idx in range(len(seg_lst)):
        for s_idx in range(len(seg_lst[sl_idx])):
            seg_lst[sl_idx][s_idx] = torch.tensor(seg_lst[sl_idx][s_idx])

        if len(seg_lst[sl_idx]) < target_len:
            seg_lst[sl_idx].extend([torch.tensor([-1, -1])] * (target_len - len(seg_lst[sl_idx])))
        else:
            seg_lst[sl_idx] = seg_lst[sl_idx][:target_len]

        seg_lst[sl_idx] = torch.stack(seg_lst[sl_idx])

    return torch.stack(seg_lst)

def pad_unnatural_phrase(phrase_lst,target_len):
    for pl_idx in range(len(phrase_lst)):
        if len(phrase_lst[pl_idx])<target_len:
            phrase_lst[pl_idx]=torch.cat((phrase_lst[pl_idx],torch.zeros([target_len-len(phrase_lst[pl_idx]),phrase_lst[pl_idx].shape[1]],dtype=torch.long)),dim=0)
        else:
            phrase_lst[pl_idx]=phrase_lst[pl_idx][:target_len]
    return torch.stack(phrase_lst)

def get_dura_info_visual(segs,fps,total_frame):
    duration_frames=[]
    duration_time=[]
    for seg in segs:
        if seg[0]==-1 and seg[1]==-1:
            continue
        if seg[0]==0 and seg[1]==0:
            continue
        else:
            duration_frames.append(seg[1]-seg[0]+1)
            duration_time.append((seg[1]-seg[0]+1)/fps)
    duration_ratio=[min(dura/total_frame,1) for dura in duration_frames]
    return torch.tensor(duration_time).cuda(),torch.tensor(duration_ratio).cuda()

