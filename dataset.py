import os
import pickle
from dgl import load_graphs
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from dgl.dataloading import GraphDataLoader
from transformers import BertTokenizer
from utils.utils import clean_text,get_verification_status
from utils.collate_fn import *
from config import base_config, KDSGAT_FNVD_config
from models.KDSGAT_FNVD  import *
from models.FakingRecipe import *

class BaseDataset(Dataset):
    def __init__(self, dataset, type='train', max_text_length=200):
        self.dataset = dataset
        self.max_text_length = max_text_length
        self._tokenizer = BertTokenizer.from_pretrained(KDSGAT_FNVD_config.bert_dir)
        self.data_root_dir = self._get_data_root_dir()
        self.data_complete = self._load_data_json()

        video_ids = self._load_video_ids(type)
        self.data = self.data_complete[self.data_complete.video_id.isin(video_ids)].reset_index(drop=True)

    def _get_data_root_dir(self):
        if self.dataset == 'FakeSV':
            return base_config.FakeSV_dataset_dir
        else:
            return base_config.FakeTT_dataset_dir

    def _load_data_json(self):
        data_json_path = os.path.join(self.data_root_dir, KDSGAT_FNVD_config.data_json_path)
        return pd.read_json(data_json_path, orient='records', dtype=False, lines=True)

    def _load_video_ids(self, type):
        path_vid = {
            'train': base_config.train_dataset,
            'val': base_config.val_dataset,
            'test': base_config.test_dataset,
        }.get(type, base_config.test_dataset)

        with open(os.path.join(self.data_root_dir, path_vid), "r") as fr:
            return [line.strip() for line in fr.readlines()]

    def __len__(self):
        return len(self.data)

    def get_text_tokens(self, text):
        text = clean_text(text, dataset=self.dataset)
        text_tokens = self._tokenizer(text, return_tensors='pt', max_length=self.max_text_length,
                                      padding='max_length', truncation=True)
        for key in text_tokens:
            text_tokens[key]=text_tokens[key].squeeze(0)
        return text_tokens

    def get_author_intro_tokens(self, author_intro, is_author_verified):
        verification_status = get_verification_status(is_author_verified, dataset=self.dataset)
        combined_text = f"{verification_status}: {author_intro}"
        author_intro_token = self._tokenizer(combined_text, return_tensors='pt',
                                             max_length=KDSGAT_FNVD_config.author_intro_max_length,
                                             padding='max_length', truncation=True)
        for key in author_intro_token:
            author_intro_token[key] = author_intro_token[key].squeeze(0)
        return author_intro_token

    def load_pickle_data(self, file_path):
        with open(file_path, "rb") as fr:
            return pickle.load(fr).clone().detach()

    def get_scene_graph(self, vid):
        scene_graph_path = os.path.join(self.data_root_dir + KDSGAT_FNVD_config.scene_graph_path, f"{vid}.bin")
        return load_graphs(scene_graph_path)[0][:KDSGAT_FNVD_config.keyframes_max_num]

class KDSGAT_FNVD_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        max_text_length = 200 if dataset == 'FakeSV' else 100
        super().__init__(dataset, type, max_text_length)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        text = item["title"] + ":" + item["text"]
        text_tokens = self.get_text_tokens(text)

        author_intro = item.get("author_intro", "No introduction provided")
        is_author_verified = item.get("is_author_verified", -1)
        author_intro_token = self.get_author_intro_tokens(author_intro, is_author_verified)

        audio_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.audio_fea_path, f"{vid}.pkl"))
        keyframes_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.keyframes_fea_path, f"{vid}.pkl"))[:KDSGAT_FNVD_config.keyframes_max_num]
        scene_graph = self.get_scene_graph(vid)

        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'text_tokens': text_tokens.to(base_config.device),
            'author_intro_token': author_intro_token.to(base_config.device),
            'audio_fea': audio_fea.to(base_config.device),
            'keyframes_fea': keyframes_fea.to(base_config.device),
            'scene_graph': [sg.to(base_config.device) for sg in scene_graph]
        }

class KF_wo_user_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        max_text_length = 200 if dataset == 'FakeSV' else 100
        super().__init__(dataset, type, max_text_length)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        text = item["title"] + ":" + item["text"]
        text_tokens = self.get_text_tokens(text)

        audio_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.audio_fea_path, f"{vid}.pkl"))
        keyframes_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.keyframes_fea_path, f"{vid}.pkl"))[:KDSGAT_FNVD_config.keyframes_max_num]
        scene_graph = self.get_scene_graph(vid)

        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'text_tokens': text_tokens.to(base_config.device),
            'audio_fea': audio_fea.to(base_config.device),
            'keyframes_fea': keyframes_fea.to(base_config.device),
            'scene_graph': [sg.to(base_config.device) for sg in scene_graph]
        }

class KF_wo_text_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        super().__init__(dataset, type)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)


        author_intro = item.get("author_intro", "No introduction provided")
        is_author_verified = item.get("is_author_verified", -1)
        author_intro_token = self.get_author_intro_tokens(author_intro, is_author_verified)

        audio_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.audio_fea_path, f"{vid}.pkl"))
        keyframes_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.keyframes_fea_path, f"{vid}.pkl"))[:KDSGAT_FNVD_config.keyframes_max_num]
        scene_graph = self.get_scene_graph(vid)

        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'author_intro_token': author_intro_token.to(base_config.device),
            'audio_fea': audio_fea.to(base_config.device),
            'keyframes_fea': keyframes_fea.to(base_config.device),
            'scene_graph': [sg.to(base_config.device) for sg in scene_graph]
        }

class KF_wo_audio_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        max_text_length = 200 if dataset == 'FakeSV' else 100
        super().__init__(dataset, type, max_text_length)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        text = item["title"] + ":" + item["text"]
        text_tokens = self.get_text_tokens(text)

        author_intro = item.get("author_intro", "No introduction provided")
        is_author_verified = item.get("is_author_verified", -1)
        author_intro_token = self.get_author_intro_tokens(author_intro, is_author_verified)

        keyframes_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.keyframes_fea_path, f"{vid}.pkl"))[:KDSGAT_FNVD_config.keyframes_max_num]
        scene_graph = self.get_scene_graph(vid)

        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'text_tokens': text_tokens.to(base_config.device),
            'author_intro_token': author_intro_token.to(base_config.device),
            'keyframes_fea': keyframes_fea.to(base_config.device),
            'scene_graph': [sg.to(base_config.device) for sg in scene_graph]
        }

class KF_wo_visual_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        max_text_length = 200 if dataset == 'FakeSV' else 100
        super().__init__(dataset, type, max_text_length)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        text = item["title"] + ":" + item["text"]
        text_tokens = self.get_text_tokens(text)

        author_intro = item.get("author_intro", "No introduction provided")
        is_author_verified = item.get("is_author_verified", -1)
        author_intro_token = self.get_author_intro_tokens(author_intro, is_author_verified)

        audio_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.audio_fea_path, f"{vid}.pkl"))
        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'text_tokens': text_tokens.to(base_config.device),
            'author_intro_token': author_intro_token.to(base_config.device),
            'audio_fea': audio_fea.to(base_config.device)
        }

class KF_wo_SG_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        max_text_length = 200 if dataset == 'FakeSV' else 100
        super().__init__(dataset, type, max_text_length)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        text = item["title"] + ":" + item["text"]
        text_tokens = self.get_text_tokens(text)

        author_intro = item.get("author_intro", "No introduction provided")
        is_author_verified = item.get("is_author_verified", -1)
        author_intro_token = self.get_author_intro_tokens(author_intro, is_author_verified)

        audio_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.audio_fea_path, f"{vid}.pkl"))
        keyframes_fea = self.load_pickle_data(os.path.join(self.data_root_dir + KDSGAT_FNVD_config.keyframes_fea_path, f"{vid}.pkl"))[:KDSGAT_FNVD_config.keyframes_max_num]

        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'text_tokens': text_tokens.to(base_config.device),
            'author_intro_token': author_intro_token.to(base_config.device),
            'audio_fea': audio_fea.to(base_config.device),
            'keyframes_fea': keyframes_fea.to(base_config.device)
        }

class oh_user_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        super().__init__(dataset, type)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        user_fea = self.load_pickle_data(
            os.path.join(self.data_root_dir + KDSGAT_FNVD_config.user_fea_path, f"{vid}.pkl"))

        # Return the data as a dictionary
        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'user_fea': user_fea.to(base_config.device)
                   }

class oh_text_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        super().__init__(dataset, type)
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        text_fea = self.load_pickle_data(
            os.path.join(self.data_root_dir + KDSGAT_FNVD_config.text_fea_path, f"{vid}.pkl"))

        # Return the data as a dictionary
        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'text_fea': text_fea.to(base_config.device)
                   }

class oh_audio_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        super().__init__(dataset, type)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        audio_fea = self.load_pickle_data(
            os.path.join(self.data_root_dir + KDSGAT_FNVD_config.audio_fea_path, f"{vid}.pkl"))

        # Return the data as a dictionary
        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'audio_fea': audio_fea.to(base_config.device)
                   }

class oh_visual_Dataset(BaseDataset):
    def __init__(self, dataset, type='train'):
        super().__init__(dataset, type)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # Label: 0 for 'c', 1 for 'false'
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        keyframes_fea = self.load_pickle_data(
            os.path.join(self.data_root_dir + KDSGAT_FNVD_config.keyframes_fea_path, f"{vid}.pkl"))[
                        :KDSGAT_FNVD_config.keyframes_max_num]
        scene_graph = self.get_scene_graph(vid)

        # Return the data as a dictionary
        return {
            'vid': vid,
            'label': label.to(base_config.device),
            'keyframes_fea': keyframes_fea.to(base_config.device),
            'scene_graph': [sg.to(base_config.device) for sg in scene_graph]
                   }

class FakingRecipe_Dataset(Dataset):
    config=FakingRecipe_config()
    def __init__(self, dataset,type='train'):

        self.dataset = dataset
        if type == 'train':
            vid_path =self.config.train_dataset
        elif type == 'val':
            vid_path = self.config.val_dataset
        else:
            vid_path = self.config.test_dataset

        if dataset == 'FakeSV':
            self.data_root_dir = self.config.FakeSV_dataset_dir
            self.data_all = pd.read_json(self.config.FakeSV_metainfo_path, orient='records', dtype=False, lines=True)
            self._load_video_ids(self.data_root_dir+vid_path)
            self.ocr_pattern_fea_path = self.config.FakeSV_ocr_pattern_fea_path
            self._load_features('fakesv')
        elif dataset == 'FakeTT':
            self.data_root_dir = self.config.FakeTT_dataset_dir
            self.data_all = pd.read_json(self.config.FakeTT_metainfo_path, orient='records', lines=True,
                                         dtype={'video_id': str})
            self._load_video_ids(self.data_root_dir+vid_path)
            self.ocr_pattern_fea_path = self.config.FakeTT_ocr_pattern_fea_path
            self._load_features('fakett')
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _load_video_ids(self, vid_path):
        self.vid = []
        with open(vid_path, "r") as fr:
            self.vid = [line.strip() for line in fr.readlines()]
        self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
        self.data.reset_index(inplace=True)

    def _load_features(self, dataset_type):
        ocr_phrase_fea_path = f'{self.config.dataset_dir}/{dataset_type}/preprocess_ocr/ocr_phrase_fea.pkl'
        with open(ocr_phrase_fea_path, 'rb') as f:
            self.ocr_phrase = torch.load(f)

        text_semantic_fea_path = f'{self.config.dataset_dir}/{dataset_type}/preprocess_text/sem_text_fea.pkl'
        with open(text_semantic_fea_path, 'rb') as f:
            self.text_semantic_fea = torch.load(f)

        text_emo_fea_path = f'{self.config.dataset_dir}/{dataset_type}/preprocess_text/emo_text_fea.pkl'
        with open(text_emo_fea_path, 'rb') as f:
            self.text_emo_fea = torch.load(f)

        self.audio_fea_path = f'{self.config.dataset_dir}/{dataset_type}/preprocess_audio'
        self.visual_fea_path = f'{self.config.dataset_dir}/{dataset_type}/preprocess_visual'

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data.iloc[idx]
        vid = item['video_id']
        label = 1 if item['annotation'] == 'fake' else 0
        fps = torch.tensor(item['fps'])
        total_frame = torch.tensor(item['frame_count'])
        visual_time_region = torch.tensor(item['transnetv2_segs'])

        all_phrase_semantic_fea = self.text_semantic_fea['last_hidden_state'][vid]
        all_phrase_emo_fea = self.text_emo_fea['pooler_output'][vid]

        v_fea_path = os.path.join(self.visual_fea_path, vid + '.pkl')
        raw_visual_frames = torch.tensor(torch.load(v_fea_path))

        a_fea_path = os.path.join(self.audio_fea_path, vid + '.pkl')
        raw_audio_emo = torch.load(a_fea_path)

        ocr_pattern_fea_file_path = os.path.join(self.ocr_pattern_fea_path, vid, 'r0.pkl')
        ocr_pattern_fea = torch.tensor(torch.load(ocr_pattern_fea_file_path))

        ocr_phrase_fea = self.ocr_phrase['ocr_phrase_fea'][vid]
        ocr_time_region = self.ocr_phrase['ocr_time_region'][vid]

        return {
            'vid': vid,
            'label': torch.tensor(label),
            'fps': fps,
            'total_frame': total_frame,
            'all_phrase_semantic_fea': all_phrase_semantic_fea,
            'all_phrase_emo_fea': all_phrase_emo_fea,
            'raw_visual_frames': raw_visual_frames,
            'raw_audio_emo': raw_audio_emo,
            'ocr_pattern_fea': ocr_pattern_fea,
            'ocr_phrase_fea': ocr_phrase_fea,
            'ocr_time_region': ocr_time_region,
            'visual_time_region': visual_time_region
        }

def getModelAndData(model_name='KDSGAT-FNVD', dataset='FakeSV'):
    # Model map to associate model name with corresponding class, dataset, and dataloader
    model_map = {
        'KDSGAT-FNVD': [KDSGAT_FNVD, KDSGAT_FNVD_Dataset, GraphDataLoader],  # [model, dataset, dataloader]
        'KDGAT-FNVD': [KDGAT_FNVD, KDSGAT_FNVD_Dataset, GraphDataLoader],
        'KF_wo_user': [KF_wo_user, KF_wo_user_Dataset, GraphDataLoader],
        'KF_wo_text': [KF_wo_text, KF_wo_text_Dataset, GraphDataLoader],
        'KF_wo_audio': [KF_wo_audio, KF_wo_audio_Dataset, GraphDataLoader],
        'KF_wo_visual': [KF_wo_visual, KF_wo_visual_Dataset, DataLoader],
        'KF_wo_SG':[KF_wo_SG,KF_wo_SG_Dataset,DataLoader],
        'oh_user':[oh_user,oh_user_Dataset,DataLoader],
        'oh_audio': [oh_audio, oh_audio_Dataset, DataLoader],
        'oh_text': [oh_text, oh_text_Dataset, DataLoader],
        'oh_visual': [oh_visual, oh_visual_Dataset, GraphDataLoader],
        'FakingRecipe': [FakingRecipe, FakingRecipe_Dataset, DataLoader]
    }

    # Check if the model name is valid
    if model_name not in model_map:
        print("The model does not exist")
        return None

    # Retrieve the model, dataset, and dataloader class based on the model name
    model_class, dataset_class, dataloader_class = model_map[model_name]

    # Create datasets
    train_dataset = dataset_class(dataset, type='train')
    val_dataset = dataset_class(dataset, type='val')
    test_dataset = dataset_class(dataset, type='test')

    # Create dataloaders
    if model_name == 'FakingRecipe':
        # Special handling for FakingRecipe (requires collate_fn)
        batch_size=FakingRecipe_config.batch_size

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn_FakeingRecipe)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn_FakeingRecipe)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_fn_FakeingRecipe)
        model = model_class(dataset).to(base_config.device)

    else:
        # For other models, use the regular dataloader
        batch_size = KDSGAT_FNVD_config.batch_size
        train_dataloader = dataloader_class(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = dataloader_class(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = dataloader_class(test_dataset, batch_size=batch_size, shuffle=False)
        model = model_class().to(base_config.device)

    return model, train_dataloader, val_dataloader, test_dataloader
    

