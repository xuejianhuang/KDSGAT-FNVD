import torch

class base_config():
    # Directories for datasets and model saving
    FakeSV_dataset_dir = './data/FakeSV/'
    FakeTT_dataset_dir = './data/FakeTT/'
    model_saved_path = './best_model/'

    data_json_path = "data.json"
    train_dataset = 'data_split/vid_time3_train.txt'
    val_dataset = 'data_split/vid_time3_val.txt'
    test_dataset = 'data_split/vid_time3_test.txt'

    # Device configuration for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of output classes for the classifier
    num_classes = 2

    lr = 5e-5  # Learning rate
    decayRate = 0.96  # Learning rate decay rate
    epoch = 20  # Number of training epochs
    patience = 3  # Patience for early stopping



class KDSGAT_FNVD_config(base_config):
    bert_dir = './bert-base-multilingual-uncased/'
    # SGATConv
    SGAT_node_feats = 768
    SGAT_edge_feats = 768
    SGAT_out_feats = 128
    SGAT_num_heads = 2
    SGAT_n_layers = 2

    text_max_length = 200
    author_intro_max_length = 50
    keyframes_max_num = 10

    text_dim = 768  # Dimension of text embeddings
    img_dim = 768  # Dimension of image embeddings
    audio_dim = 1024

    fea_dim = 128

    classifier_hidden_dim = 512  # Hidden dimension size for the classifier
    # Attention mechanism parameters
    att_num_heads = 2  # Number of attention heads
    att_dropout = 0.1  # Dropout rate for attention layers

    f_dropout = 0.1  # Dropout rate for fully connected layers

    # Training parameters
    seed = 2024
    batch_size = 32  # Batch size for training



    # Feature paths
    audio_fea_path = 'audio_fea/'
    keyframes_fea_path = 'keyframes_fea/'
    scene_graph_path = 'scene_graph/'
    text_fea_path='text_fea/'
    user_fea_path='user_fea/'


class FakingRecipe_config(base_config):
    dataset_dir="./data/baseline/fea/"
    FakeSV_metainfo_path=f'{dataset_dir}/fakesv/metainfo.json'
    FakeTT_metainfo_path=f'{dataset_dir}/fakett/metainfo.json'
    FakeSV_ocr_pattern_fea_path=f'{dataset_dir}/fakesv/preprocess_ocr/sam'
    FakeTT_ocr_pattern_fea_path=f'{dataset_dir}/fakett/preprocess_ocr/sam'

    # Training parameters
    seed = 2023
    batch_size = 128  # Batch size for training


