##The paper "Knowledge-Enhanced Dynamic Scene Graph Attention Network for Fake News Video Detection" has been accepted by IEEE TMM
## Datasets
### FakeSV
FakeSV is the largest publicly available Chinese dataset for fake news detection on short video platforms. It includes samples collected from Douyin and Kuaishou, two widely-used Chinese short video platforms. For more details, please visit [this repo](https://github.com/ICTMCG/FakeSV).
### FakeTT
FakeTT is the latest publicly available English dataset for fake news detection on short video platforms. The data is sourced from TikTok, a globally popular short video platform. For more information, please refer to [this repo](https://github.com/ICTMCG/FakingRecipe).

## Data Preprocess
To facilitate reproduction, we provide preprocessed features, which you can download from [this link](https://pan.baidu.com/s/1VtoyVtSPrcrSF9BTljOqdA?pwd=qub5)(pwd: qub5). Additionally, we offer [checkpoints](https://pan.baidu.com/s/1sJZznacbG_8WYFCOkxXqzg?pwd=49dc) (pwd: 49dc) for two datasets.

## Quick Start
You can train and test KDSGAT-FNVD using the following code:
### train
 ```
  # Train the examples from FakeSV
    python main.py  --dataset FakeSV  --model KDSGAT-FNVD --mode train

  # Train the examples from FakeTT
    python main.py  --dataset FakeTT  --model KDSGAT-FNVD --mode train
 ```

### test
 ```
  # Train the examples from FakeSV
    python main.py  --dataset FakeSV  --model KDSGAT-FNVD --mode test

  # Train the examples from FakeTT
    python main.py  --dataset FakeTT  --model KDSGAT-FNVD --mode test
 ```

## Dependencies
* matplotlib
* librosa==0.9.1
* pytorch_lightning==1.9.0
* pandas==1.1.5
* numpy==1.21.0
* tqdm==4.63.1
* wordcloud==1.8.1
* einops==0.8.0
* scikit_learn==0.24.2
* moviepy==1.0.3
* Cython==0.29.28
* huggingface_hub==0.23.2
* pycocotools
* packaging==21.3
* torchvision==0.13.1+cu113
* SPARQLWrapper==2.0.0
* Pillow==8.4.0
* opencv_python==4.6.0
* jieba==0.42.1
* transformers==4.41.2
* dgl_cu111==0.6.1
* torch==1.12.1+cu113
* torchmetrics==1.4.0.post0
* requests==2.27.1

## Acknowledgements
Thank you to **Peng Qi** (National University of Singapore, Singapore), **Yuyan Bu** (Institute of Computing Technology, Chinese Academy of Sciences University of Chinese Academy of Sciences Beijing, China), **Juan Cao** (Institute of Computing Technology, Chinese Academy of Sciences University of Chinese Academy of Sciences Beijing, China) for providing the datasets and baseline models.


