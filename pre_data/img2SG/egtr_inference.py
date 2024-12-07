from glob import glob

import torch
import json
from PIL import Image

from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from PIL import Image, ImageDraw, ImageFont


def visualization(img, bboxes, obj_list):
    colors = ['red', 'blue', 'orange', 'green', 'yellow', 'purple']
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    i = 0
    for bbox, obj in zip(bboxes, obj_list):
        x_center, y_center, width, height = bbox
        x_max = int((x_center + width/2) * img.width)
        y_max = int((y_center + height/2) * img.height)
        x_min = int((x_center - width/2) * img.width)
        y_min = int((y_center - height/2) * img.height)
        text_width, text_height = 10, 10
        text_x = max(x_min + text_width + 5, 0)
        text_y = y_min + text_height + 5
        draw.text((text_x, text_y), str(obj), fill=colors[i], font=font)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=colors[i], fill=None, width=3)
        i = (i + 1)%6
    img.show()

if __name__ == '__main__':
    # config
    architecture = "./SenseTime/deformable-detr/oi"
    min_size = 800
    max_size = 1333
    artifact_path = "artifact/oi"

    # 读取object类别、关系类别文件
    with open('obj_rel.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        id2obj=data['obj']
        id2rel=data['rel']

    # feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        architecture, size=min_size, max_size=max_size
    )

    # inference image
    input_image = Image.open("../../data/Twitter/test/img/7.jpg")
    image = feature_extractor(input_image, return_tensors="pt")

    # model
    config = DeformableDetrConfig.from_pretrained(artifact_path)
    model = DetrForSceneGraphGeneration.from_pretrained(
        architecture, config=config, ignore_mismatched_sizes=True
    )
    ckpt_path = sorted(
        glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )[-1]
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k[6:]] = state_dict.pop(k)  # "model."

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    # output
    outputs = model(
        pixel_values=image['pixel_values'].cuda(),
        pixel_mask=image['pixel_mask'].cuda(),
        output_attention_states=True
    )

    pred_logits = outputs['logits'][0]
    obj_scores, pred_classes = torch.max(pred_logits.softmax(-1), -1)
    pred_boxes = outputs['pred_boxes'][0]

    pred_connectivity = outputs['pred_connectivity'][0]
    pred_rel = outputs['pred_rel'][0]
    pred_rel = torch.mul(pred_rel, pred_connectivity)

    # get valid objects and triplets
    obj_threshold = 0.4
    valid_obj_indices = (obj_scores >= obj_threshold).nonzero()[:, 0]
    obj_indices = pred_classes[valid_obj_indices] # [num_valid_objects]

    valid_obj_classes=[id2obj[i] for i in obj_indices if i < len(id2obj)]
    print(valid_obj_classes)
    valid_obj_boxes = pred_boxes[valid_obj_indices] # [num_valid_objects, 4]
    print(valid_obj_boxes)

    visualization(input_image,valid_obj_boxes,valid_obj_classes)

    rel_threshold = 1e-1
    valid_triplets = (pred_rel[valid_obj_indices][:, valid_obj_indices] >= rel_threshold).nonzero() # [num_valid_triplets, 3]
    relation_list=[]
    for i,j,rel_indices in valid_triplets:
        sub=valid_obj_classes[i]
        obj=valid_obj_classes[j]
        rel=id2rel[rel_indices-1]
        relation_list.append([sub,rel,obj ])

    print(relation_list)