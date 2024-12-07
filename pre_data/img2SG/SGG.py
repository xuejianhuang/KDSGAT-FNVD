from glob import glob
import torch
import json
from PIL import Image, ImageDraw, ImageFont
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
import os
from tqdm import tqdm


def visualization(img, bboxes, obj_list):
    colors = ['red', 'blue', 'orange', 'green', 'yellow', 'purple']
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i, (bbox, obj) in enumerate(zip(bboxes, obj_list)):
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * img.width)
        y_min = int((y_center - height / 2) * img.height)
        x_max = int((x_center + width / 2) * img.width)
        y_max = int((y_center + height / 2) * img.height)

        draw.text((max(x_min + 15, 0), y_min + 15), str(obj), fill=colors[i % 6], font=font)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=colors[i % 6], width=3)

    return img


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_model(architecture, artifact_path):
    config = DeformableDetrConfig.from_pretrained(artifact_path)
    model = DetrForSceneGraphGeneration.from_pretrained(
        architecture, config=config, ignore_mismatched_sizes=True
    )

    ckpt_path = sorted(glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
                       key=lambda x: int(x.split("epoch=")[1].split("-")[0]))[-1]
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {k[6:]: v for k, v in state_dict.items()}  # Remove "model."

    model.load_state_dict(state_dict)
    model.cuda().eval()
    return model


if __name__ == '__main__':
    # Configuration
    architecture = "./SenseTime/deformable-detr/oi"
    artifact_path = "artifact/oi"
    images_directory = "../data/FakeSV/keyframes"
    sgg_output_path= '../data/FakeSV/keyframes/sg'

    # Load object and relationship categories
    data = load_json('obj_rel.json')
    id2obj, id2rel = data['obj'], data['rel']

    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(architecture, size=800, max_size=1333)

    # Initialize scene graphs storage
    scene_graphs = []

    # Load model
    model = load_model(architecture, artifact_path)

    # Get all image paths
    img_paths = glob(os.path.join(images_directory, "*.jpg"))

    # Process each image in the directory with a progress bar
    for img_path in tqdm(img_paths):
        input_image = Image.open(img_path).convert("RGB")  # Ensure RGB format
        image = feature_extractor(input_image, return_tensors="pt")

        # Output
        outputs = model(pixel_values=image['pixel_values'].cuda(), pixel_mask=image['pixel_mask'].cuda(),
                        output_attention_states=True)
        pred_logits = outputs['logits'][0]
        obj_scores, pred_classes = torch.max(pred_logits.softmax(-1), -1)
        pred_boxes = outputs['pred_boxes'][0]

        # Get valid objects
        obj_threshold = 0.3
        valid_obj_indices = (obj_scores >= obj_threshold).nonzero(as_tuple=True)[0]
        obj_indices = pred_classes[valid_obj_indices]
        valid_obj_classes = [id2obj[i] for i in obj_indices if i < len(id2obj)]
        valid_obj_boxes = pred_boxes[valid_obj_indices]

        # Visualization
        vis_image = visualization(input_image.copy(), valid_obj_boxes, valid_obj_classes)
        vis_image.save(f"{sgg_output_path}/visualization_{os.path.basename(img_path)}")  # Save visualization

        # Get valid triplets
        rel_threshold = 1e-1
        pred_rel = outputs['pred_rel'][0]
        pred_connectivity = outputs['pred_connectivity'][0]
        pred_rel *= pred_connectivity
        valid_triplets = (pred_rel[valid_obj_indices][:, valid_obj_indices] >= rel_threshold).nonzero(as_tuple=True)

        relation_list = [
            [valid_obj_classes[i], id2rel[rel_index - 1], valid_obj_classes[j]]
            for i, j, rel_index in zip(*valid_triplets)
        ]

        # Store scene graph information
        scene_graphs.append({
            "image": os.path.basename(img_path),
            "objects": valid_obj_classes,
            "boxes": valid_obj_boxes.tolist(),
            "relations": relation_list
        })
    # Save scene graphs to JSON
    with open(sgg_output_path+"/scene_graphs.json", 'w', encoding='utf-8') as f:
        json.dump(scene_graphs, f, ensure_ascii=False, indent=4)
