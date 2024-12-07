import json
import os
import dgl
import torch as th
from dgl import save_graphs
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('../bert-base-multilingual-uncased')
model = BertModel.from_pretrained('../bert-base-multilingual-uncased')

def get_bert_embeddings(text_list):
    """
    Get BERT embeddings for a list of texts.
    :param text_list: List of texts
    :return: BERT embeddings of the texts (tensor)
    """
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
    with th.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation

def process_edges_and_nodes(edges):
    """
    Parse edge relations and extract source nodes, target nodes, relations, and isolated nodes.
    :param edges: Text representation of edges
    :return: Lists of source nodes, target nodes, relations, and isolated nodes
    """

    src_nodes, dst_nodes, rels, isolated_nodes = [], [], [], []
    for edge in edges:
        edge = edge.strip('( ,')
        parts = [part.strip() for part in edge.split(', ')]
        if len(parts) == 3:  # 确保边格式正确
            src, rel, dst = parts
            src_nodes.append(src.strip())
            dst_nodes.append(dst.strip())
            rels.append(rel.strip())
        else:  # 孤立节点
            isolated_nodes.extend(parts)

    return src_nodes, dst_nodes, rels, isolated_nodes



def convert_to_dgl_graph(sg_data, save_path, max_sg_count=10, knowledge_distillation=False, knowledge=None):
    """
    Convert scene graph data to DGL graph format and save.
    :param sg_data: Scene graph data
    :param save_path: Path to save the graph
    :param max_sg_count: Maximum number of scene graphs to process per video_id
    :param knowledge_distillation: Whether to enable knowledge distillation
    :param knowledge: Data required for knowledge distillation
    """

    for video_id, sg_list in tqdm(sg_data.items(), desc="Processing graphs"):
        graphs = []

        # Limit to max_sg_count scene graphs, pad with isolated nodes if needed
        if len(sg_list) < max_sg_count:
            sg_list += ["(NULL)"] * (max_sg_count - len(sg_list))
        else:
            sg_list = sg_list[:max_sg_count]

        for sg_text in sg_list:
            text_edges = sg_text.split(')')[:-1]
            src_nodes, dst_nodes, rels, isolated_nodes = process_edges_and_nodes(text_edges)

            # Knowledge distillation handling
            if knowledge_distillation and video_id in knowledge:
                for node in set(src_nodes + dst_nodes + isolated_nodes):
                    entity_uri = knowledge[video_id]["entity_uri"].get(node)
                    if entity_uri:
                        concepts = knowledge[video_id]['concepts'].get(entity_uri, [])
                        for concept in concepts:
                            src_nodes.append(node)
                            dst_nodes.append(concept)
                            rels.append("is")
            # Construct node and edge lists
            all_nodes = list(set(src_nodes + dst_nodes + isolated_nodes))
            src_node_ids = [all_nodes.index(src) for src in src_nodes]
            dst_node_ids = [all_nodes.index(dst) for dst in dst_nodes]
            edges = th.tensor(src_node_ids, dtype=th.int32), th.tensor(dst_node_ids, dtype=th.int32)

            # Create DGL graph
            dgl_graph = dgl.graph(edges, idtype=th.int32, num_nodes=len(all_nodes))
            dgl_graph.ndata['x'] = get_bert_embeddings(all_nodes)  # Node features
            if rels:
                dgl_graph.edata['x'] = get_bert_embeddings(rels)  #Edge features

            graphs.append(dgl_graph)
            # Save DGL graph
            video_save_path = os.path.join(save_path, f"{video_id}.bin")
            save_graphs(video_save_path, graphs)

if __name__ == '__main__':

    base_path = '../data/FakeTT/scene_graph'

    with open(os.path.join(base_path, 'SG.json'), encoding="utf-8") as f:
        sg_data = json.load(f)

    # Load knowledge distillation data (optional)
    knowledge_distillation = False
    knowledge = {}
    if knowledge_distillation:
        with open(os.path.join(base_path, 'knowledge_distillation.json'), encoding="utf-8") as f:
            knowledge = json.load(f)

    # Convert scene graphs to DGL format, specifying the maximum number of scene graphs
    max_sg_count = 10
    convert_to_dgl_graph(sg_data, base_path, max_sg_count=max_sg_count, knowledge_distillation=knowledge_distillation,
                         knowledge=knowledge)
