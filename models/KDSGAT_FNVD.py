import torch
import torch.nn as nn
import dgl
from config import KDSGAT_FNVD_config as config
from models.layers import SGATConv,SelfAttention, CoAttention
from transformers import BertModel
from dgl.nn import GATConv

class KDSGAT_FNVD(nn.Module):
    def __init__(self, SGAT_node_feats=config.SGAT_node_feats, SGAT_edge_feats=config.SGAT_edge_feats, SGAT_out_feats=config.SGAT_out_feats,
                 SGAT_num_heads=config.SGAT_num_heads, SGAT_n_layers=config.SGAT_n_layers, residual=True):
        super(KDSGAT_FNVD, self).__init__()

        self.linear_user = torch.nn.Linear(config.text_dim, config.fea_dim)
        self.linear_audio = torch.nn.Linear(config.audio_dim, config.fea_dim)
        # Initialize SGAT layers
        self.SGAT_layers = nn.ModuleList([
            SGATConv(
                in_feats=SGAT_node_feats if i == 0 else SGAT_out_feats,
                edge_feats=SGAT_edge_feats,
                out_feats=SGAT_out_feats,
                num_heads=SGAT_num_heads,
                residual=residual,
                allow_zero_in_degree=True
            ) for i in range(SGAT_n_layers)
        ])

        self.bert= BertModel.from_pretrained(config.bert_dir)

        # Attention layers
        self.self_attention_k = SelfAttention(config.img_dim, config.fea_dim)  # for keyframes
        self.self_attention_s = SelfAttention(SGAT_out_feats, config.fea_dim)    # for scene graph

        # Co-attention layers for keyframes & scene graph
        self.co_attention_sk = CoAttention(
            input_dim1=config.fea_dim,
            input_dim2=config.fea_dim,
            hidden_dim=config.fea_dim,
            num_heads=config.att_num_heads,
            dropout=config.att_dropout
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.text_dim+config.fea_dim*3, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

    def _process_scene_graph(self, graph):
        """
        Process a single scene graph through SGAT layers.
        """
        node_feats = graph.ndata['x']
        for layer in self.SGAT_layers:
            node_feats = layer(
                graph,
                node_feats,
                graph.edata.get('x', torch.zeros((graph.num_edges(), config.SGAT_edge_feats)).to(config.device))
            ).mean(dim=1)
        graph.ndata['h'] = node_feats
        return dgl.readout_nodes(graph, 'h', op='mean')

    def forward(self, data):

        text_tokens = data['text_tokens']   #shape=[batch_size,max_len,768]
        audio_fea = data['audio_fea']       #shape=[batch_size,49,1024]
        scene_graph = data['scene_graph']  #shape=[batch_size,10]
        keyframes_fea = data['keyframes_fea']  #shape=[batch_size,10,768]
        author_intro_token = data['author_intro_token']

        text_fea=self.bert(**text_tokens).last_hidden_state[:,0,:]
        user_fea=self.bert(**author_intro_token).last_hidden_state[:,0,:]
        user_fea=self.linear_user(user_fea)
        audio_fea = self.linear_audio(audio_fea.mean(dim=1))

        scene_graph_fea = torch.stack([self._process_scene_graph(graph) for graph in scene_graph], dim=1) #shape=[batch_size,10,128]

        h_s = self.self_attention_s(scene_graph_fea)
        h_k = self.self_attention_k(keyframes_fea)
        co_sk,co_ks=self.co_attention_sk(h_s, h_k)
        co_sk=co_sk.mean(dim=1)
        co_ks=co_ks.mean(dim=1)
        ks_fea=torch.stack([co_sk,co_ks],dim=1).mean(dim=1)

        fused_features=torch.cat((text_fea,audio_fea,ks_fea,user_fea),dim=1)

        logits = self.classifier(fused_features)
        return logits

class KF_wo_user(nn.Module):
    def __init__(self, SGAT_node_feats=config.SGAT_node_feats, SGAT_edge_feats=config.SGAT_edge_feats, SGAT_out_feats=config.SGAT_out_feats,
                 SGAT_num_heads=config.SGAT_num_heads, SGAT_n_layers=config.SGAT_n_layers, residual=True):
        super(KF_wo_user, self).__init__()

        self.linear_audio = torch.nn.Linear(config.audio_dim, config.fea_dim)
        # Initialize SGAT layers
        self.SGAT_layers = nn.ModuleList([
            SGATConv(
                in_feats=SGAT_node_feats if i == 0 else SGAT_out_feats,
                edge_feats=SGAT_edge_feats,
                out_feats=SGAT_out_feats,
                num_heads=SGAT_num_heads,
                residual=residual,
                allow_zero_in_degree=True
            ) for i in range(SGAT_n_layers)
        ])

        self.bert= BertModel.from_pretrained(config.bert_dir)

        # Attention layers
        self.self_attention_k = SelfAttention(config.img_dim, config.fea_dim)  # for keyframes
        self.self_attention_s = SelfAttention(SGAT_out_feats, config.fea_dim)    # for scene graph

        # Co-attention layers for keyframes & scene graph
        self.co_attention_sk = CoAttention(
            input_dim1=config.fea_dim,
            input_dim2=config.fea_dim,
            hidden_dim=config.fea_dim,
            num_heads=config.att_num_heads,
            dropout=config.att_dropout
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.text_dim+config.fea_dim*2, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

    def _process_scene_graph(self, graph):
        """
        Process a single scene graph through SGAT layers.
        """
        node_feats = graph.ndata['x']
        for layer in self.SGAT_layers:
            node_feats = layer(
                graph,
                node_feats,
                graph.edata.get('x', torch.zeros((graph.num_edges(), config.SGAT_edge_feats)).to(config.device))
            ).mean(dim=1)
        graph.ndata['h'] = node_feats
        return dgl.readout_nodes(graph, 'h', op='mean')

    def forward(self, data):

        text_tokens = data['text_tokens']   #shape=[batch_size,max_len,768]
        audio_fea = data['audio_fea']       #shape=[batch_size,49,1024]
        scene_graph = data['scene_graph']  #shape=[batch_size,10]
        keyframes_fea = data['keyframes_fea']  #shape=[batch_size,10,768]

        text_fea=self.bert(**text_tokens).last_hidden_state[:,0,:]
        audio_fea = self.linear_audio(audio_fea.mean(dim=1))

        scene_graph_fea = torch.stack([self._process_scene_graph(graph) for graph in scene_graph], dim=1) #shape=[batch_size,10,128]

        h_s = self.self_attention_s(scene_graph_fea)
        h_k = self.self_attention_k(keyframes_fea)
        co_sk,co_ks=self.co_attention_sk(h_s, h_k)
        co_sk=co_sk.mean(dim=1)
        co_ks=co_ks.mean(dim=1)
        ks_fea=torch.stack([co_sk,co_ks],dim=1).mean(dim=1)

        fused_features=torch.cat((text_fea,audio_fea,ks_fea),dim=1)

        logits = self.classifier(fused_features)
        return logits

class KF_wo_text(nn.Module):
    def __init__(self, SGAT_node_feats=config.SGAT_node_feats, SGAT_edge_feats=config.SGAT_edge_feats, SGAT_out_feats=config.SGAT_out_feats,
                 SGAT_num_heads=config.SGAT_num_heads, SGAT_n_layers=config.SGAT_n_layers, residual=True):
        super(KF_wo_text, self).__init__()

        self.linear_user = torch.nn.Linear(config.text_dim, config.fea_dim)
        self.linear_audio = torch.nn.Linear(config.audio_dim, config.fea_dim)
        # Initialize SGAT layers
        self.SGAT_layers = nn.ModuleList([
            SGATConv(
                in_feats=SGAT_node_feats if i == 0 else SGAT_out_feats,
                edge_feats=SGAT_edge_feats,
                out_feats=SGAT_out_feats,
                num_heads=SGAT_num_heads,
                residual=residual,
                allow_zero_in_degree=True
            ) for i in range(SGAT_n_layers)
        ])

        self.bert= BertModel.from_pretrained(config.bert_dir)

        # Attention layers
        self.self_attention_k = SelfAttention(config.img_dim, config.fea_dim)  # for keyframes
        self.self_attention_s = SelfAttention(SGAT_out_feats, config.fea_dim)    # for scene graph

        # Co-attention layers for keyframes & scene graph
        self.co_attention_sk = CoAttention(
            input_dim1=config.fea_dim,
            input_dim2=config.fea_dim,
            hidden_dim=config.fea_dim,
            num_heads=config.att_num_heads,
            dropout=config.att_dropout
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.fea_dim*3, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

    def _process_scene_graph(self, graph):
        """
        Process a single scene graph through SGAT layers.
        """
        node_feats = graph.ndata['x']
        for layer in self.SGAT_layers:
            node_feats = layer(
                graph,
                node_feats,
                graph.edata.get('x', torch.zeros((graph.num_edges(), config.SGAT_edge_feats)).to(config.device))
            ).mean(dim=1)
        graph.ndata['h'] = node_feats
        return dgl.readout_nodes(graph, 'h', op='mean')

    def forward(self, data):

        audio_fea = data['audio_fea']       #shape=[batch_size,49,1024]
        scene_graph = data['scene_graph']  #shape=[batch_size,10]
        keyframes_fea = data['keyframes_fea']  #shape=[batch_size,10,768]
        author_intro_token = data['author_intro_token']

        user_fea=self.bert(**author_intro_token).last_hidden_state[:,0,:]
        user_fea=self.linear_user(user_fea)
        audio_fea = self.linear_audio(audio_fea.mean(dim=1))

        scene_graph_fea = torch.stack([self._process_scene_graph(graph) for graph in scene_graph], dim=1) #shape=[batch_size,10,128]

        h_s = self.self_attention_s(scene_graph_fea)
        h_k = self.self_attention_k(keyframes_fea)
        co_sk,co_ks=self.co_attention_sk(h_s, h_k)
        co_sk=co_sk.mean(dim=1)
        co_ks=co_ks.mean(dim=1)
        ks_fea=torch.stack([co_sk,co_ks],dim=1).mean(dim=1)

        fused_features=torch.cat((audio_fea,ks_fea,user_fea),dim=1)

        logits = self.classifier(fused_features)
        return logits

class KF_wo_audio(nn.Module):
    def __init__(self, SGAT_node_feats=config.SGAT_node_feats, SGAT_edge_feats=config.SGAT_edge_feats, SGAT_out_feats=config.SGAT_out_feats,
                 SGAT_num_heads=config.SGAT_num_heads, SGAT_n_layers=config.SGAT_n_layers, residual=True):
        super(KF_wo_audio, self).__init__()

        self.linear_user = torch.nn.Linear(config.text_dim, config.fea_dim)
        # Initialize SGAT layers
        self.SGAT_layers = nn.ModuleList([
            SGATConv(
                in_feats=SGAT_node_feats if i == 0 else SGAT_out_feats,
                edge_feats=SGAT_edge_feats,
                out_feats=SGAT_out_feats,
                num_heads=SGAT_num_heads,
                residual=residual,
                allow_zero_in_degree=True
            ) for i in range(SGAT_n_layers)
        ])

        self.bert= BertModel.from_pretrained(config.bert_dir)

        # Attention layers
        self.self_attention_k = SelfAttention(config.img_dim, config.fea_dim)  # for keyframes
        self.self_attention_s = SelfAttention(SGAT_out_feats, config.fea_dim)    # for scene graph

        # Co-attention layers for keyframes & scene graph
        self.co_attention_sk = CoAttention(
            input_dim1=config.fea_dim,
            input_dim2=config.fea_dim,
            hidden_dim=config.fea_dim,
            num_heads=config.att_num_heads,
            dropout=config.att_dropout
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.text_dim+config.fea_dim*2, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

    def _process_scene_graph(self, graph):
        """
        Process a single scene graph through SGAT layers.
        """
        node_feats = graph.ndata['x']
        for layer in self.SGAT_layers:
            node_feats = layer(
                graph,
                node_feats,
                graph.edata.get('x', torch.zeros((graph.num_edges(), config.SGAT_edge_feats)).to(config.device))
            ).mean(dim=1)
        graph.ndata['h'] = node_feats
        return dgl.readout_nodes(graph, 'h', op='mean')

    def forward(self, data):

        text_tokens = data['text_tokens']   #shape=[batch_size,max_len,768]
        scene_graph = data['scene_graph']  #shape=[batch_size,10]
        keyframes_fea = data['keyframes_fea']  #shape=[batch_size,10,768]
        author_intro_token = data['author_intro_token']

        text_fea=self.bert(**text_tokens).last_hidden_state[:,0,:]
        user_fea=self.bert(**author_intro_token).last_hidden_state[:,0,:]
        user_fea=self.linear_user(user_fea)

        scene_graph_fea = torch.stack([self._process_scene_graph(graph) for graph in scene_graph], dim=1) #shape=[batch_size,10,128]

        h_s = self.self_attention_s(scene_graph_fea)
        h_k = self.self_attention_k(keyframes_fea)
        co_sk,co_ks=self.co_attention_sk(h_s, h_k)
        co_sk=co_sk.mean(dim=1)
        co_ks=co_ks.mean(dim=1)
        ks_fea=torch.stack([co_sk,co_ks],dim=1).mean(dim=1)

        fused_features=torch.cat((text_fea,ks_fea,user_fea),dim=1)

        logits = self.classifier(fused_features)
        return logits

#Remove all visual-related feature information,including the visual representations of key frames and the scene graph features
class KF_wo_visual(nn.Module):
    def __init__(self):
        super(KF_wo_visual, self).__init__()

        self.linear_user = torch.nn.Linear(config.text_dim, config.fea_dim)
        self.linear_audio = torch.nn.Linear(config.audio_dim, config.fea_dim)

        self.bert= BertModel.from_pretrained(config.bert_dir)



        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.text_dim+config.fea_dim*2, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )


    def forward(self, data):

        text_tokens = data['text_tokens']   #shape=[batch_size,max_len,768]
        audio_fea = data['audio_fea']       #shape=[batch_size,49,1024]
        author_intro_token = data['author_intro_token']

        text_fea=self.bert(**text_tokens).last_hidden_state[:,0,:]
        user_fea=self.bert(**author_intro_token).last_hidden_state[:,0,:]
        user_fea=self.linear_user(user_fea)
        audio_fea = self.linear_audio(audio_fea.mean(dim=1))



        fused_features=torch.cat((text_fea,audio_fea,user_fea),dim=1)

        logits = self.classifier(fused_features)
        return logits

#Remove the scene graph features while retaining the key frame features
class KF_wo_SG(nn.Module):
    def __init__(self, SGAT_node_feats=config.SGAT_node_feats, SGAT_edge_feats=config.SGAT_edge_feats, SGAT_out_feats=config.SGAT_out_feats,
                 SGAT_num_heads=config.SGAT_num_heads, SGAT_n_layers=config.SGAT_n_layers, residual=True):
        super(KF_wo_SG, self).__init__()

        self.linear_user = torch.nn.Linear(config.text_dim, config.fea_dim)
        self.linear_audio = torch.nn.Linear(config.audio_dim, config.fea_dim)


        self.bert= BertModel.from_pretrained(config.bert_dir)

        # Attention layers
        self.self_attention_k = SelfAttention(config.img_dim, config.fea_dim)  # for keyframes


        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.text_dim+config.fea_dim*3, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

    def forward(self, data):

        text_tokens = data['text_tokens']   #shape=[batch_size,max_len,768]
        audio_fea = data['audio_fea']       #shape=[batch_size,49,1024]
        keyframes_fea = data['keyframes_fea']  #shape=[batch_size,10,768]
        author_intro_token = data['author_intro_token']

        text_fea=self.bert(**text_tokens).last_hidden_state[:,0,:]
        user_fea=self.bert(**author_intro_token).last_hidden_state[:,0,:]
        user_fea=self.linear_user(user_fea)
        audio_fea = self.linear_audio(audio_fea.mean(dim=1))

        h_k = self.self_attention_k(keyframes_fea).mean(dim=1)


        fused_features=torch.cat((text_fea,audio_fea,h_k,user_fea),dim=1)

        logits = self.classifier(fused_features)
        return logits

class oh_user(nn.Module):
    def __init__(self):
        super(oh_user, self).__init__()
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.text_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )


    def forward(self, data):

        user_fea = data['user_fea']
        logits = self.classifier(user_fea)
        return logits

class oh_text(nn.Module):
    def __init__(self):
        super(oh_text, self).__init__()
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.text_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )


    def forward(self, data):

        text_fea = data['text_fea'][:,0,:]
        logits = self.classifier(text_fea)
        return logits

class oh_audio(nn.Module):
    def __init__(self):
        super(oh_audio, self).__init__()

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.audio_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

    def forward(self, data):

        audio_fea = data['audio_fea'].mean(dim=1)       #shape=[batch_size,49,1024]
        logits = self.classifier(audio_fea)
        return logits

class oh_visual(nn.Module):
    def __init__(self, SGAT_node_feats=config.SGAT_node_feats, SGAT_edge_feats=config.SGAT_edge_feats, SGAT_out_feats=config.SGAT_out_feats,
                 SGAT_num_heads=config.SGAT_num_heads, SGAT_n_layers=config.SGAT_n_layers, residual=True):
        super(oh_visual, self).__init__()

        # Initialize SGAT layers
        self.SGAT_layers = nn.ModuleList([
            SGATConv(
                in_feats=SGAT_node_feats if i == 0 else SGAT_out_feats,
                edge_feats=SGAT_edge_feats,
                out_feats=SGAT_out_feats,
                num_heads=SGAT_num_heads,
                residual=residual,
                allow_zero_in_degree=True
            ) for i in range(SGAT_n_layers)
        ])

        # Attention layers
        self.self_attention_k = SelfAttention(config.img_dim, config.fea_dim)  # for keyframes
        self.self_attention_s = SelfAttention(SGAT_out_feats, config.fea_dim)    # for scene graph

        # Co-attention layers for keyframes & scene graph
        self.co_attention_sk = CoAttention(
            input_dim1=config.fea_dim,
            input_dim2=config.fea_dim,
            hidden_dim=config.fea_dim,
            num_heads=config.att_num_heads,
            dropout=config.att_dropout
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.fea_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

    def _process_scene_graph(self, graph):
        """
        Process a single scene graph through SGAT layers.
        """
        node_feats = graph.ndata['x']
        for layer in self.SGAT_layers:
            node_feats = layer(
                graph,
                node_feats,
                graph.edata.get('x', torch.zeros((graph.num_edges(), config.SGAT_edge_feats)).to(config.device))
            ).mean(dim=1)
        graph.ndata['h'] = node_feats
        return dgl.readout_nodes(graph, 'h', op='mean')

    def forward(self, data):

        scene_graph = data['scene_graph']  #shape=[batch_size,10]
        keyframes_fea = data['keyframes_fea']  #shape=[batch_size,10,768]


        scene_graph_fea = torch.stack([self._process_scene_graph(graph) for graph in scene_graph], dim=1) #shape=[batch_size,10,128]

        h_s = self.self_attention_s(scene_graph_fea)
        h_k = self.self_attention_k(keyframes_fea)
        co_sk,co_ks=self.co_attention_sk(h_s, h_k)
        co_sk=co_sk.mean(dim=1)
        co_ks=co_ks.mean(dim=1)
        ks_fea=torch.stack([co_sk,co_ks],dim=1).mean(dim=1)

        logits = self.classifier(ks_fea)
        return logits

class KDGAT_FNVD(nn.Module):
    def __init__(self, GAT_node_feats=config.SGAT_node_feats, GAT_out_feats=config.SGAT_out_feats,
                 GAT_num_heads=config.SGAT_num_heads, GAT_n_layers=config.SGAT_n_layers):
        super(KDGAT_FNVD, self).__init__()

        self.linear_user = torch.nn.Linear(config.text_dim, config.fea_dim)
        self.linear_audio = torch.nn.Linear(config.audio_dim, config.fea_dim)

        self.GAT_layers = nn.ModuleList([
            GATConv(
                in_feats=GAT_node_feats if i == 0 else GAT_out_feats,
                out_feats=GAT_out_feats,
                num_heads=GAT_num_heads,
                allow_zero_in_degree=True
            ) for i in range(GAT_n_layers)
        ])

        self.bert= BertModel.from_pretrained(config.bert_dir)

        # Attention layers
        self.self_attention_k = SelfAttention(config.img_dim, config.fea_dim)  # for keyframes
        self.self_attention_s = SelfAttention(GAT_out_feats, config.fea_dim)    # for scene graph

        # Co-attention layers for keyframes & scene graph
        self.co_attention_sk = CoAttention(
            input_dim1=config.fea_dim,
            input_dim2=config.fea_dim,
            hidden_dim=config.fea_dim,
            num_heads=config.att_num_heads,
            dropout=config.att_dropout
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(config.text_dim+config.fea_dim*3, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

    def _process_scene_graph(self, graph):
        """
        Process a single scene graph through SGAT layers.
        """
        node_feats = graph.ndata['x']
        for layer in self.GAT_layers:
            node_feats = layer(
                graph,
                node_feats
            ).mean(dim=1)
        graph.ndata['h'] = node_feats
        return dgl.readout_nodes(graph, 'h', op='mean')

    def forward(self, data):

        text_tokens = data['text_tokens']   #shape=[batch_size,max_len,768]
        audio_fea = data['audio_fea']       #shape=[batch_size,49,1024]
        scene_graph = data['scene_graph']  #shape=[batch_size,10]
        keyframes_fea = data['keyframes_fea']  #shape=[batch_size,10,768]
        author_intro_token = data['author_intro_token']

        text_fea=self.bert(**text_tokens).last_hidden_state[:,0,:]
        user_fea=self.bert(**author_intro_token).last_hidden_state[:,0,:]
        user_fea=self.linear_user(user_fea)
        audio_fea = self.linear_audio(audio_fea.mean(dim=1))

        scene_graph_fea = torch.stack([self._process_scene_graph(graph) for graph in scene_graph], dim=1) #shape=[batch_size,10,128]

        h_s = self.self_attention_s(scene_graph_fea)
        h_k = self.self_attention_k(keyframes_fea)
        co_sk,co_ks=self.co_attention_sk(h_s, h_k)
        co_sk=co_sk.mean(dim=1)
        co_ks=co_ks.mean(dim=1)
        ks_fea=torch.stack([co_sk,co_ks],dim=1).mean(dim=1)

        fused_features=torch.cat((text_fea,audio_fea,ks_fea,user_fea),dim=1)

        logits = self.classifier(fused_features)
        return logits
        

