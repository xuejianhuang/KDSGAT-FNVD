import torch
import numpy as np
from einops import rearrange
import torch.nn as nn
import torch.nn.init as init
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
import torch as th
import json
from config import FakingRecipe_config
class SGATConv(nn.Module):
    def __init__(
            self,
            in_feats,
            edge_feats,
            out_feats,
            num_heads,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=True,
            activation=None,
            allow_zero_in_degree=False,
            bias=True,
    ):
        super(SGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        # Initialize linear layers based on input features type
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        # Initialize attention parameters
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Initialize bias
        self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,))) if bias else None

        # Initialize residual connection
        self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False) if residual else None

        # Initialize edge feature layers
        self._edge_feats = edge_feats
        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=False)
        self.attn_edge = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError("There are 0-in-degree nodes in the graph, output for those nodes will be invalid.")

            # Process source and destination features
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            # Linearly transform the edge features
            n_edges = edge_feat.shape[:-1]
            feat_edge = self.fc_edge(edge_feat).view(*n_edges, self._num_heads, self._out_feats)

            # Add edge features to graph
            graph.edata["ft_edge"] = feat_edge

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            # Calculate scalar for each edge
            ee = (feat_edge * self.attn_edge).sum(dim=-1).unsqueeze(-1)
            graph.edata["ee"] = ee

            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # Compute edge attention
            graph.apply_edges(fn.u_add_v("el", "er", "e_tmp"))

            # Combine attention weights of source and destination node
            graph.edata["e"] = graph.edata["e_tmp"] + graph.edata["ee"]

            # Create new edge features combining source node features and edge features
            graph.apply_edges(fn.u_add_e("ft", "ft_edge", "ft_combined"))

            e = self.leaky_relu(graph.edata.pop("e"))
            # Compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # Multiply combined features by attention coefficients
            graph.edata["m_combined"] = graph.edata["ft_combined"] * graph.edata["a"]

            # Copy edge features and sum them up
            graph.update_all(fn.copy_e("m_combined", "m"), fn.sum("m", "ft"))

            rst = graph.dstdata["ft"]
            # Apply residual connection
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst += resval
            # Add bias
            if self.bias is not None:
                rst += self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # Apply activation function
            if self.activation:
                rst = self.activation(rst)

            return (rst, graph.edata["a"]) if get_attention else rst

class SelfAttention(nn.Module):
    """
    Self-Attention mechanism using nn.MultiheadAttention.
    This computes self-attention on a single input sequence.
    """

    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):
        """
        Initialize the Self-Attention layer.

        Args:
            input_dim (int): The dimension of the input sequence (e.g., text features).
            hidden_dim (int): The hidden dimension for attention computation.
            num_heads (int): The number of attention heads.
            dropout (float): Dropout probability for regularization.
        """
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim

        # Linear projection layer for the input sequence
        self.fc = nn.Linear(input_dim, hidden_dim)

        # MultiheadAttention layer
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Final output layer to combine attended features
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Forward pass through the self-attention layer.

        Args:
            x (torch.Tensor): The input sequence of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The attended feature representation.
        """

        # Apply linear projection to input to match the attention dimension
        q = self.fc(x)  # (batch_size, seq_len, hidden_dim)

        # Prepare input for MultiheadAttention: (seq_len, batch_size, hidden_dim)
        q = q.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)

        # Self-attention: Attention over the same sequence
        attn_output, _ = self.attn(q, q, q)  # Self-attention on the input

        # Apply the final output layer
        output = self.fc_out(attn_output.permute(1, 0, 2))  # Back to (batch_size, seq_len, hidden_dim)

        # Apply dropout for regularization
        output = self.dropout(output)

        return output

class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism using nn.MultiheadAttention.
    This computes cross-attention between two sequences.
    """

    def __init__(self, input_dim1, input_dim2, hidden_dim, num_heads=8, dropout=0.1):
        """
        Initialize the Cross-Attention layer.

        Args:
            input_dim1 (int): The dimension of the first input sequence (e.g., text features).
            input_dim2 (int): The dimension of the second input sequence (e.g., image features).
            hidden_dim (int): The hidden dimension for attention computation.
            num_heads (int): The number of attention heads.
            dropout (float): Dropout probability for regularization.
        """
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim

        # Linear projection layers for both input sequences
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)

        # MultiheadAttention layer
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Final output layer to combine attended features
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x1, x2):
        """
        Forward pass through the cross-attention layer.

        Args:
            x1 (torch.Tensor): The first input sequence of shape (batch_size, seq_len1, input_dim1).
            x2 (torch.Tensor): The second input sequence of shape (batch_size, seq_len2, input_dim2).

        Returns:
            torch.Tensor: The attended feature representation.
        """

        # Apply linear projections to both inputs to match the attention dimension
        q1 = self.fc1(x1)  # (batch_size, seq_len1, hidden_dim)
        q2 = self.fc2(x2)  # (batch_size, seq_len2, hidden_dim)

        # Prepare input for MultiheadAttention: (seq_len, batch_size, hidden_dim)
        q1 = q1.permute(1, 0, 2)  # (seq_len1, batch_size, hidden_dim)
        q2 = q2.permute(1, 0, 2)  # (seq_len2, batch_size, hidden_dim)

        # Cross-attention between x1 and x2 (attention from x1 to x2)
        attn_output, _ = self.attn(q1, q2, q2)  # Attention from x1 to x2

        # Apply the final output layer
        output = self.fc_out(attn_output.permute(1, 0, 2))  # Back to (batch_size, seq_len1, hidden_dim)

        # Apply dropout for regularization
        output = self.dropout(output)

        return output

class CoAttention(nn.Module):
    """
    Co-attention mechanism using MultiheadAttention to model interactions between two sequences.
    """

    def __init__(self, input_dim1, input_dim2, hidden_dim, num_heads=8, dropout=0.1):
        """
        Initialize the Co-Attention layer using MultiheadAttention.

        Args:
            input_dim1 (int): The dimension of the first input sequence (e.g., text features).
            input_dim2 (int): The dimension of the second input sequence (e.g., image features).
            hidden_dim (int): The hidden dimension for attention computation.
            num_heads (int): The number of attention heads.
            dropout (float): Dropout probability for regularization.
        """
        super(CoAttention, self).__init__()

        self.hidden_dim = hidden_dim

        # Linear projection layers for both inputs
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)

        # MultiheadAttention layers
        self.attn1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        """
        Forward pass through the co-attention layer.

        Args:
            x1 (torch.Tensor): The first input sequence of shape (batch_size, seq_len1, input_dim1).
            x2 (torch.Tensor): The second input sequence of shape (batch_size, seq_len2, input_dim2).

        Returns:
            torch.Tensor: The attended feature representation for both sequences.
        """

        # Apply linear projections to both inputs to match the attention dimension
        q1 = self.fc1(x1)  # (batch_size, seq_len1, hidden_dim)
        q2 = self.fc2(x2)  # (batch_size, seq_len2, hidden_dim)

        # Prepare input for MultiheadAttention: (seq_len, batch_size, hidden_dim)
        q1 = q1.permute(1, 0, 2)  # (seq_len1, batch_size, hidden_dim)
        q2 = q2.permute(1, 0, 2)  # (seq_len2, batch_size, hidden_dim)

        # Cross-attention between x1 and x2 (attention from x1 to x2 and vice versa)
        cross_attn1, _ = self.attn1(q1, q2, q2)  # Attention from x1 to x2
        cross_attn2, _ = self.attn2(q2, q1, q1)  # Attention from x2 to x1

        # Permute back to (batch_size, seq_len, hidden_dim) for both attentions
        cross_attn1 = cross_attn1.permute(1, 0, 2)  # (batch_size, seq_len1, hidden_dim)
        cross_attn2 = cross_attn2.permute(1, 0, 2)  # (batch_size, seq_len2, hidden_dim)

        # Apply dropout for regularization on both attention results
        cross_attn1 = self.dropout(cross_attn1)
        cross_attn2 = self.dropout(cross_attn2)

        return cross_attn1, cross_attn2

class Attention(nn.Module):
    def __init__(self, dim, heads = 2, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear(d_model, d_k * n_heads)
        self.w_k = Linear(d_model, d_k * n_heads)
        self.w_v = Linear(d_model, d_v * n_heads)

    def forward(self, q, k, v):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        return q_s, k_s, v_s

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn_v = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.multihead_attn_s = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_emb_v = PosEncoding(visual_len * 10, d_model)
        self.pos_emb_s = PosEncoding(sen_len * 10, d_model)
        self.linear_v = nn.Linear(in_features=fea_v, out_features=d_model)
        self.linear_s = nn.Linear(in_features=fea_s, out_features=d_model)
        self.proj_v = Linear(n_heads * d_v, d_model)
        self.proj_s = Linear(n_heads * d_v, d_model)
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_v = LayerNormalization(d_model)
        self.layer_norm_s = LayerNormalization(d_model)
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.pos = pos

    def forward(self, v, s, v_len, s_len):
        b_size = v.size(0)
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        v, s = self.linear_v(v), self.linear_s(s)
        if self.pos:
            pos_v, pos_s = self.pos_emb_v(v_len), self.pos_emb_s(s_len)
            residual_v, residual_s = v + pos_v, s + pos_s
        else:
            residual_v, residual_s = v, s
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        q_v, k_v, v_v = self.multihead_attn_v(v, v, v)
        q_s, k_s, v_s = self.multihead_attn_s(s, s, s)
        context_v, attn_v = self.attention(q_v, k_s, v_s)
        context_s, attn_s = self.attention(q_s, k_v, v_v)
        context_v = context_v.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        context_s = context_s.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output_v = self.dropout(self.proj_v(context_v))
        output_s = self.dropout(self.proj_s(context_s))
        return self.layer_norm_v(residual_v + output_v), self.layer_norm_s(residual_s + output_s)

class co_attention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(co_attention, self).__init__()
        self.multi_head = MultiHeadAttention(d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout, d_model=d_model,
                                             visual_len=visual_len, sen_len=sen_len, fea_v=fea_v, fea_s=fea_s, pos=pos)
        self.PoswiseFeedForwardNet_v = PoswiseFeedForwardNet(d_model=d_model, d_ff=128, dropout=dropout)
        self.PoswiseFeedForwardNet_s = PoswiseFeedForwardNet(d_model=d_model, d_ff=128,dropout=dropout)
    def forward(self, v, s, v_len, s_len):
        v, s = self.multi_head(v, s, v_len, s_len)
        v = self.PoswiseFeedForwardNet_v(v)
        s = self.PoswiseFeedForwardNet_s(s)
        return v, s

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class PosEncoding_fix(nn.Module):
    def __init__(self, d_word_vec):
        super(PosEncoding_fix, self).__init__()
        self.d_word_vec = d_word_vec
        self.w_k = np.array([1 / (np.power(10000, 2 * (i // 2) / d_word_vec)) for i in range(d_word_vec)])

    def forward(self, inputs):

        pos_embs = []
        for pos in inputs:
            pos_emb = torch.tensor([self.w_k[i] * pos.cpu() for i in range(self.d_word_vec)])
            if pos != 0:
                pos_emb[0::2] = np.sin(pos_emb[0::2])
                pos_emb[1::2] = np.cos(pos_emb[1::2])
                pos_embs.append(pos_emb)
            else:
                pos_embs.append(torch.zeros(self.d_word_vec))
        pos_embs = torch.stack(pos_embs)
        return pos_embs.cuda()

class DurationEncoding(nn.Module):
    def __init__(self, dim, dataset):
        super(DurationEncoding, self).__init__()
        if dataset == 'FakeTT':
            # './fea/fakett/fakett_segment_duration.json' record the duration of each clip(segment) for each video
            with open(f'{FakingRecipe_config.dataset_dir}/fakett/fakett_segment_duration.json', 'r') as json_file:
                seg_dura_info = json.load(json_file)
        elif dataset == 'FakeSV':
            # './fea/fakesv/fakesv_segment_duration.json' record the duration of each clip(segment) for each video
            with open(f'{FakingRecipe_config.dataset_dir}/fakesv/fakesv_segment_duration.json', 'r') as json_file:
                seg_dura_info = json.load(json_file)

        self.all_seg_duration = seg_dura_info['all_seg_duration']
        self.all_seg_dura_ratio = seg_dura_info['all_seg_dura_ratio']
        self.absolute_bin_edges = torch.quantile(torch.tensor(self.all_seg_duration).to(torch.float64),
                                                 torch.arange(0, 1, 0.01).to(torch.float64)).cuda()
        self.relative_bin_edges = torch.quantile(torch.tensor(self.all_seg_dura_ratio).to(torch.float64),
                                                 torch.arange(0, 1, 0.02).to(torch.float64)).cuda()
        self.ab_duration_embed = torch.nn.Embedding(101, dim)
        self.re_duration_embed = torch.nn.Embedding(51, dim)

        self.ocr_all_seg_duration = seg_dura_info['ocr_all_seg_duration']
        self.ocr_all_seg_dura_ratio = seg_dura_info['ocr_all_seg_dura_ratio']
        self.ocr_absolute_bin_edges = torch.quantile(torch.tensor(self.ocr_all_seg_duration).to(torch.float64),
                                                     torch.arange(0, 1, 0.01).to(torch.float64)).cuda()
        self.ocr_relative_bin_edges = torch.quantile(torch.tensor(self.ocr_all_seg_dura_ratio).to(torch.float64),
                                                     torch.arange(0, 1, 0.02).to(torch.float64)).cuda()
        self.ocr_ab_duration_embed = torch.nn.Embedding(101, dim)
        self.ocr_re_duration_embed = torch.nn.Embedding(51, dim)

        self.result_dim = dim

    def forward(self, time_value, attribute):
        all_segs_embedding = []
        if attribute == 'natural_ab':
            for dv in time_value:
                bucket_indice = torch.searchsorted(self.absolute_bin_edges, torch.tensor(dv, dtype=torch.float64))
                dura_embedding = self.ab_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
        elif attribute == 'natural_re':
            for dv in time_value:
                bucket_indice = torch.searchsorted(self.relative_bin_edges, torch.tensor(dv, dtype=torch.float64))
                dura_embedding = self.re_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
        elif attribute == 'ocr_ab':
            for dv in time_value:
                bucket_indice = torch.searchsorted(self.ocr_absolute_bin_edges, torch.tensor(dv, dtype=torch.float64))
                dura_embedding = self.ocr_ab_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)

        elif attribute == 'ocr_re':
            for dv in time_value:
                bucket_indice = torch.searchsorted(self.ocr_relative_bin_edges, torch.tensor(dv, dtype=torch.float64))
                dura_embedding = self.ocr_re_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)

        if len(all_segs_embedding) == 0:
            return torch.zeros((1, self.result_dim)).cuda()
        return torch.stack(all_segs_embedding, dim=0).cuda()

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(self.softmax(scores))

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out

class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
            for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)

        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)
        self.max_len = int(max_seq_len/10)
    def forward(self, input_len):
        max_len = self.max_len            # torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = tensor([list(range(1, len+1)) + [0]*(max_len-len) for len in input_len])
        return self.pos_enc(input_pos)
