import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_dura_info_visual

from .layers import *

class MSAM(torch.nn.Module):
    def __init__(self,dataset):
        super(MSAM,self).__init__()
        if dataset=='FakeTT':
            self.encoded_text_semantic_fea_dim=512 
        elif dataset=='FakeSV':
            self.encoded_text_semantic_fea_dim=768 
        self.input_visual_frames=83

        self.mlp_text_emo = nn.Sequential(nn.Linear(768,128),nn.ReLU(),nn.Dropout(0.1))
        self.mlp_text_semantic = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim,128),nn.ReLU(),nn.Dropout(0.1)) 
        
        self.mlp_img = nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.1))
    
        self.mlp_audio = nn.Sequential(torch.nn.Linear(768, 128), torch.nn.ReLU(),nn.Dropout(0.1))
        
        self.co_attention_tv = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,
                                        visual_len=self.input_visual_frames, sen_len=512, fea_v=128, fea_s=128, pos=False)
        
        self.trm_emo=nn.TransformerEncoderLayer(d_model = 128, nhead = 2, batch_first = True)
        self.trm_semantic=nn.TransformerEncoderLayer(d_model = 128, nhead = 2, batch_first = True)
        self.content_classifier = nn.Sequential(nn.Linear(128*2,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,2))

    def forward(self,kwargs):
        all_phrase_semantic_fea=kwargs['all_phrase_semantic_fea']
        all_phrase_emo_fea=kwargs['all_phrase_emo_fea']
        raw_visual_frames=kwargs['raw_visual_frames']
        raw_audio_emo=kwargs['raw_audio_emo']
        
        raw_t_fea_emo=self.mlp_text_emo(all_phrase_emo_fea).unsqueeze(1)
        raw_a_fea_emo=self.mlp_audio(raw_audio_emo).unsqueeze(1) 
        fusion_emo_fea=self.trm_emo(torch.cat((raw_t_fea_emo,raw_a_fea_emo),1))
        fusion_emo_fea=torch.mean(fusion_emo_fea,1)

        raw_t_fea_semantic=self.mlp_text_semantic(all_phrase_semantic_fea)
        raw_v_fea=self.mlp_img(raw_visual_frames)
        content_v, content_t = self.co_attention_tv(v=raw_v_fea, s=raw_t_fea_semantic, v_len=raw_v_fea.shape[1],s_len=raw_t_fea_semantic.shape[1])
        content_v=torch.mean(content_v,-2) 
        content_t=torch.mean(content_t,-2)
        fusion_semantic_fea=self.trm_semantic(torch.cat((content_t.unsqueeze(1),content_v.unsqueeze(1)),1))
        fusion_semantic_fea=torch.mean(fusion_semantic_fea,1) 

        msam_fea=torch.cat((fusion_emo_fea,fusion_semantic_fea),1)
        output_msam=self.content_classifier(msam_fea)
        return output_msam


class MEAM(torch.nn.Module):
    def __init__(self,dataset):
        super(MEAM,self).__init__()
        self.input_visual_frames=83
        self.pad_seg_count=83
        self.pad_ocr_phrase_count=80

        self.ocr_pattern_fea_downscaling = nn.Sequential(
            nn.Conv2d(256, 256 //4, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(256 // 4),
            nn.GELU(),
            nn.Conv2d(256 // 4, 256 // 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.mlp_ocr_pattern=nn.Sequential(nn.Linear(4096,2048),nn.ReLU(),nn.Dropout(0.1),torch.nn.Linear(2048, 512),torch.nn.ReLU(),nn.Dropout(0.1),torch.nn.Linear(512, 128),torch.nn.ReLU())
        self.proj_t=nn.Linear(512,128) 
        self.t_interseg_attention = Attention(128,heads=4,dim_head=64)

        self.proj_v=nn.Linear(512,128)
        self.intraseg_att_v=Attention(dim=128,heads=4)
        self.v_interseg_attention = Attention(dim=128,heads=4)


        self.position_encoder=PosEncoding_fix(128)
        self.dura_encoder=DurationEncoding(64,dataset)

        self.narative_interact_trm=nn.TransformerEncoderLayer(d_model = 128, nhead = 2, batch_first = True)

        self.narative_classifier = nn.Sequential(nn.Linear(128*2,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,2))

    def segment_feature_aggregation_att(self,frame_fea, frames_seg_indicator):
        seg_counts = torch.bincount(frames_seg_indicator[frames_seg_indicator != -1])
        max_frames = torch.max(seg_counts).item()
        unique_segments = torch.unique(frames_seg_indicator[frames_seg_indicator != -1])

        padded_seg_frames=[]
        for idx, seg_id in enumerate(unique_segments):
            frames = frame_fea[frames_seg_indicator == seg_id]
            pad_amount = max_frames - frames.size(0)
            if pad_amount > 0:
                frames = F.pad(frames, (0, 0, 0, pad_amount), "constant", 0)
            padded_seg_frames.append(frames)

        padded_seg_frames = torch.stack(padded_seg_frames).cuda()
        aggregated_seg_fea=self.intraseg_att_v(padded_seg_frames)
        return  torch.mean(aggregated_seg_fea,1)

    def forward(self,kwargs):
        ocr_phrases_fea=kwargs['ocr_phrase_fea']
        ocr_time_region=kwargs['ocr_time_region']
        visual_frames_fea=kwargs['visual_frames_fea']
        visual_frames_seg_indicator=kwargs['visual_frames_seg_indicator']
        visual_seg_paded=kwargs['visual_seg_paded']
        fps=kwargs['fps']
        total_frames=kwargs['total_frame']
        ocr_pattern_fea=kwargs['ocr_pattern_fea']

        down_scaling_ocr_pattern_fea=self.ocr_pattern_fea_downscaling(ocr_pattern_fea)
        flatten_ocr_pattern_fea=down_scaling_ocr_pattern_fea.view(down_scaling_ocr_pattern_fea.size(0), -1)
        ocr_layout_pattern=self.mlp_ocr_pattern(flatten_ocr_pattern_fea) 
        
        v_temporal=[]
        narrative_v_fea=self.proj_v(visual_frames_fea)
        for v_idx in range(len(narrative_v_fea)):
            v_seg_fea=self.segment_feature_aggregation_att(narrative_v_fea[v_idx],visual_frames_seg_indicator[v_idx]) 
            v_ab_value,v_re_value=get_dura_info_visual(visual_seg_paded[v_idx],fps[v_idx],total_frames[v_idx])
            v_ab_emb=self.dura_encoder(v_ab_value,'natural_ab') 
            v_re_emb=self.dura_encoder(v_re_value,'natural_re') 
            dura_emd=torch.cat([v_ab_emb,v_re_emb],dim=1) 
            seg_general_fea=v_seg_fea+dura_emd 
            
            #add position embedding
            seg_index=torch.tensor([i for i in range(v_seg_fea.shape[0])]).cuda()
            seg_position_embedding=self.position_encoder(seg_index) 
            seg_general_fea=v_seg_fea+seg_position_embedding
            if seg_general_fea.shape[0]<self.pad_seg_count:
                pad_seg=torch.zeros((self.pad_seg_count-seg_general_fea.shape[0],128)).cuda()
                seg_general_fea=torch.cat([seg_general_fea,pad_seg],dim=0)
            v_temporal.append(seg_general_fea)
        v_temporal=torch.stack(v_temporal,dim=0) 

        t_temporal=[]
        for v_idx in range(len(ocr_phrases_fea)):
            ocr_phrase_fea=self.proj_t(ocr_phrases_fea[v_idx]) 
            ocr_ab_value,ocr_re_value=get_dura_info_visual(ocr_time_region[v_idx],fps[v_idx],total_frames[v_idx])
            ocr_ab_emb=self.dura_encoder(ocr_ab_value,'ocr_ab')
            ocr_re_emb=self.dura_encoder(ocr_re_value,'ocr_re') 
            ocr_dura_emb=torch.cat([ocr_ab_emb,ocr_re_emb],dim=1) 
            ocr_phrase_fea=ocr_phrase_fea[:ocr_dura_emb.shape[0]]
            ocr_word_fea=ocr_phrase_fea+ocr_dura_emb
            #add position embedding
            phrase_index=torch.tensor([i for i in range(ocr_re_emb.shape[0])]).cuda()
            phrase_position_embedding=self.position_encoder(phrase_index)
            ocr_word_fea=ocr_word_fea+phrase_position_embedding
        
            if ocr_word_fea.shape[0]<self.pad_ocr_phrase_count:
                pad_phrase=torch.zeros((self.pad_ocr_phrase_count-ocr_word_fea.shape[0],128)).cuda()
                ocr_word_fea=torch.cat((ocr_word_fea,pad_phrase),dim=0)
            t_temporal.append(ocr_word_fea)
        t_temporal=torch.stack(t_temporal,dim=0) 
        

        narative_t=self.t_interseg_attention(t_temporal)
        ocr_seg_count=torch.tensor([len(ocr_time_region[i]) for i in range(len(ocr_time_region))]).cuda()
        narative_t=torch.sum(narative_t,dim=1)/ocr_seg_count.unsqueeze(1)

        narrative_v=self.v_interseg_attention(v_temporal)
        v_seg_count=torch.tensor([len(visual_seg_paded[i]) for i in range(len(visual_seg_paded))]).cuda()
        narrative_v=torch.sum(narrative_v,dim=1)/v_seg_count.unsqueeze(1) 

        narrative_multimodal_segs_fea=torch.cat((narative_t.unsqueeze(1),narrative_v.unsqueeze(1)),1)
        narrative_multimodal_segs_fea=self.narative_interact_trm(narrative_multimodal_segs_fea)
        narrative_temporal_fea=torch.mean(narrative_multimodal_segs_fea,dim=1) 

        meam_fea=torch.cat((ocr_layout_pattern,narrative_temporal_fea),1) 
        output_meam=self.narative_classifier(meam_fea)
        return output_meam


class FakingRecipe(torch.nn.Module):
    def __init__(self,dataset):
        super(FakingRecipe, self).__init__()
        self.content_branch=MSAM(dataset=dataset)
        self.editing_branch=MEAM(dataset=dataset)
        self.tanh = nn.Tanh()

    def forward(self, kwargs):
        output_msam=self.content_branch(kwargs)
        output_meam=self.editing_branch(kwargs)
        output=output_msam*self.tanh(output_meam)
        return output  #,output_msam,output_meam