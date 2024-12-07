import torch

from .utils import  pad_frame_sequence,pad_frame_by_seg,pad_segment,pad_unnatural_phrase

def collate_fn_FakeingRecipe(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_visual_frames=83
    num_segs=83
    num_phrase=80

    vid = [item['vid'] for item in batch]
    label = torch.stack([item['label'] for item in batch])
    all_phrase_semantic_fea = [item['all_phrase_semantic_fea'] for item in batch]
    all_phrase_emo_fea = torch.stack([item['all_phrase_emo_fea'] for item in batch])

    raw_visual_frames = [item['raw_visual_frames'] for item in batch]
    raw_audio_emo = [item['raw_audio_emo'] for item in batch]
    fps = torch.stack([item['fps'] for item in batch])
    total_frame = torch.stack([item['total_frame'] for item in batch])

    content_visual_frames, _ = pad_frame_sequence(num_visual_frames,raw_visual_frames)
    raw_audio_emo = torch.cat(raw_audio_emo,dim=0)

    all_phrase_semantic_fea=[x if x.shape[0]==512 else torch.cat((x,torch.zeros([512-x.shape[0],x.shape[1]],dtype=torch.float)),dim=0) for x in all_phrase_semantic_fea] #batch*512*768
    all_phrase_semantic_fea=torch.stack(all_phrase_semantic_fea)

    ocr_pattern_fea = torch.stack([item['ocr_pattern_fea'] for item in batch])
    ocr_phrase_fea = [item['ocr_phrase_fea'] for item in batch]
    ocr_time_region = [item['ocr_time_region'] for item in batch]

    visual_time_region = [item['visual_time_region'] for item in batch]

    visual_frames_fea,visual_frames_seg_indicator,sampled_seg=pad_frame_by_seg(num_visual_frames,raw_visual_frames,visual_time_region)
    visual_seg_paded=pad_segment(sampled_seg,num_segs)

    ocr_phrase_fea=pad_unnatural_phrase(ocr_phrase_fea,num_phrase)
    ocr_time_region=pad_unnatural_phrase(ocr_time_region,num_phrase)

    return {
        'vid': vid,
        'label': label.to(device),
        'fps': fps.to(device),
        'total_frame': total_frame.to(device),
        'all_phrase_semantic_fea': all_phrase_semantic_fea.to(device),
        'all_phrase_emo_fea': all_phrase_emo_fea.to(device),
        'raw_visual_frames': content_visual_frames.to(device),
        'raw_audio_emo': raw_audio_emo.to(device),
        'ocr_pattern_fea': ocr_pattern_fea.to(device),
        'ocr_phrase_fea': ocr_phrase_fea.to(device),
        'ocr_time_region': ocr_time_region.to(device),
        'visual_frames_fea': visual_frames_fea.to(device),
        'visual_frames_seg_indicator': visual_frames_seg_indicator.to(device),
        'visual_seg_paded': visual_seg_paded.to(device)
    }
