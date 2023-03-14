# %%
from torch import nn
import torch
from config import HF_DataConfig, HF_TrainConfig
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from models.pl_model_hf import PL_model_MMER_multiloss
from dataset_hf import multimodal_dataset, multimodal_collator
from transformers import AutoTokenizer, Wav2Vec2Processor
from torch.utils.data import DataLoader
from models.model_hf import Emotion_MMER
# %%
data_config = HF_DataConfig()
train_config = HF_TrainConfig(
    batch_size=8
)

# Load train and validation data
csv = pd.read_csv(data_config.csv_path)
csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)

csv['wav_length'] = csv['wav_end'] - csv['wav_start']
csv = csv.query("wav_length <= %d"%25)
dev, _ = train_test_split(csv, test_size=0.2, random_state=1004)
train, val = train_test_split(dev, test_size=0.1, random_state=1004)
# %%
text_tokenizer = AutoTokenizer.from_pretrained(train_config.text_encoder)
audio_processor = Wav2Vec2Processor.from_pretrained("w11wo/wav2vec2-xls-r-300m-korean")
# %%

val_dataset = multimodal_dataset(val, data_config)

val_loader = DataLoader(val_dataset, train_config.batch_size, num_workers=8,
                            collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True, 
                            drop_last=True, shuffle=False)
# %%
model = Emotion_MMER(train_config)
# %%
for data in val_loader:
    break
# %%
text_inputs, audio_inputs, labels = data

txt_out = model.text_encoder(**text_inputs)
txt_feat_pooled = txt_out['pooler_output']
txt_feat = txt_out['last_hidden_state']
# %%
aud_feat = model.audio_encoder(**audio_inputs)[0]
# %%
def create_negative_samples(pooled_txt_feat: torch.Tensor, pos_idx: int = 0):
    
    x = pooled_txt_feat.unsqueeze(1) # (bs, 1, hideen_size)
    
    # Create negative sample using shift
    # Number of negative sample = 4
    negative_samples = torch.cat([
        x.roll(shifts=1, dims=0)[:,:2,:],
        x.roll(shifts=2, dims=0)[:,:2,:],
        x.roll(shifts=-1, dims=0)[:,:2,:],
        x.roll(shifts=-2, dims=0)[:,:2,:]
    ], dim=1)
    output = torch.cat([x, negative_samples], dim=1) # (bs, 5, hideen_size)
    
    # Shuffle negative samples and positive samples
    idx = torch.randperm(5)
    output = output[:, idx].view(output.size())
    pos_idx = (idx == pos_idx).nonzero(as_tuple=True)[0].item()
    output[:, pos_idx] = pooled_txt_feat
    
    labels = torch.zeros(output.shape[:2]).type_as(output)
    labels[:, pos_idx] = 1
    assert torch.all(output[:, pos_idx] == pooled_txt_feat)
    
    return output, labels
# %%
output, labels = create_negative_samples(txt_feat_pooled)
# %%
pooled_aud = model.pool_layer(aud_feat)
# %%
l2_dist = 1 / torch.cdist(pooled_aud, output)
# %%
cross_entropy = nn.CrossEntropyLoss()
# %%
cross_entropy(l2_dist.squeeze(), labels)
