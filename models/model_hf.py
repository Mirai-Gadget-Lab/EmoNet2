from transformers import AutoModel, Wav2Vec2ForCTC
from torch import nn
import torch 
from models.module import *
from models.tools import create_negative_samples

class Emotion_MultinomialModel(nn.Module):
    def __init__(self, config):
        super(Emotion_MultinomialModel, self).__init__()
        
        self.config = config
        
        if config['model']['using_model'] == 'both': 
            self.text_encoder = AutoModel.from_pretrained(config['model']['text_encoder'])
            self.audio_encoder = Wav2Vec2ForCTC.from_pretrained(config['model']['audio_processor'])
            self.audio_encoder.lm_head = nn.Linear(1024, 768)
            self.audio_pool = nn.AdaptiveAvgPool2d((1, 768))
        
            self.emotion_out = nn.Linear(1536, 7)

        elif config['model']['using_model'] == 'audio':
            self.audio_encoder =Wav2Vec2ForCTC.from_pretrained(config['model']['audio_processor'])
            self.audio_encoder.lm_head = nn.Linear(1024, 768)
            self.audio_pool = nn.AdaptiveAvgPool2d((1, 768))
            
            self.emotion_out = nn.Linear(768, 7)
            
        elif config['model']['using_model'] =='text':
            self.text_encoder = AutoModel.from_pretrained(config['model']['text_encoder'])
            self.emotion_out = nn.Linear(768, 7)
        else:
            raise "WrongModelName"

        if self.config['model']['using_contra']:
            self.cos_sim = nn.CosineSimilarity(dim=2)
    def forward(self, text_inputs, audio_inputs):
        
        sim = None
        contrastive_label = None
        if self.config['model']['using_model'] == 'both': 
            text_feat = self.text_encoder(**text_inputs)['pooler_output']
            audio_feat = self.audio_encoder(**audio_inputs)[0]
            audio_feat = self.audio_pool(audio_feat).squeeze()
            
            feat = torch.cat([text_feat, audio_feat], dim=1)

            if self.config['model']['using_contra']:
                text_posneg, contrastive_label = create_negative_samples(text_feat.squeeze())
                with torch.cuda.amp.autocast():
                    sim = self.cos_sim(audio_feat.unsqueeze(1), text_posneg)
                
        elif self.config['model']['using_model'] == 'text':
            feat = self.text_encoder(**text_inputs)['pooler_output']
        elif self.config['model']['using_model'] == 'audio':
            feat = self.audio_encoder(**audio_inputs)[0]
            feat = self.audio_pool(feat).squeeze()
        pred_emotion = self.emotion_out(feat)
        
        return pred_emotion, sim, contrastive_label
    
class Emotion_MMER(nn.Module):
    def __init__(self, config):
        super(Emotion_MMER, self).__init__()
        
        self.config = config
        
        # Encoder
        self.text_encoder = AutoModel.from_pretrained(config['model']['text_encoder'])
        self.audio_encoder = Wav2Vec2ForCTC.from_pretrained(config['model']['audio_processor'])
        self.audio_encoder.lm_head = nn.Linear(1024, 768)
        
        # MMER
        self.CMA_1 = CrossModalEncoder(768, config['model']['n_head'], config['model']['dropout_p'])
        self.CMA_2 = CrossModalEncoder(768, config['model']['n_head'], config['model']['dropout_p'])
        
        # pooling
        self.pool_layer = nn.AdaptiveAvgPool2d((1, 768))
        self.emotion_out = nn.Linear(1536, 7)
        self.emotion_out_act = nn.Tanh()
        self.emotion_out_dropout = nn.Dropout(config['model']['dropout_p'])
        
        if self.config['model']['using_contra']:
            self.cos_sim = nn.CosineSimilarity(dim=2)
    def forward(self, text_inputs, audio_inputs):
        
        sim = None
        contrastive_label = None
        # Get features from each encoders
        text_feat = self.text_encoder(**text_inputs)['last_hidden_state']
        audio_feat = self.audio_encoder(**audio_inputs)[0]
        
        # Cross attention each modality
        pooled_text = self.pool_layer(self.CMA_1(text_feat, audio_feat)).squeeze()
        pooled_audio = self.pool_layer(self.CMA_2(audio_feat, text_feat)).squeeze()
        
        # Get emotion output
        concated_feat = torch.cat([pooled_audio, pooled_text], dim=1)
        emotion_out = self.emotion_out_dropout(self.emotion_out_act(self.emotion_out(concated_feat)))

        # Get constrastive out
        if self.config['model']['using_contra']:
            text_posneg, contrastive_label = create_negative_samples(pooled_text.squeeze())
            with torch.cuda.amp.autocast():
                sim = self.cos_sim(pooled_audio.unsqueeze(1), text_posneg)
        return emotion_out, sim, contrastive_label