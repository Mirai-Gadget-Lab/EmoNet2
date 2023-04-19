import torch
from torch import nn
import pytorch_lightning as pl
from models.model_hf import Emotion_MultinomialModel, Emotion_MMER
import torchmetrics
import torch.nn.functional as F
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

class PL_model(pl.LightningModule):
    def __init__(self, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = Emotion_MMER(train_config) if train_config['model']['using_cma'] else Emotion_MultinomialModel(train_config)
        self.train_config = train_config
        self.loss_weight = self.train_config['model']['contra_loss_weight'] if train_config['model']['using_contra'] else 0.0
        # Define Accuracy
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=7)
        self.valid_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=7)
        
        # Define Loss
        self.ce = nn.CrossEntropyLoss()

    def forward(self, text_inputs, audio_inputs):
        return self.model(text_inputs, audio_inputs)
    
    def training_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        emo_out, sim, contrastive_label = self.forward(text_inputs, audio_inputs)
        emo_loss, contra_loss= self.cal_loss(emo_out, labels['emotion'], sim, contrastive_label)
        
        self.train_accuracy(emo_out, labels['emotion'])
        loss = emo_loss + contra_loss
        self.log("train_full_loss",  loss, on_epoch=False, on_step=True)
        self.log("train_emotion_loss",  emo_loss, on_epoch=False, on_step=True)
        self.log("train_contrastive_loss",  contra_loss, on_epoch=False, on_step=True)
        self.log("train_emotion_accuracy",  self.train_accuracy, on_epoch=False, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        text_inputs, audio_inputs, labels = batch
        emo_out, sim, contrastive_label = self.forward(text_inputs, audio_inputs)
        emo_loss, contra_loss= self.cal_loss(emo_out, labels['emotion'], sim, contrastive_label)
        
        self.valid_accuracy(emo_out, labels['emotion'])
        
        loss = emo_loss + contra_loss
        self.log("val_full_loss",  loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_emotion_loss",  emo_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_contrastive_loss",  contra_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_emotion_accuracy",  self.valid_accuracy, on_epoch=True, on_step=False, sync_dist=True)
        
    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        text_inputs, audio_inputs, labels = batch
        pred, _, _ = self.forward(text_inputs, audio_inputs)
        pred = nn.functional.softmax(pred, dim=1)
        return pred, labels['emotion']
    
    def configure_optimizers(self):
        # optimizer = DeepSpeedCPUAdam(
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            betas=self.train_config["optimizer"]["betas"],
            eps=self.train_config["optimizer"]["eps"],
            weight_decay=self.train_config["optimizer"]["weight_decay"],
            lr=self.train_config["optimizer"]['lr']
        )
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=self.train_config['step']['total_step'],
            num_warmup_steps=self.train_config['step']['warm_up_step'],
            num_cycles=self.train_config['step']['num_cycle']
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def cal_loss(self, emo_out, emo_label, sim, contrastive_label):
        emo_loss = (1 - self.loss_weight) * self.ce(emo_out, emo_label)
        if self.train_config['model']['using_contra']:
            contra_loss = self.loss_weight * self.ce(sim, contrastive_label)
        else:
            contra_loss = 0
        return emo_loss, contra_loss