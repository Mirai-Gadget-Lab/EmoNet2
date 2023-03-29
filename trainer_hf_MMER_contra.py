import os
from models.pl_model_hf_contra import PL_model
import pytorch_lightning as pl
from dataset_hf import *
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2Processor
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import OmegaConf as OC

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("-t", '--train_config', default='./configs/train.yaml', type=str)
    p.add_argument("-p", '--preprocess_config', default='./configs/preprocess.yaml', type=str)
    config = p.parse_args()

    return config


def main(args):
    pl.seed_everything(42)
    num_gpu = torch.cuda.device_count()
    train_config = OC.load(args.train_config)
    preprocess_config = OC.load(args.preprocess_config)


    # Load train and validation data
    csv = pd.read_csv(preprocess_config['path']['csv_path'])
    csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)
        
    csv['wav_length'] = csv['wav_end'] - csv['wav_start']
    csv = csv.query("wav_length <= %d"%25)
    dev, _ = train_test_split(csv, test_size=0.2, random_state=1004)
    train, val = train_test_split(dev, test_size=0.1, random_state=1004)
    
    text_tokenizer = AutoTokenizer.from_pretrained(train_config['model']['text_encoder'])
    audio_processor = Wav2Vec2Processor.from_pretrained(train_config['model']['audio_processor'])
    
    train_dataset = multimodal_dataset(train, preprocess_config)
    val_dataset = multimodal_dataset(val, preprocess_config)
    
    train_loader = DataLoader(train_dataset, train_config['optimizer']['batch_size'], num_workers=8,
                              collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
                              shuffle=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, train_config['optimizer']['batch_size'], num_workers=8,
                              collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True, 
                              drop_last=True, shuffle=False)
        
    # Load model and configuration.
    model = PL_model(train_config)
    
    setattr(model, 'train_dataloader', lambda: train_loader)
    setattr(model, 'val_dataloader', lambda: val_loader)
    

    checkpoint_callback = plc.ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(train_config['path']['ckpt_path'], train_config['path']['exp_name']),
        filename="{step:06d}-{val_loss:.5f}",
        save_top_k=10,
        mode="min",
        every_n_train_steps=train_config['step']['save_step']
    )

    logger = TensorBoardLogger(
        train_config['path']['log_path'], name=train_config['path']['exp_name'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpu,
        strategy = DDPStrategy(find_unused_parameters=True),
        max_steps=train_config['step']['total_step'],
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, lr_monitor],
        profiler="simple",
        accumulate_grad_batches=train_config['trainer']['grad_acc'],
        logger=logger,
        gradient_clip_val=train_config['trainer']['grad_clip_thresh'],
    )
    trainer.fit(model)
if __name__ == '__main__':
    args = define_argparser()
    main(args)