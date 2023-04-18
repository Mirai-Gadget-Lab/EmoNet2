import os
from models.pl_model_hf import *

import os
from models.pl_model_hf import PL_model
import pytorch_lightning as pl
from dataset_hf import *
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2Processor
from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import OmegaConf as OC
from pytorch_lightning.strategies import DeepSpeedStrategy

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("-t", '--train_config', default='./configs/train.yaml', type=str)
    p.add_argument("-p", '--preprocess_config', default='./configs/preprocess.yaml', type=str)
    p.add_argument('--exp_name', required=True, type=str)
    p.add_argument('--using_model', required=True, type=str)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--accumulate_grad', type=int, default=1)
    p.add_argument('--loss', type=str, default="ce")
    config = p.parse_args()

    return config


def main(args):
    pl.seed_everything(1004)
    num_gpu = torch.cuda.device_count()
    train_config = OC.load(args.train_config)
    preprocess_config = OC.load(args.preprocess_config)

    train_config['path']['exp_name'] = args.exp_name
    train_config['optimizer']['batch_size'] = args.batch_size
    train_config['trainer']['grad_acc'] = args.accumulate_grad
    train_config['model']['using_model'] = args.using_model
    # Load train and validation data
    csv = pd.read_csv(preprocess_config['path']['csv_path'])
    csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)
        
    csv['wav_length'] = csv['wav_end'] - csv['wav_start']
    csv = csv.query("wav_length <= %d"%25)
    dev, _ = train_test_split(csv, test_size=0.2, random_state=1004, stratify=csv['emotion'])
    train, val = train_test_split(dev, test_size=0.1, random_state=1004, stratify=dev['emotion'])
    
    text_tokenizer = AutoTokenizer.from_pretrained(train_config['model']['text_encoder'])
    audio_processor = Wav2Vec2Processor.from_pretrained(train_config['model']['audio_processor'])
    
    train_dataset = multimodal_dataset(train, preprocess_config)
    val_dataset = multimodal_dataset(val, preprocess_config)
    
    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(val_dataset),
    )

    total_batch_size = train_config['optimizer']['batch_size'] * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / (total_batch_size * train_config['trainer']['grad_acc']) * train_config['step']['max_epochs'])
    n_warmup_steps = int(n_total_iterations * train_config['step']['warmup_ratio'])
    
    train_config['step']['total_step'] = n_total_iterations
    train_config['step']['warm_up_step'] = n_warmup_steps
    
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
    
    train_loader = DataLoader(
        train_dataset, train_config['optimizer']['batch_size'], num_workers=4,
        collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
        shuffle=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, train_config['optimizer']['batch_size'], num_workers=4,
        collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True, 
        drop_last=True, shuffle=False
    )
        
    # Load model and configuration.
    
    if args.loss == "ce":
        model = PL_model_ce(train_config)
    elif args.loss == "cs_and_ce":
        model = PL_model(train_config)
    else:
        raise "WrongLossName"
    setattr(model, 'train_dataloader', lambda: train_loader)
    setattr(model, 'val_dataloader', lambda: val_loader)
        
    checkpoint_callback = plc.ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(train_config['path']['ckpt_path'], train_config['path']['exp_name']),
        filename="{step:06d}-{val_loss:.5f}",
        save_top_k=1,
        mode="min",
        every_n_train_steps=train_config['step']['total_step'] // 10 
    )
    logger = TensorBoardLogger(
        train_config['path']['log_path'], name=train_config['path']['exp_name'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpu,
        strategy="deepspeed_stage_2",
        max_steps=train_config['step']['total_step'],
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, lr_monitor],
        profiler="simple",
        accumulate_grad_batches=train_config['trainer']['grad_acc'],
        logger=logger,
        gradient_clip_val=train_config['trainer']['grad_clip_thresh'],
        precision=16,
    )
    
    trainer.fit(model)
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)
