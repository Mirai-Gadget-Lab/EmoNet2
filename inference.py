from omegaconf import OmegaConf as OC
import pandas as pd
import os, argparse
from dataset_hf import multimodal_dataset, multimodal_collator
from torch.utils.data import DataLoader
from models.pl_model_hf import *
from transformers import AutoTokenizer, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch import nn
from glob import glob

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_save_path', default='./output/ckpt/', type=str)
    p.add_argument('--config_path', default='./output/log/', type=str)
    p.add_argument('--train_config', default='./configs/train.yaml', type=str)
    p.add_argument('--preprocess_config', default='./configs/preprocess.yaml', type=str)
    p.add_argument('--out_path', default='./result/metrics.csv', type=str)
    config = p.parse_args()

    return config

def predict(trainer, loader, train_config, ckp_path):
    print("inference_start from model:", ckp_path) 
    model = PL_model(train_config).load_from_checkpoint(ckp_path)
    predictions = trainer.predict(model, loader)
    preds = [i[0] for i in predictions]
    labels = [i[1] for i in predictions]
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    f1 = f1_score(labels, np.argmax(preds, axis=1), average='weighted')
    acc = accuracy_score(labels, np.argmax(preds, axis=1))
    return f1, acc

# %%
def main(args):

    train_config = OC.load(args.train_config)
    preprocess_config = OC.load(args.preprocess_config)

    csv = pd.read_csv(preprocess_config['path']['csv_path'])
    csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)
            
    csv['wav_length'] = csv['wav_end'] - csv['wav_start']
    csv = csv.query("wav_length <= %d"%25)
    _, test = train_test_split(csv, test_size=0.2, random_state=1004, stratify=csv['emotion'])
    text_tokenizer = AutoTokenizer.from_pretrained(train_config['model']['text_encoder'])
    audio_processor = Wav2Vec2Processor.from_pretrained(train_config['model']['audio_processor'])

    test_dataset = multimodal_dataset(test, preprocess_config)
    test_loader = DataLoader(test_dataset, 16, num_workers=8,
                                collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
                                shuffle=False, drop_last=False)
    trainer = Trainer(gpus=1,
                    logger=False)
    
    ckpt_path = sorted(glob(os.path.join(args.model_save_path, '*')))
    dict_ls = []
    for path in ckpt_path:
        model_name = os.path.basename(path)
        config_path = sorted(glob(os.path.join(args.config_path, model_name+'/*/hparams.yaml')))[0]
        weight_path = os.path.join(path, "model_weights/lightning_model.pt")
        train_config = OC.load(config_path)['train_config']
        f1, acc = predict(trainer, test_loader, train_config, weight_path)
        
        dict_ls.append({
            "model" : model_name,
            "accuracy": acc,
            "f1_score": f1,
        })
    pd.DataFrame(dict_ls).to_csv(args.out_path, index=False)
# %%
if __name__ == '__main__':
    args = define_argparser()
    main(args)