from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from glob import glob
import os, argparse

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_save_path', default='./output/ckpt/', type=str)
    config = p.parse_args()
    return config

def main(args):
    pathes = glob(os.path.join(args.model_save_path, '*/*.ckpt'))
    for ckpt_path in pathes:
        save_path = os.path.join(os.path.dirname(ckpt_path), 'model_weights')
        os.makedirs(save_path, exist_ok=True)
        convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, os.path.join(save_path, "lightning_model.pt")) 
        
if __name__ == '__main__':
    args = define_argparser()
    main(args)