CUDA_VISIBLE_DEVICES=0,2 python trainer_hf.py --exp_name audio_ce --using_model audio --batch_size 12 --accumulate_grad 1
CUDA_VISIBLE_DEVICES=0,2 python trainer_hf.py --exp_name both_ce  --using_model both --batch_size 12 --accumulate_grad 1
CUDA_VISIBLE_DEVICES=0,2 python trainer_hf.py --exp_name text_ce  --using_model text --batch_size 128 --accumulate_grad 1

CUDA_VISIBLE_DEVICES=0,2 python trainer_hf_MMER_contra.py --exp_name multimodal_contra --batch_size 12 --accumulate_grad 1