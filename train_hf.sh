CUDA_VISIBLE_DEVICES=0,2 python trainer_hf.py --exp_name audio --using_model audio --using_contra False --using_cma False --batch_size 12 --accumulate_grad 1
CUDA_VISIBLE_DEVICES=0,2 python trainer_hf.py --exp_name both  --using_model both --using_contra False --using_cma False --batch_size 12 --accumulate_grad 1
CUDA_VISIBLE_DEVICES=0,2 python trainer_hf.py --exp_name text  --using_model text --using_contra False --using_cma False --batch_size 128 --accumulate_grad 1

CUDA_VISIBLE_DEVICES=0,2 python trainer_hf.py --exp_name both_cma --using_model both --using_contra False --using_cma True  --batch_size 12 --accumulate_grad 1
CUDA_VISIBLE_DEVICES=0,2 python trainer_hf.py --exp_name both_contra --using_model both --using_contra True --using_cma False  --batch_size 12 --accumulate_grad 1
CUDA_VISIBLE_DEVICES=0,2 python trainer_hf.py --exp_name both_contra_cma --using_model both --using_contra True --using_cma True  --batch_size 12 --accumulate_grad 1