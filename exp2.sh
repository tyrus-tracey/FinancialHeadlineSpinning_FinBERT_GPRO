#!/bin/bash

python3 TrainGRPO.py --save_dir rundir --policy_name HuggingFaceTB/SmolLM2-135M-Instruct --autocast 1 --gpus 0 --speedup gpu --lr 1e-4 --wandb online --eval_iter 1 --epochs_warmup 4 --epochs 32 --save_iter 32 --save_iter_t 0.1 --suffix e4_tok36 --tags e4_tok36 --wandb_project 413a3runs --wandb_entity ttracey-simon-fraser-university --grpo_max_new_tokens 36

python3 TrainGRPO.py --save_dir rundir --policy_name HuggingFaceTB/SmolLM2-135M-Instruct --autocast 1 --gpus 0 --speedup gpu --lr 1e-4 --wandb online --eval_iter 1 --epochs_warmup 4 --epochs 32 --save_iter 32 --save_iter_t 0.1 --suffix e4_tok40 --tags e4_tok40 --wandb_project 413a3runs --wandb_entity ttracey-simon-fraser-university --grpo_max_new_tokens 40

python3 TrainGRPO.py --save_dir rundir --policy_name HuggingFaceTB/SmolLM2-135M-Instruct --autocast 1 --gpus 0 --speedup gpu --lr 1e-4 --wandb online --eval_iter 1 --epochs_warmup 4 --epochs 32 --save_iter 32 --save_iter_t 0.1 --suffix e4_tok42 --tags e4_tok42 --wandb_project 413a3runs --wandb_entity ttracey-simon-fraser-university --grpo_max_new_tokens 42