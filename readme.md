# Repo for Background Based Conversation Task

Useage:  
Leverage `torch.distribution` to use multi-GPU for training.  
``` 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 Run_CaKe.py
```
--- 
This Repo is based on the paper: https://arxiv.org/abs/1906.06685  
