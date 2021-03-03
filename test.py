from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

from modules.Attentions import *
for i in range(10):
    if i == 0:
        continue
    print(i)
