from transformers import BertTokenizer, BertModel, AutoTokenizer
import torch

t = BertTokenizer.from_pretrained('bert-base-uncased')
print(t.vocab.get("[SEP]"))