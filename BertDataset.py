import codecs
from torch.utils.data import Dataset
from Constants import *
from data.Utils import *
import json

class MALDataset(Dataset):
    def __init__(self, files, src_vocab2id, tgt_vocab2id,n=1E10):
        super(MALDataset, self).__init__()
        self.ids = list()
        self.contexts = list()
        self.queries = list()
        self.outputs = list()
        self.backgrounds = list()

        self.id_arrays = list()
        self.context_arrays = list()
        self.query_arrays = list()
        self.output_arrays = list()
        self.background_arrays = list()
        self.background_selection_arrays = list()
        self.background_ref_start_arrays = list()
        self.background_ref_end_arrays = list()

        self.bg_dyn_vocab2ids=list()
        self.bg_dyn_id2vocabs=list()
        self.background_copy_arrays= list()

        self.src_vocab2id=src_vocab2id
        self.tgt_vocab2id=tgt_vocab2id
        self.files=files
        self.n=n

        self.load()

    def load(self):
        with codecs.open(self.files[0], encoding='utf-8') as f:
            data = json.load(f)
            for id in range(len(data)):
                sample=data[id]
                r""" 
                1. 取出原始文本
                2. 放进contexts
                3. 把原始文本变成id_tensor并保存
                """
                context = sample['context'].split(' ')[-120:]
                self.contexts.append(context)
                self.context_arrays.append(torch.tensor([self.src_vocab2id.get(w.lower(), UNK) for w in context], requires_grad=False).long())

                query = sample['query'].split(' ')
                self.queries.append(query)
                self.query_arrays.append(torch.tensor([self.src_vocab2id.get(w.lower(), UNK) for w in query], requires_grad=False).long())

                background = sample['background'].split(' ')[-256:]
                self.backgrounds.append(background)
                self.background_arrays.append(torch.tensor([self.src_vocab2id.get(w.lower(), UNK) for w in background], requires_grad=False).long())

                bg_dyn_vocab2id, bg_dyn_id2vocab = build_vocab(sample['background'].lower().split(' '))
                self.bg_dyn_vocab2ids.append((id, bg_dyn_vocab2id))
                self.bg_dyn_id2vocabs.append((id, bg_dyn_id2vocab))

                output = sample['response'].lower().split(' ')
                self.outputs.append(output)
                self.output_arrays.append(torch.tensor([self.tgt_vocab2id.get(w, UNK) for w in output] + [EOS], requires_grad=False).long())
                self.background_copy_arrays.append(torch.tensor([bg_dyn_vocab2id.get(w, UNK) for w in output] + [EOS], requires_grad=False).long())

                output = set(output)
                self.background_selection_arrays.append(torch.tensor([1 if w.lower() in output else 0 for w in background], requires_grad=False).long())

                if 'bg_ref_start' in sample:
                    self.background_ref_start_arrays.append(torch.tensor([sample['bg_ref_start']], requires_grad=False))
                    self.background_ref_end_arrays.append(torch.tensor([sample['bg_ref_end'] - 1], requires_grad=False))
                else:
                    self.background_ref_start_arrays.append(torch.tensor([-1], requires_grad=False))
                    self.background_ref_end_arrays.append(torch.tensor([-1], requires_grad=False))

                self.ids.append(id)
                self.id_arrays.append(torch.tensor([id]).long())

                if len(self.contexts)>=self.n:
                    break
        self.len = len(self.contexts)
        print('data size: ', self.len)