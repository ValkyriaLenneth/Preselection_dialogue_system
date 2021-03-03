import torch.nn as nn
import torch.nn.functional as F
from data.Utils import *
from EncDecModel import *
from modules.Criterions import *
from modules.Generators import *
import numpy
from torch.nn.init import *
import transformers
from transformers import BertModel
from Constants import *

class BertEncoder(nn.Module):
    """
    Init:
    Input:input: [cls]context[sep]back or last_word, mask, len_c, len_b
    Output: context_enc: B*l_c*H, back_enc: B*l_b*H, last_word_emb: B*1*H
    """
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, data, is_dec=False):
        if is_dec:
            return self.encoder(data)[0]
        else:
            context = data['context'] # B L_c
            back = data['background'] # B L_b
            batch_size, len_c = context.size()
            batch_size, len_b = back.size()

            # CLS + Context + SEP + background
            cat_input = torch.cat([torch.tensor([CLS], dtype=torch.long).cuda().expand(batch_size).unsqueeze(1), \
                        context, \
                        torch.tensor([SEP], dtype=torch.long).cuda().expand(batch_size).unsqueeze(1), \
                        back], dim=1)
            cat_mask = cat_input.ne(0).float()

            c_mask = context.ne(0).float()
            b_mask = context.ne(0).float()
            c_len = c_mask.sum(dim=1).unsqueeze(1)
            # b_len = b_mask.sum(dim=1).unsqueeze(1)

            # B * cat_len * H
            cat_output = self.encoder(cat_input, attention_mask=cat_mask)[0]

            context_enc = cat_output[:, 1: 1+len_c, :]
            back_enc = cat_output[:, -len_b:, :]

            # get avg
            # b_mask = b_mask.unsqueeze(2).expand([batch_size, len_b, 768])
            c_mask = c_mask.unsqueeze(2).expand([batch_size, len_c, 768])
            context_avg = torch.div((context_enc*c_mask).sum(dim=1), c_len)
            return {"context_enc": context_enc, # B, l_c, H
                    "back_enc": back_enc, # B, l_b, H
                    "context_avg": context_avg} # B, 1, H

class Selector(nn.Module):
    """
    Init: Hidden_size: 768, encoder: BertEncoder
    Input: back_enc: B*l_b*H, last_state: B*1*H, last_word: B*1, data
    Output: P1: B*l_b , last_state
    """
    def __init__(self,hidden_size, encoder):
        super(Selector, self).__init__()

        self.encoder = encoder
        self.selector = nn.Linear(hidden_size * 3, 1)
        self.init_weights()

    def init_weights(self):
        self.selector.weight.data.normal_(mean=0.0, std=0.02)
        self.selector.bias.data.zero_()

    def forward(self, back_enc, last_state, last_word, data):
        back_mask = data['background'].ne(PAD) # B * l_b
        last_state = last_state[0] # B * 1 * H

        batch_size, back_size, hidden_size = back_enc.shape

        last_word_embed = self.encoder(last_word, is_dec=True) # last_word: the index of the last word from BERT_PRETRAINED_VOCAB
        # B * l_b * 3H
        input = torch.cat([last_state.expand(batch_size, back_size, hidden_size),
                           last_word_embed.expand(batch_size, back_size, hidden_size),
                           back_enc], dim=-1)
        # Linear => B * l_b :as prob_score
        output = self.selector(input).squeeze(-1)
        # Mask the PAD
        output = output.masked_fill(1-back_mask, -float('inf'))

        return output

class Generator(nn.Module):
    def __init__(self,encoder, hidden_size, tgt_vocab_size):
        super(Generator, self).__init__()

        self.hidden_size=hidden_size
        self.encoder = encoder

        self.attn = BilinearAttention(
            query_size=self.hidden_size, key_size=self.hidden_size, hidden_size=self.hidden_size, dropout=0.5, coverage=False
        )

        self.dec = nn.GRU(2*hidden_size, hidden_size, bidirectional=False, num_layers=1, dropout=0.5, batch_first=True)
        self.readout = nn.Linear(3*hidden_size, hidden_size)
        self.gen=LinearGenerator(feature_size=hidden_size, tgt_vocab_size=tgt_vocab_size)

    def forward(self, data, last_word, state, encode_output):
        self.dec.flatten_parameters()
        # B * l_c * H
        c_enc_output = encode_output['context_enc']
        c_mask=data['context'].ne(PAD)
        # state[0]: B * 1 * H
        state = state[0]

        # last_word: B => unsqueeze(-1) ==> B * 1 * H
        last_word_embed = self.encoder(last_word, is_dec=True)

        # attended vector, attention
        # attn_context_1: B * 1 * H
        # print("state: ", state.shape)
        # print("c_enc_out: ", c_enc_output.shape)
        c_enc_output = c_enc_output.contiguous()
        attn_context_1, attn = self.attn(state, c_enc_output, c_enc_output, query_mask=None, key_mask=c_mask)

        # input:x_{t-1}, attended vector
        # GRU_input: B * 1 * 2H
        gru_input = torch.cat([last_word_embed, attn_context_1], dim=2)

        # update state by attended vector and x_{t-1}
        # gru_output: B * 1 * H
        # state: 1 * B * 256
        gru_output, state = self.dec(gru_input, state.transpose(0,1)) #gru_input: 2hidden+embedding
        # state: B * 1 * 256vim new
        state=state.transpose(0,1)

        # use new state to get new attention vector
        # attn_context: B * 1 * H => B * H
        attn_context, attn = self.attn(state, c_enc_output, c_enc_output, query_mask=None, key_mask=c_mask)
        attn_context = attn_context.squeeze(1)

        # concat_output: B * 3H
        concat_output = torch.cat((last_word_embed.squeeze(1), state.squeeze(1), attn_context), dim=1)

        # feature_output: B * H
        feature_output = self.readout(concat_output)

        # gen_output: B * vocab_size
        return self.gen(feature_output, softmax=False), [state]

class Mixture(nn.Module):
    def __init__(self, state_size):
        super(Mixture, self).__init__()
        self.linear_mixture = nn.Linear(state_size, 1)

    def forward(self, state,  selector_action, generator_action, b_map):
        # p_s_g: B * 1 作为生成门限
        p_s_g = torch.sigmoid(self.linear_mixture(state[0].squeeze(1)))
        # P_background: B * len_b
        selector_action=F.softmax(selector_action, dim=1)  # p_background=softmax(p1)
        # P_vocab: B * vocab_size
        generator_action = F.softmax(generator_action, dim=1) #
        # p * p_vocab: B * vocab_size
        generator_action = torch.mul(generator_action, p_s_g.expand_as(generator_action))  #p*p_vocab

        # b_map: B * max_len_b * max_len_dyn_vocab: 相当于一个mask matrix, 对于每一行来说，只有对应的dyn_vocab位置是1，其他为0
        # b_map * P_background: B * max_len_dyn_vocab => P_background: 相当于映射到dyn_vocab上的概率分布
        selector_action = torch.bmm(selector_action.unsqueeze(1), b_map.float()).squeeze(1)  #selector action is p1
        # (1-p)*P_background: B * max_len_dyn_vocab
        selector_action = torch.mul(selector_action, (1-p_s_g).expand_as(selector_action)) #(1-p)p_background
        # output: B * (vocab_size + max_dyn_vocab_size)
        return torch.cat([generator_action, selector_action], 1)  #return final

class MAL(EncDecModel):
    def __init__(self, encoder, selector, generator, src_id2vocab, src_vocab2id, tgt_id2vocab, tgt_vocab2id, max_dec_len, beam_width, eps=1e-10):
        super(MAL, self).__init__(src_vocab_size=len(src_id2vocab), embedding_size=generator.hidden_size,
                                      hidden_size=generator.hidden_size, tgt_vocab_size=len(tgt_id2vocab), src_id2vocab=src_id2vocab,
                                      src_vocab2id=src_vocab2id, tgt_id2vocab=tgt_id2vocab, tgt_vocab2id=tgt_vocab2id, max_dec_len=max_dec_len, beam_width=beam_width,
                                      eps=eps)
        self.encoder=encoder
        self.selector=selector
        self.generator=generator

        self.state_initializer = nn.Linear(self.hidden_size, self.hidden_size)
        self.mixture=Mixture(self.hidden_size)
        self.criterion = CopyCriterion(len(tgt_id2vocab), force_copy=False, eps=eps)

    def encode(self,data):
        return self.encoder(data, is_dec=False)

    def init_decoder_states(self,data, encode_output):
        context_avg= encode_output['context_avg'] # B, 1, H
        batch_size=context_avg.size(0)
        return [self.state_initializer(context_avg.contiguous().view(batch_size,-1)).view(batch_size, 1, -1)]

    def decode(self, data, tgt, state, encode_output):
        # sel_decode_output=self.selector(data, tgt, state, encode_output)
        tgt = tgt.unsqueeze(1)
        gen_decode_output=self.generator(data, tgt, state, encode_output)
        sel_decode_output=self.selector(encode_output['back_enc'], state, tgt, data)
        return [sel_decode_output,gen_decode_output[0]], gen_decode_output[1]

    def generate(self, data, decode_output, softmax=True):
        actions, state = decode_output
        return self.mixture(state, actions[0], actions[1], data['background_map'])

    def loss(self,data, all_gen_output, all_decode_output, encode_output, reduction='mean'):
        loss=self.criterion(all_gen_output, data['output'], data['background_copy'], reduction=reduction)
        return loss
        # return loss+1e-2*torch.distributions.categorical.Categorical(probs=all_gen_output.view(-1, all_gen_output.size(2))).entropy().mean()

    def generation_to_decoder_input(self, data, indices):
        return indices.masked_fill(indices>=self.tgt_vocab_size, UNK)

    def to_word(self, data, gen_output, k=5, sampling=False):
        if not sampling:
            return copy_topk(gen_output, data['background_vocab_map'], data['background_vocab_overlap'], k=k)
        else:
            return randomk(gen_output, k=k)

    def to_sentence(self, data, batch_indices):
        return to_copy_sentence(data, batch_indices, self.tgt_id2vocab, data['background_dyn_vocab'])

    def forward(self, data, method='mle_train'):
        if method=='mle_train':
            return self.mle_train(data)
        elif method=='mal_train':
            return self.mal_train(data)
        elif method=='env_train':
            return self.env_train(data)
        elif method=='test':
            if self.beam_width==1:
                return self.greedy(data)
            else:
                return self.beam(data)
        elif method=='sample':
            return self.sample(data)

    def env_train(self, data):
        c_mask = data['context'].ne(PAD).detach()
        o_mask = data['output'].ne(PAD).detach()

        with torch.no_grad():
            c = self.encoder.c_embedding(data['context']).detach()
            o = self.generator.o_embedding(data['output']).detach()

            a, encode_outputs, init_decoder_states, all_decode_outputs, all_gen_outputs=sample(self, data, max_len=self.max_dec_len)
            a.masked_fill_(a >= self.tgt_vocab_size, UNK)
            a_mask = a.ne(PAD).detach()
            a = self.generator.o_embedding(a).detach()

        return self.env(c, o, c_mask, o_mask, 1).unsqueeze(0), self.env(c, a, c_mask, a_mask, 0).unsqueeze(0)

    def mle_train(self, data):
        encode_output, init_decoder_state, all_decode_output, all_gen_output=decode_to_end(self,data,schedule_rate=1)

        gen_loss=self.loss(data,all_gen_output,all_decode_output,encode_output).unsqueeze(0)

        return gen_loss

