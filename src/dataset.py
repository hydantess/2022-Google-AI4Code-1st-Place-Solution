import pandas as pd
import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import os
# from utils import create_label
import numpy as np


class MarkdownDataset(Dataset):

    def __init__(self, meta_data: pd.DataFrame, tokenizer, fold: int = -1, mode='train', parameter=None, max_seq_length=2048):
        self.meta_data = meta_data.copy()
        self.meta_data.reset_index(drop=True, inplace=True)
        if mode == 'train':
            pass
#             self.meta_data = self.meta_data[self.meta_data['fold_flag'] != fold].copy()
#             self.meta_data = self.meta_data.iloc[:60000]
        elif mode == 'valid':
            self.meta_data = self.meta_data[self.meta_data['fold_flag'] == fold].copy()
            self.meta_data = self.meta_data[self.meta_data['id'].isin(self.meta_data['id'].values[:1000])]
        elif mode == 'test':
            pass
        else:
            raise ValueError(mode)
        self.meta_data.reset_index(drop=True, inplace=True)
        if tokenizer.sep_token != '[SEP]':
            self.meta_data['source'] = self.meta_data['source'].apply(
                lambda x: [
                    y.replace(tokenizer.sep_token, '').replace(tokenizer.cls_token, '').replace(tokenizer.pad_token, '')
                    for y in x])
        self.parameter = parameter
        self.seq_length = max_seq_length
        self.source = self.meta_data['source'].values
        self.cell_type = self.meta_data['cell_type'].values
        # self.cell_id = self.meta_data['cell_id'].values
        self.rank = self.meta_data['rank'].values
        # self.dense_features = self.meta_data[['cell_count','markdown_count', 'code_count']].values
        self.mode = mode
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        source = self.source[index]
        cell_type = self.cell_type[index]
        rank = self.rank[index]
        # dense_features = 1#self.dense_features[index]
#         if self.mode == 'train':
#             range_tmp1 = [ i for i in range(len(cell_type)) if cell_type[i]==0]
#             range_tmp2 = [ i for i in range(len(cell_type)) if cell_type[i]==1]
#             np.random.shuffle(range_tmp2)
#             source = [source[i] for i in range_tmp1 + range_tmp2]
#             rank = [rank[i] for i in range_tmp1 + range_tmp2]
#             rank2 = [rank2[i] for i in range_tmp1 + range_tmp2]

        cell_inputs = self.tokenizer.batch_encode_plus(
            source,
            add_special_tokens=False,
            max_length=self.parameter.cell_max_length,
            # padding="max_length",
            return_attention_mask=False,
            truncation=True,
        )
        seq, seq_mask, target_mask, target = self.max_length_rule_base(cell_inputs['input_ids'],
                                                                                    cell_type, rank)
        # print(seq, seq_mask, dense_features, target_mask, target)
        # if self.mode == 'train':
        #     attention_mask, target = self.random_mask(attention_mask, target)
        # print(encoded)
        # print(target)
        return seq, seq_mask, target_mask, target
        # return encoded['input_ids'][0], encoded['attention_mask'][0], np.array(target, dtype=np.float32)

    def __len__(self):
        return len(self.meta_data)

    def max_length_rule_base(self, cell_inputs, cell_type, rank):
        init_length = [len(x) for x in cell_inputs]
        total_max_length = self.seq_length - len(init_length)
        min_length = total_max_length // len(init_length)
        cell_length = self.search_length(init_length, min_length, total_max_length, len(init_length))
        # print(init_code_length,code_length)

        seq = []
        for i in range(len(cell_length)):
            if cell_type[i] == 0:
                seq.append(self.tokenizer.cls_token_id)
            else:
                seq.append(self.tokenizer.sep_token_id)

            if cell_length[i] > 0:
                seq.extend(cell_inputs[i][:cell_length[i]])

        # print(len(seq),'1111', np.sum(init_length),np.sum(cell_length))
#         if len(seq) < self.seq_length:
#             seq_mask = [1] * len(seq) + [0] * (self.seq_length - len(seq))
#             seq = seq + [self.tokenizer.pad_token_id] * (self.seq_length - len(seq))
#         else:
#             seq_mask = [1] * self.seq_length
#             seq = seq[:self.seq_length]
        if len(seq) > self.seq_length:
            seq_mask = [1] * self.seq_length
            seq = seq[:self.seq_length]
        else:
            seq_mask = [1] * len(seq) 
        seq, seq_mask = np.array(seq, dtype=np.int), np.array(seq_mask, dtype=np.int)
        target_mask = np.where((seq == self.tokenizer.cls_token_id) | (seq == self.tokenizer.sep_token_id), 1, 0)  # todo
        target = np.zeros(len(seq), dtype=np.float32)
        tmp = np.where((seq == self.tokenizer.cls_token_id) | (seq == self.tokenizer.sep_token_id))
        target[tmp] = rank
#         sample_weight = np.zeros(len(seq), dtype=np.float32)
#         sample_weight = np.where(seq == self.tokenizer.cls_token_id, 0.33, sample_weight)
#         sample_weight = np.where(seq == self.tokenizer.sep_token_id, 1.0, sample_weight)
#         dense_features = np.zeros(self.seq_length, dtype=np.float32)
#         dense_features[tmp] = rank2
        return seq, seq_mask, target_mask, target

    @staticmethod
    def search_length(init_length, min_length, total_max_length, cell_count, step=4, max_search_count=50):
        if np.sum(init_length) <= total_max_length:
            return init_length

        res = [min(init_length[i], min_length) for i in range(cell_count)]
        for s_i in range(max_search_count):
            tmp = [min(init_length[i], res[i] + step) for i in range(cell_count)]
            if np.sum(tmp) < total_max_length:
                res = tmp
            else:
                break
        for s_i in range(cell_count):
            tmp = [i for i in res]
            tmp[s_i] = min(init_length[s_i], res[s_i] + step)
            if np.sum(tmp) < total_max_length:
                res = tmp
            else:
                break
        return res

    
class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        # print(len(batch),batch)
        input_ids = [x[0] for x in batch]
        attention_mask = [x[1] for x in batch]
        target_mask = [x[2] for x in batch]
        target = [x[3] for x in batch]
        # sample_weight = [x[4] for x in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in input_ids])

        # add padding
        input_ids = [list(s) + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in input_ids]
        attention_mask = [list(s) + (batch_max - len(s)) * [0] for s in attention_mask]
        target_mask = [list(s) + (batch_max - len(s)) * [0] for s in target_mask]
        target = [list(s) + (batch_max - len(s)) * [0] for s in target]
        # sample_weight = [list(s) + (batch_max - len(s)) * [0] for s in target]
#         # convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        target_mask = torch.tensor(target_mask, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        # sample_weight = torch.tensor(sample_weight, dtype=torch.float32)

        return input_ids, attention_mask, target_mask, target