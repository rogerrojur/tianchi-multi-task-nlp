#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:29:08 2020

@author: luokai
"""
import random
import torch
from transformers import BertTokenizer

class Data_generator():
    def __init__(self, ocnli_dict, ocemotion_dict, tnews_dict, label_dict, device, tokenizer, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.device = device
        self.label2idx = dict()
        self.idx2label = dict()
        for key in ['TNEWS', 'OCNLI', 'OCEMOTION']:
            self.label2idx[key] = dict()
            self.idx2label[key] = dict()
            for i, e in enumerate(label_dict[key]):
                self.label2idx[key][e] = i
                self.idx2label[key][i] = e
        self.ocnli_data = dict()
        self.ocnli_data['s1'] = []
        self.ocnli_data['s2'] = []
        self.ocnli_data['label'] = []
        for k, v in ocnli_dict.items():
            self.ocnli_data['s1'].append(v['s1'])
            self.ocnli_data['s2'].append(v['s2'])
            self.ocnli_data['label'].append(self.label2idx['OCNLI'][v['label']])
        self.ocemotion_data = dict()
        self.ocemotion_data['s1'] = []
        self.ocemotion_data['label'] = []
        for k, v in ocemotion_dict.items():
            self.ocemotion_data['s1'].append(v['s1'])
            self.ocemotion_data['label'].append(self.label2idx['OCEMOTION'][v['label']])
        self.tnews_data = dict()
        self.tnews_data['s1'] = []
        self.tnews_data['label'] = []
        for k, v in tnews_dict.items():
            self.tnews_data['s1'].append(v['s1'])
            self.tnews_data['label'].append(self.label2idx['TNEWS'][v['label']])
        self.reset()
    def reset(self):
        self.ocnli_ids = list(range(len(self.ocnli_data['s1'])))
        self.ocemotion_ids = list(range(len(self.ocemotion_data['s1'])))
        self.tnews_ids = list(range(len(self.tnews_data['s1'])))
        random.shuffle(self.ocnli_ids)
        random.shuffle(self.ocemotion_ids)
        random.shuffle(self.tnews_ids)
    def get_next_batch(self, batchSize=64):
        ocnli_len = len(self.ocnli_ids)
        ocemotion_len = len(self.ocemotion_ids)
        tnews_len = len(self.tnews_ids)
        total_len = ocnli_len + ocemotion_len + tnews_len
        if total_len == 0:
            return None
        elif total_len > batchSize:
            if ocnli_len > 0:
                ocnli_tmp_len = int((ocnli_len / total_len) * batchSize)
                ocnli_cur = self.ocnli_ids[:ocnli_tmp_len]
                self.ocnli_ids = self.ocnli_ids[ocnli_tmp_len:]
            if ocemotion_len > 0:
                ocemotion_tmp_len = int((ocemotion_len / total_len) * batchSize)
                ocemotion_cur = self.ocemotion_ids[:ocemotion_tmp_len]
                self.ocemotion_ids = self.ocemotion_ids[ocemotion_tmp_len:]
            if tnews_len > 0:
                tnews_tmp_len = batchSize - len(ocnli_cur) - len(ocemotion_cur)
                tnews_cur = self.tnews_ids[:tnews_tmp_len]
                self.tnews_ids = self.tnews_ids[tnews_tmp_len:]
        else:
            ocnli_cur = self.ocnli_ids
            self.ocnli_ids = []
            ocemotion_cur = self.ocemotion_ids
            self.ocemotion_ids = []
            tnews_cur = self.tnews_ids
            self.tnews_ids = []
        max_len = self._get_max_total_len(ocnli_cur, ocemotion_cur, tnews_cur)
        input_ids = []
        token_type_ids = []
        attention_mask = []
        ocnli_gold = None
        ocemotion_gold = None
        tnews_gold = None
        if len(ocnli_cur) > 0:
            flower = self.tokenizer([self.ocnli_data['s1'][idx] for idx in ocnli_cur], [self.ocnli_data['s2'][idx] for idx in ocnli_cur], add_special_tokens=True, max_length=max_len, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            ocnli_gold = torch.tensor([self.ocnli_data['label'][idx] for idx in ocnli_cur]).to(self.device)
        if len(ocemotion_cur) > 0:
            flower = self.tokenizer([self.ocemotion_data['s1'][idx] for idx in ocemotion_cur], add_special_tokens=True, max_length=max_len, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            ocemotion_gold = torch.tensor([self.ocemotion_data['label'][idx] for idx in ocemotion_cur]).to(self.device)
        if len(tnews_cur) > 0:
            flower = self.tokenizer([self.tnews_data['s1'][idx] for idx in tnews_cur], add_special_tokens=True, max_length=max_len, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            tnews_gold = torch.tensor([self.tnews_data['label'][idx] for idx in tnews_cur]).to(self.device)
        st = 0
        ed = len(ocnli_cur)
        ocnli_tensor = torch.tensor([i for i in range(st, ed)]).to(self.device)
        st += len(ocnli_cur)
        ed += len(ocemotion_cur)
        ocemotion_tensor = torch.tensor([i for i in range(st, ed)]).to(self.device)
        st += len(ocemotion_cur)
        ed += len(tnews_cur)
        tnews_tensor = torch.tensor([i for i in range(st, ed)]).to(self.device)
        input_ids = torch.cat(input_ids, axis=0).to(self.device)
        token_type_ids = torch.cat(token_type_ids, axis=0).to(self.device)
        attention_mask = torch.cat(attention_mask, axis=0).to(self.device)
        res = dict()
        res['input_ids'] = input_ids
        res['token_type_ids'] = token_type_ids
        res['attention_mask'] = attention_mask
        res['ocnli_ids'] = ocnli_tensor
        res['ocemotion_ids'] = ocemotion_tensor
        res['tnews_ids'] = tnews_tensor
        res['ocnli_gold'] = ocnli_gold
        res['ocemotion_gold'] = ocemotion_gold
        res['tnews_gold'] = tnews_gold
        return res

    def _get_max_total_len(self, ocnli_cur, ocemotion_cur, tnews_cur):
        res = 1
        for idx in ocnli_cur:
            res = max(res, 3 + len(self.ocnli_data['s1'][idx]) + len(self.ocnli_data['s2'][idx]))
        for idx in ocemotion_cur:
            res = max(res, 2 + len(self.ocemotion_data['s1'][idx]))
        for idx in tnews_cur:
            res = max(res, 2 + len(self.tnews_data['s1'][idx]))
        return min(res, self.max_len)