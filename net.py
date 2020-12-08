#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:20:35 2020

@author: luokai
"""

import torch
from torch import nn
from transformers import BertModel


class Net(nn.Module):
    def __init__(self, bert_model):
        super(Net, self).__init__()
        self.bert = bert_model
        self.atten_layer = nn.Linear(768, 16)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.OCNLI_layer = nn.Linear(768, 16 * 3)
        self.OCEMOTION_layer = nn.Linear(768, 16 * 7)
        self.TNEWS_layer = nn.Linear(768, 16 * 15)

    def forward(self, input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :].squeeze(1)
        if ocnli_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[ocnli_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            ocnli_value = self.OCNLI_layer(cls_emb[ocnli_ids, :]).contiguous().view(-1, 16, 3)
            ocnli_out = torch.matmul(attention_score, ocnli_value).squeeze(1)
        else:
            ocnli_out = None
        if ocemotion_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[ocemotion_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            ocemotion_value = self.OCEMOTION_layer(cls_emb[ocemotion_ids, :]).contiguous().view(-1, 16, 7)
            ocemotion_out = torch.matmul(attention_score, ocemotion_value).squeeze(1)
        else:
            ocemotion_out = None
        if tnews_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[tnews_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            tnews_value = self.TNEWS_layer(cls_emb[tnews_ids, :]).contiguous().view(-1, 16, 15)
            tnews_out = torch.matmul(attention_score, tnews_value).squeeze(1)
        else:
            tnews_out = None
        return ocnli_out, ocemotion_out, tnews_out