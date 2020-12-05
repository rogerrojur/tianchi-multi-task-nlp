#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:20:35 2020

@author: luokai
"""

from torch import nn
from transformers import BertModel


class Net(nn.Module):
    def __init__(self, bert_model):
        super(Net, self).__init__()
        self.bert = bert_model
        self.OCNLI_out = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm((128,)), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3))
        self.OCEMOTION_out = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm((128,)), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 7))
        self.TNEWS_out = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm((128,)), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 15))

    def forward(self, input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :].squeeze(1)
        if ocnli_ids.size()[0] > 0:
            ocnli_out = self.OCNLI_out(cls_emb[ocnli_ids, :])
        else:
            ocnli_out = None
        if ocemotion_ids.size()[0] > 0:
            ocemotion_out = self.OCEMOTION_out(cls_emb[ocemotion_ids, :])
        else:
            ocemotion_out = None
        if tnews_ids.size()[0] > 0:
            tnews_out = self.TNEWS_out(cls_emb[tnews_ids, :])
        else:
            tnews_out = None
        return ocnli_out, ocemotion_out, tnews_out