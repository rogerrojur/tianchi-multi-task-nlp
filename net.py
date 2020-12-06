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
        self.share_layer = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm((128,)), nn.ReLU(), nn.Dropout(0.2))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.OCNLI_mid = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm((128,)), nn.ReLU(), nn.Dropout(0.2))
        self.OCNLI_out = nn.Linear(128, 3)
        self.OCEMOTION_mid = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm((128,)), nn.ReLU(), nn.Dropout(0.2))
        self.OCEMOTION_out = nn.Linear(128, 7)
        self.TNEWS_mid = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm((128,)), nn.ReLU(), nn.Dropout(0.2))
        self.TNEWS_out = nn.Linear(128, 15)

    def forward(self, input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids=None, attention_mask=None):
        cls_emb = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :].squeeze(1)
        if ocnli_ids.size()[0] > 0:
            share_part = self.share_layer(cls_emb[ocnli_ids, :]).unsqueeze(2)
            ocnli_part = self.OCNLI_mid(cls_emb[ocnli_ids, :]).unsqueeze(2)
            ocnli_with_share = torch.cat([share_part, ocnli_part], axis=2)
            ocnli_mix = self.avg_pool(ocnli_with_share).squeeze(2)
            ocnli_out = self.OCNLI_out(ocnli_mix)
        else:
            ocnli_out = None
        if ocemotion_ids.size()[0] > 0:
            share_part = self.share_layer(cls_emb[ocemotion_ids, :]).unsqueeze(2)
            ocemotion_part = self.OCEMOTION_mid(cls_emb[ocemotion_ids, :]).unsqueeze(2)
            ocemotion_with_share = torch.cat([share_part, ocemotion_part], axis=2)
            ocemotion_mix = self.avg_pool(ocemotion_with_share).squeeze(2)
            ocemotion_out = self.OCEMOTION_out(ocemotion_mix)
        else:
            ocemotion_out = None
        if tnews_ids.size()[0] > 0:
            share_part = self.share_layer(cls_emb[tnews_ids, :]).unsqueeze(2)
            tnews_part = self.TNEWS_mid(cls_emb[tnews_ids, :]).unsqueeze(2)
            tnews_with_share = torch.cat([share_part, tnews_part], axis=2)
            tnews_mix = self.avg_pool(tnews_with_share).squeeze(2)
            tnews_out = self.TNEWS_out(tnews_mix)
        else:
            tnews_out = None
        return ocnli_out, ocemotion_out, tnews_out