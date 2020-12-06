#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:31:42 2020

@author: luokai
"""

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, f1_score
from transformers import BertModel, BertTokenizer


def get_f1(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    return marco_f1_score

def print_result(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    print(marco_f1_score)
    print(f"{'confusion_matrix':*^80}")
    print(confusion_matrix(l_t, l_p, ))
    print(f"{'classification_report':*^80}")
    print(classification_report(l_t, l_p, ))

def load_tokenizer(path_or_name):
    return BertTokenizer.from_pretrained(path_or_name)

def load_pretrained_model(path_or_name):
    return BertModel.from_pretrained(path_or_name)

def get_task_chinese(task_type):
    if task_type == 'ocnli':
        return '(中文原版自然语言推理)'
    elif task_type == 'ocemotion':
        return '(中文情感分类)'
    else:
        return '(今日头条新闻标题分类)'