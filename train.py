#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:34:27 2020

@author: luokai
"""

import torch
from transformers import BertModel, BertTokenizer
import json
from utils import get_f1, print_result, load_pretrained_model, load_tokenizer
from net import Net
from data_generator import Data_generator
from calculate_loss import Calculate_loss


def train(epochs=20, batchSize=64, lr=0.0001, device='cuda:3', accumulate=True, a_step=16, load_saved=False, file_path='./saved_best.pt', use_dtp=False, pretrained_model='./bert_pretrain_model', tokenizer_model='bert-base-chinese'):
    device = device
    tokenizer = load_tokenizer(tokenizer_model)
    bert_model = load_pretrained_model(pretrained_model)
    my_net = torch.load(file_path) if load_saved else Net(bert_model)
    my_net.to(device)
    label_dict = dict()
    with open('./tianchi_datasets/label.json') as f:
        for line in f:
            label_dict = json.loads(line)
            break
    ocnli_train = dict()
    with open('./tianchi_datasets/OCNLI/train.json') as f:
        for line in f:
            ocnli_train = json.loads(line)
            break
    ocnli_dev = dict()
    with open('./tianchi_datasets/OCNLI/dev.json') as f:
        for line in f:
            ocnli_dev = json.loads(line)
            break
    ocemotion_train = dict()
    with open('./tianchi_datasets/OCEMOTION/train.json') as f:
        for line in f:
            ocemotion_train = json.loads(line)
            break
    ocemotion_dev = dict()
    with open('./tianchi_datasets/OCEMOTION/dev.json') as f:
        for line in f:
            ocemotion_dev = json.loads(line)
            break
    tnews_train = dict()
    with open('./tianchi_datasets/TNEWS/train.json') as f:
        for line in f:
            tnews_train = json.loads(line)
            break
    tnews_dev = dict()
    with open('./tianchi_datasets/TNEWS/dev.json') as f:
        for line in f:
            tnews_dev = json.loads(line)
            break
    train_data_generator = Data_generator(ocnli_train, ocemotion_train, tnews_train, label_dict, device, tokenizer)
    dev_data_generator = Data_generator(ocnli_dev, ocemotion_dev, tnews_dev, label_dict, device, tokenizer)
    loss_object = Calculate_loss(label_dict)
    optimizer=torch.optim.Adam(my_net.parameters(),lr=lr)
    best_dev_f1 = 0.0
    best_epoch = -1
    for epoch in range(epochs):
        my_net.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        train_ocnli_correct = 0
        train_ocemotion_correct = 0
        train_tnews_correct = 0
        train_ocnli_pred_list = []
        train_ocnli_gold_list = []
        train_ocemotion_pred_list = []
        train_ocemotion_gold_list = []
        train_tnews_pred_list = []
        train_tnews_gold_list = []
        cnt_train = 0
        while True:
            raw_data = train_data_generator.get_next_batch(batchSize)
            if raw_data == None:
                break
            data = dict()
            data['input_ids'] = raw_data['input_ids']
            data['token_type_ids'] = raw_data['token_type_ids']
            data['attention_mask'] = raw_data['attention_mask']
            data['ocnli_ids'] = raw_data['ocnli_ids']
            data['ocemotion_ids'] = raw_data['ocemotion_ids']
            data['tnews_ids'] = raw_data['tnews_ids']
            tnews_gold = raw_data['tnews_gold']
            ocnli_gold = raw_data['ocnli_gold']
            ocemotion_gold = raw_data['ocemotion_gold']
            if not accumulate:
                optimizer.zero_grad()
            ocnli_pred, ocemotion_pred, tnews_pred = my_net(**data)
            if use_dtp:
                tnews_kpi = 0.1 if len(train_tnews_pred_list) == 0 else train_tnews_correct / len(train_tnews_pred_list)
                ocnli_kpi = 0.1 if len(train_ocnli_pred_list) == 0 else train_ocnli_correct / len(train_ocnli_pred_list)
                ocemotion_kpi = 0.1 if len(train_ocemotion_pred_list) == 0 else train_ocemotion_correct / len(train_ocemotion_pred_list)
                current_loss = loss_object.compute_dtp(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold,
                                                   ocemotion_gold, tnews_kpi, ocnli_kpi, ocemotion_kpi)
            else:
                current_loss = loss_object.compute(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
            train_loss += current_loss.item()
            current_loss.backward()
            if accumulate and (cnt_train + 1) % a_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if not accumulate:
                optimizer.step()
            if use_dtp:
                good_tnews_nb, good_ocnli_nb, good_ocemotion_nb, total_tnews_nb, total_ocnli_nb, total_ocemotion_nb = loss_object.correct_cnt_each(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                tmp_good = sum([good_tnews_nb, good_ocnli_nb, good_ocemotion_nb])
                tmp_total = sum([total_tnews_nb, total_ocnli_nb, total_ocemotion_nb])
                train_ocemotion_correct += good_ocemotion_nb
                train_ocnli_correct += good_ocnli_nb
                train_tnews_correct += good_tnews_nb
            else:
                tmp_good, tmp_total = loss_object.correct_cnt(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
            train_correct += tmp_good
            train_total += tmp_total
            p, g = loss_object.collect_pred_and_gold(ocnli_pred, ocnli_gold)
            train_ocnli_pred_list += p
            train_ocnli_gold_list += g
            p, g = loss_object.collect_pred_and_gold(ocemotion_pred, ocemotion_gold)
            train_ocemotion_pred_list += p
            train_ocemotion_gold_list += g
            p, g = loss_object.collect_pred_and_gold(tnews_pred, tnews_gold)
            train_tnews_pred_list += p
            train_tnews_gold_list += g
            cnt_train += 1
            if (cnt_train + 1) % 1000 == 0:
                print('[', cnt_train + 1, '- th batch : train acc is:', train_correct / train_total, '; train loss is:', train_loss / cnt_train, ']')
        if accumulate:
            optimizer.step()
        optimizer.zero_grad()
        train_ocnli_f1 = get_f1(train_ocnli_gold_list, train_ocnli_pred_list)
        train_ocemotion_f1 = get_f1(train_ocemotion_gold_list, train_ocemotion_pred_list)
        train_tnews_f1 = get_f1(train_tnews_gold_list, train_tnews_pred_list)
        train_avg_f1 = (train_ocnli_f1 + train_ocemotion_f1 + train_tnews_f1) / 3
        print(epoch, 'th epoch train average f1 is:', train_avg_f1)
        print(epoch, 'th epoch train ocnli is below:')
        print_result(train_ocnli_gold_list, train_ocnli_pred_list)
        print(epoch, 'th epoch train ocemotion is below:')
        print_result(train_ocemotion_gold_list, train_ocemotion_pred_list)
        print(epoch, 'th epoch train tnews is below:')
        print_result(train_tnews_gold_list, train_tnews_pred_list)
        
        train_data_generator.reset()
        
        my_net.eval()
        dev_loss = 0.0
        dev_total = 0
        dev_correct = 0
        dev_ocnli_correct = 0
        dev_ocemotion_correct = 0
        dev_tnews_correct = 0
        dev_ocnli_pred_list = []
        dev_ocnli_gold_list = []
        dev_ocemotion_pred_list = []
        dev_ocemotion_gold_list = []
        dev_tnews_pred_list = []
        dev_tnews_gold_list = []
        cnt_dev = 0
        with torch.no_grad():
            while True:
                raw_data = dev_data_generator.get_next_batch(batchSize)
                if raw_data == None:
                    break
                data = dict()
                data['input_ids'] = raw_data['input_ids']
                data['token_type_ids'] = raw_data['token_type_ids']
                data['attention_mask'] = raw_data['attention_mask']
                data['ocnli_ids'] = raw_data['ocnli_ids']
                data['ocemotion_ids'] = raw_data['ocemotion_ids']
                data['tnews_ids'] = raw_data['tnews_ids']
                tnews_gold = raw_data['tnews_gold']
                ocnli_gold = raw_data['ocnli_gold']
                ocemotion_gold = raw_data['ocemotion_gold']
                ocnli_pred, ocemotion_pred, tnews_pred = my_net(**data)
                if use_dtp:
                    tnews_kpi = 0.1 if len(dev_tnews_pred_list) == 0 else dev_tnews_correct / len(
                        dev_tnews_pred_list)
                    ocnli_kpi = 0.1 if len(dev_ocnli_pred_list) == 0 else dev_ocnli_correct / len(
                        dev_ocnli_pred_list)
                    ocemotion_kpi = 0.1 if len(dev_ocemotion_pred_list) == 0 else dev_ocemotion_correct / len(
                        dev_ocemotion_pred_list)
                    current_loss = loss_object.compute_dtp(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold,
                                                           ocnli_gold,
                                                           ocemotion_gold, tnews_kpi, ocnli_kpi, ocemotion_kpi)
                else:
                    current_loss = loss_object.compute(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                dev_loss += current_loss.item()
                if use_dtp:
                    good_tnews_nb, good_ocnli_nb, good_ocemotion_nb, total_tnews_nb, total_ocnli_nb, total_ocemotion_nb = loss_object.correct_cnt_each(
                        tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                    tmp_good += sum([good_tnews_nb, good_ocnli_nb, good_ocemotion_nb])
                    tmp_total += sum([total_tnews_nb, total_ocnli_nb, total_ocemotion_nb])
                    dev_ocemotion_correct += good_ocemotion_nb
                    dev_ocnli_correct += good_ocnli_nb
                    dev_tnews_correct += good_tnews_nb
                else:
                    tmp_good, tmp_total = loss_object.correct_cnt(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                dev_correct += tmp_good
                dev_total += tmp_total
                p, g = loss_object.collect_pred_and_gold(ocnli_pred, ocnli_gold)
                dev_ocnli_pred_list += p
                dev_ocnli_gold_list += g
                p, g = loss_object.collect_pred_and_gold(ocemotion_pred, ocemotion_gold)
                dev_ocemotion_pred_list += p
                dev_ocemotion_gold_list += g
                p, g = loss_object.collect_pred_and_gold(tnews_pred, tnews_gold)
                dev_tnews_pred_list += p
                dev_tnews_gold_list += g
                cnt_dev += 1
                #if (cnt_dev + 1) % 1000 == 0:
                #    print('[', cnt_dev + 1, '- th batch : dev acc is:', dev_correct / dev_total, '; dev loss is:', dev_loss / cnt_dev, ']')
            dev_ocnli_f1 = get_f1(dev_ocnli_gold_list, dev_ocnli_pred_list)
            dev_ocemotion_f1 = get_f1(dev_ocemotion_gold_list, dev_ocemotion_pred_list)
            dev_tnews_f1 = get_f1(dev_tnews_gold_list, dev_tnews_pred_list)
            dev_avg_f1 = (dev_ocnli_f1 + dev_ocemotion_f1 + dev_tnews_f1) / 3
            print(epoch, 'th epoch dev average f1 is:', dev_avg_f1)
            print(epoch, 'th epoch dev ocnli is below:')
            print_result(dev_ocnli_gold_list, dev_ocnli_pred_list)
            print(epoch, 'th epoch dev ocemotion is below:')
            print_result(dev_ocemotion_gold_list, dev_ocemotion_pred_list)
            print(epoch, 'th epoch dev tnews is below:')
            print_result(dev_tnews_gold_list, dev_tnews_pred_list)

            dev_data_generator.reset()
            
            if dev_avg_f1 > best_dev_f1:
                best_dev_f1 = dev_avg_f1
                best_epoch = epoch
                torch.save(my_net, file_path)
            print('best epoch is:', best_epoch, '; with best f1 is:', best_dev_f1)
                
if __name__ == '__main__':
    print('---------------------start training-----------------------')
    train(batchSize=16, device='cuda:3', lr=0.0001, use_dtp=True)