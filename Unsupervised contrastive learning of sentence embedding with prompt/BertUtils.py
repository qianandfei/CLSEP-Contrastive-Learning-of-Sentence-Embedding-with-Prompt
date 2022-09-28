# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 16:04
# @Author  : wangqian

import copy
import os
import random
import string
import re
from itertools import chain
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from zhon import hanzi

from Config import Config




def getPuncs():
    return set(hanzi.punctuation + string.punctuation+' \n\r')
Puncs=getPuncs()
stopPuncs=set(' ,，”、>》}】:;!?。：；？！.\n\r')

def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])#最小padding长度效果最好78.80
    for i in range(len(ls)):
        ls[i]=[val]*(maxLen-len(ls[i]))+ls[i]
    return torch.tensor(ls,device='cuda') if returnTensor else ls




class my_Dataset(Dataset):
    #传入句子对列表
    def __init__(self,data:list,tk:BertTokenizer,for_test=False):
        super().__init__()
        self.tk=tk
        self.origin_data=data#[0:100000]
        self.tkNum=self.tk.vocab_size

    def __len__(self):
        return len(self.origin_data)

    def random_mask(self,text_ids):
        if len(text_ids)==0:
            return [],[]
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        idx=0
        while idx<len(rands):
            if rands[idx]<0.15:#需要mask
                ngram=np.random.choice([1,2,3], p=[0.6,0.3,0.1])#若要mask，进行x_gram mask的概率
                if ngram==3 and len(rands)<7:#太大的gram不要应用于过短文本
                    ngram=2
                if ngram==2 and len(rands)<4:
                    ngram=1
                L=idx+1
                R=idx+ngram#最终需要mask的右边界（开）
                while L<R and L<len(rands):
                    rands[L]=np.random.random()*0.15#强制mask
                    L+=1
                idx=R
                if idx<len(rands):
                    rands[idx]=1#禁止mask片段的下一个token被mask，防止一大片连续mask
            idx+=1

        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(self.tk.mask_token_id)
                output_ids.append(i)#mask预测自己
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)#自己预测自己
            elif r < 0.15:
                input_ids.append(np.random.randint(0,self.tkNum))
                output_ids.append(i)#随机的一个词预测自己，有小概率抽到自己和特殊符号
            else:
                input_ids.append(i)
                output_ids.append(-100)#保持原样不预测

        return input_ids, output_ids



    def get_only_chars(self,line):
        clean_line = ""
        line = line.replace("'", "")
        line = line.replace("-", " ")  # replace hyphens with spaces
        line = line.replace("\t", " ")
        line = line.replace("\n", " ")
        line = line.lower()


        new=[]

        for word in line.split():
            if word not in self.stop_words:
                new.append(word)
        for i in range(len(new)):
            clean_line+=' '+new[i]

        clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
        return clean_line
    #耗时操作在此进行，可用上多进程
    def __getitem__(self, item):
        now=self.origin_data[item].replace("\""," ")
        template = 'This sentence : " " means [MASK].'
        template_ids = self.tk.encode(template, max_length=Config.enc_maxlen, truncation=True)
        sentence_id = self.tk.encode(now, max_length=Config.enc_maxlen - len(template_ids), truncation=True,
                                     add_special_tokens=False)
        input_ids = template_ids[:5] + sentence_id + template_ids[5:]
        mlm_input_ids, mlm_labels = self.random_mask(input_ids)

        return {'input_ids':input_ids,'mlm_input_ids':mlm_input_ids,'mlm_labels':mlm_labels}

    @classmethod
    def collate(cls,batch):
        now=dict()
        for k,padding_v in zip(['input_ids','mlm_input_ids','mlm_labels'],[0,0,-100]):
            v=[i[k] for i in batch]
            v=paddingList(v,padding_v,returnTensor=True)
            now[k]=v
        now['attention_mask']=(now['input_ids']!=0)
        return now

#为某些值加入滑动平均
class EMA():
    def __init__(self,decay):
        self.decay=decay
        self.val=0
        self.beta_exp=1
        self.ema_val=None#

    def update(self, now_val):
        self.val=self.val*self.decay+(1-self.decay)*now_val
        self.beta_exp*=self.decay
        self.ema_val=self.val/(1-self.beta_exp)
        return self.ema_val

class Multi_accum_ema_loss():
    def __init__(self,weights:list):
        self.weight_list=weights#
        self.accum_loss_list=[0]*len(weights)#
        self.EMA_list=[EMA(Config.loss_momentum) for i in range(len(weights))]#
        self.total_weighted_loss=None#

    def add_accum_loss(self,loss_list:list):
        assert len(self.accum_loss_list)==len(loss_list)
        for i in range(len(self.accum_loss_list)):
            self.accum_loss_list[i]+=loss_list[i]

    def update_and_get_ema_losses(self):
        res=[]
        self.total_weighted_loss=0
        for i in range(len(self.EMA_list)):
            now=self.EMA_list[i].update(self.accum_loss_list[i])#
            res.append(now)
            self.total_weighted_loss+=(now*self.weight_list[i])#
        self.accum_loss_list=[0]*len(self.weight_list)#
        return res,self.total_weighted_loss



class STS_Dataset(Dataset):

    def __init__(self,path:str,tk:BertTokenizer):
        super().__init__()
        self.tk=tk
        data=readForSL(path,10)
        s1,s2,label=data[-3:]
        s1,s2,label=s1[0][1:],s2[0][1:],[float(i) for i in label[0][1:]]
        self.origin_data=[]
        for i,j,k in zip(s1,s2,label):
            self.origin_data.append([self.tk.encode(i,max_length=Config.enc_maxlen,truncation=True),
                                     self.tk.encode(j,max_length=Config.enc_maxlen,truncation=True),
                                     k])

    def __len__(self):
        return len(self.origin_data)

    def __getitem__(self, item):
        now=self.origin_data[item]
        return {'s1':now[0],'s2':now[1]}

    @classmethod
    def collate(cls,batch):
        now=dict()
        for k,padding_v in zip(['s1','s2'],[0,0]):
            v=[i[k] for i in batch]
            v=paddingList(v,padding_v,returnTensor=True)
            now[k]=v
        return now




