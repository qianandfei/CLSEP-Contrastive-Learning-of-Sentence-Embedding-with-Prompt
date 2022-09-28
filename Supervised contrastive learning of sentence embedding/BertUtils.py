# -*- coding: utf-8 -*-
# @Time    : 2021/9/25 16:04
# @Author  : wangqian

import string
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from zhon import hanzi
from Config import Config




def getPuncs():
    return set(hanzi.punctuation + string.punctuation+' \n\r')
Puncs=getPuncs()
stopPuncs=set(' ,，”、>》}】:;!?。：；？！.\n\r')

def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=Config.enc_maxlen#max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))#如果是prompt  padding 放在前面更有利于mask的表达
    return torch.tensor(ls,device='cuda') if returnTensor else ls




class my_Dataset(Dataset):
    #传入句子对列表  data--275601
    def __init__(self,data:list,tk:BertTokenizer,for_test=False):
        super().__init__()
        self.tk=tk
        self.origin_data=data#
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
            if rands[idx]<0.15:#
                ngram=np.random.choice([1,2,3], p=[0.6,0.3,0.1])#
                if ngram==3 and len(rands)<7:#
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
                    rands[idx]=1#
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




    #耗时操作在此进行，可用上多进程
    def __getitem__(self, item):
        now=self.origin_data[item]
        s1=now[0]
        s2=now[1]
        neg=now[2]
        s1=self.tk.encode(s1,max_length=Config.enc_maxlen,truncation=True)
        s2 = self.tk.encode(s2, max_length=Config.enc_maxlen, truncation=True)
        neg = self.tk.encode(neg, max_length=Config.enc_maxlen, truncation=True)
        s1_ids=s1
        s2_ids = s2
        neg_ids = neg
        return {'s1_ids':s1_ids,'s2_ids':s2_ids,'neg_ids':neg_ids}

    @classmethod
    def collate(cls,batch):
        now=dict()
        for k,padding_v in zip(['s1_ids','s2_ids','neg_ids'],[0,0,0]):
            v=[i[k] for i in batch]
            v=paddingList(v,padding_v,returnTensor=True)
            now[k]=v
        now['s1_attention_mask']=(now['s1_ids']!=0)
        now['s2_attention_mask'] = (now['s2_ids'] != 0)
        now['neg_attention_mask'] = (now['neg_ids'] != 0)
        return now

#为某些值加入滑动平均
class EMA():
    def __init__(self,decay):
        self.decay=decay
        self.val=0
        self.beta_exp=1
        self.ema_val=None#滑动平均后的值

    def update(self, now_val):
        self.val=self.val*self.decay+(1-self.decay)*now_val
        self.beta_exp*=self.decay
        self.ema_val=self.val/(1-self.beta_exp)
        return self.ema_val

#适应于多任务下，含梯度累积、指数滑动平均的loss多显示
class Multi_accum_ema_loss():
    def __init__(self,weights:list):
        self.weight_list=weights#每个loss的权重
        self.accum_loss_list=[0]*len(weights)#用于梯度累积过程累加的loss
        self.EMA_list=[EMA(Config.loss_momentum) for i in range(len(weights))]#滑动平均后用于显示的loss
        self.total_weighted_loss=None#分别滑动平均，再加权平均后的loss

    def add_accum_loss(self,loss_list:list):
        assert len(self.accum_loss_list)==len(loss_list)
        for i in range(len(self.accum_loss_list)):
            self.accum_loss_list[i]+=loss_list[i]

    #更新并获取滑动平均后loss，清空累积loss
    def update_and_get_ema_losses(self):
        res=[]
        self.total_weighted_loss=0
        for i in range(len(self.EMA_list)):
            now=self.EMA_list[i].update(self.accum_loss_list[i])#每个loss进行滑动平均
            res.append(now)
            self.total_weighted_loss+=(now*self.weight_list[i])#加权求和
        self.accum_loss_list=[0]*len(self.weight_list)#清空显示loss的累积
        return res,self.total_weighted_loss




