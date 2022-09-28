# -*- coding: utf-8 -*-
# @Time    : 2021/9/25 16:04
# @Author  : wangqian

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
import torch
import random
import traceback
from optimizers import get_optimizer, LR_Scheduler
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from BertUtils import my_Dataset
from NLP_Utils import readFromJsonFile
from Config import Config
from Trainer import Trainer
from modeling_bert_cl_mlm import bert_cl_mlm
from torch.nn import DataParallel
torch.set_num_threads(1)#
import numpy as np
seed=0#
#python和numpy
random.seed(seed)
os.environ['PYTHONHASHSEED'] =str(seed)#消除hash算法的随机性
np.random.seed(seed)
#torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#当前使用gpu
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True


tk=BertTokenizer.from_pretrained(Config.pretrained_path)
train_data=my_Dataset(readFromJsonFile(Config.train_data_path),tk)
train_dataLoader=DataLoader(train_data,shuffle=True,batch_size=Config.batch_size,collate_fn=my_Dataset.collate)
model=DataParallel(bert_cl_mlm(Config.pretrained_path)).cuda()
#
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in Config.no_decay)], 'weight_decay': Config.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in Config.no_decay)], 'weight_decay': 0.0}
]
#定义优化器
optimizer = get_optimizer(  # sgdadamw
            'adamw', model,
            lr=Config.lr_begin,
            momentum=0.9,
            weight_decay=0.01)

lr_scheduler = LR_Scheduler(
            optimizer,
            0, 0,
            Config.epoch, Config.lr_begin, 1e-8,
            len(train_dataLoader),
            constant_predictor_lr=True
        )

opt=AdamW(optimizer_grouped_parameters,lr=Config.lr_begin)


scheduler=get_cosine_schedule_with_warmup(opt, num_warmup_steps=Config.warmup_step, num_training_steps=Config.epoch*len(train_dataLoader))

trainer=Trainer(model,opt,train_dataLoader,tk,scheduler,val_dataLoader=None)

def stay():
    import time
    print("定住！")
    time.sleep(10000000)
if __name__ == '__main__':
    try:
        trainer.train()
    except:
        trainer.save_model(Config.save_path+'interrupted/',Config.global_step/len(trainer.train_dataLoader),trainer.losses.total_weighted_loss,0,True)
        traceback.print_exc()
        if Config.multi_gpu==True:
            trainer.model.load_state_dict(Config.bestWeights)
        else:trainer.model.load_state_dict(Config.bestWeights)#最佳模型也保存
        trainer.save_model(Config.save_path+'best/',Config.last_better_step/len(trainer.train_dataLoader),Config.best_cp_loss,Config.best_val_score)
    stay()





