# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 16:04
# @Author  : wangqian
import random
import traceback
from optimizers import get_optimizer, LR_Scheduler
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_polynomial_decay_schedule_with_warmup
from BertUtils import my_Dataset,STS_Dataset
from NLP_Utils import readLine
from Config import Config
from Trainer import Trainer
from modeling_bert_cl_mlm import bert_cl_mlm

torch.set_num_threads(4)
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

seed=0

random.seed(seed)
os.environ['PYTHONHASHSEED'] =str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True


tk=BertTokenizer.from_pretrained(Config.pretrained_path)

train_data=my_Dataset(readLine(Config.train_data_path),tk)
val_data=STS_Dataset(Config.val_data_path,tk)

train_dataLoader=DataLoader(train_data,shuffle=True,batch_size=Config.batch_size,collate_fn=my_Dataset.collate)

val_dataLoader=DataLoader(val_data,batch_size=Config.batch_size*4,shuffle=False,collate_fn=STS_Dataset.collate)

model=bert_cl_mlm(Config.pretrained_path).cuda()


optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in Config.no_decay)], 'weight_decay': Config.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in Config.no_decay)], 'weight_decay': 0.0}
]

optimizer = get_optimizer(
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


scheduler=get_polynomial_decay_schedule_with_warmup(
            opt, num_warmup_steps=Config.warmup_step, num_training_steps=Config.epoch*len(train_dataLoader),lr_end=Config.lr_end
        )
trainer=Trainer(model,opt,train_dataLoader,tk,scheduler,val_dataLoader)

if __name__ == '__main__':

    try:
        trainer.train()
    except:
        trainer.save_model(Config.save_path+'interrupted/',Config.global_step/len(trainer.train_dataLoader),trainer.losses.total_weighted_loss,0,True)
        traceback.print_exc()
        trainer.model.load_state_dict(Config.bestWeights)
        trainer.save_model(Config.save_path+'best/',Config.last_better_step/len(trainer.train_dataLoader),Config.best_cp_loss,Config.best_val_score)






