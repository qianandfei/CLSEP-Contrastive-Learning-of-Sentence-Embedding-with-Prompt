# -*- coding: utf-8 -*-
# @Time    : 2021/9/25 16:04
# @Author  : wangqian

from collections import OrderedDict
import torch
#训练调参专用
class Config:
    #数据相关
    pretrained_path= 'save/bert-large-uncased/'#bert-base-uncased/
    save_path='save/'
    train_data_path='data/simcse_train.json'
    val_data_path='data/STS_dev.tsv'
    test_data_path= 'data/test.json'

    #训练相关
    multi_gpu=True
    num_gpus=torch.cuda.device_count()
    is_nagetive_samples=True
    is_aug=True
    out_model='last_hidden'
    loss_mod='cross_s'
    neg_evidences=70
    fgm_e=0
    fp16=False
    loss_weights=OrderedDict([('cl',1),('mlm',0)])
    epoch=3#
    enc_maxlen =42#

    lr_begin=5e-5#
    lr_end=0
    warmup_step=0#
    batch_size=512#
    grad_accum_steps=1#3梯度累积
    weight_decay=0.01#权重衰减率
    no_decay = ['bias', 'LayerNorm.weight']
    #eval相关
    global_step=0
    loss_momentum=0.999#0.999loss滑动平均，便于显示，设0.99，前第220个值相对于当前值有0.1的影响力
    val_step=100#(1000000/batch_size)//20#多少个batch验证一次
    bak_step=val_step*250#多少batch备份一次
    early_stop_step=val_step*500#多少次验证都没提高就早停
    last_better_step=0#上一次验证得分增加时的step
    best_val_score=0
    bestWeights=None#保存最好的模型的权重
    best_cp_loss=None#保存最好的模型checkpoint的loss
    start_val_steps=100#多少epoch之后开始eval，可以节约eval时间
    start_val_loss=200#loss降到之后开始eval，可以节约eval时间 #1
    #句子senEva相关
    task_set='sts'#'transfer' 'full'
    mode='dev'






