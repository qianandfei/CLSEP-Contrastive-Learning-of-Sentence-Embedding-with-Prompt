# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 16:04
# @Author  : wangqian
from collections import OrderedDict
class Config:
    pretrained_path= 'save/bert-base-uncased/'
    save_path='save/'
    train_data_path='data/wiki.txt'
    val_data_path='data/STS_dev.tsv'
    test_data_path= 'data/test.json'

    #Training
    is_nagetive_samples=True
    is_aug=True
    out_model='prompt'#
    loss_mod='cross_s'#
    fgm_e=0#
    fp16=False#
    loss_weights=OrderedDict([('cl',1),('mlm',0)])#
    epoch=1#
    enc_maxlen = 50
    lr_begin=3e-5#
    lr_end=0
    warmup_step=0#
    batch_size=64#
    grad_accum_steps=1#
    weight_decay=0.01#
    no_decay = ['bias', 'LayerNorm.weight']
    #evaluation
    global_step=0
    loss_momentum=0.999#
    val_step=111#
    bak_step=val_step*250#
    early_stop_step=val_step*100000#
    last_better_step=0#
    best_val_score=0
    bestWeights=None#
    best_cp_loss=None#
    start_val_steps=1#
    start_val_loss=200#
    #senEva
    task_set='sts'#
    mode='dev'






