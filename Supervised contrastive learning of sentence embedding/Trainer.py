# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 16:04
# @Author  : zhangweiqi

from Config import Config
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from BertUtils import Multi_accum_ema_loss
from transformers import BertTokenizer
import os
import shutil
from modeling_bert_cl_mlm import bert_cl_mlm
from torch.cuda.amp import autocast,GradScaler
from contextlib import  nullcontext
scaler = GradScaler()
import sys
import  train_test_eval
sys.setrecursionlimit(5000000)#设置最大递归深度，计算LCS需要

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.25, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer():
    def __init__(self, model:bert_cl_mlm,  #模型
                 opt:torch.optim.Optimizer,  #优化器
                 train_dataLoader:DataLoader,  #训练数据
                 tk:BertTokenizer,  #分词器
                 scheduler:torch.optim.lr_scheduler.LambdaLR,
                 val_dataLoader:DataLoader,  #验证数据
                 ):
        self.model=model
        self.fgm = FGM(model)
        self.opt=opt
        self.train_dataLoader=train_dataLoader
        self.tk=tk
        self.val_dataLoader=val_dataLoader
        self.losses=Multi_accum_ema_loss(list(Config.loss_weights.values()))
        self.scheduler=scheduler
    #训练
    def train(self):
        self.model.train()
        for epoch in range(Config.epoch):
            epoch_iterator=tqdm(self.train_dataLoader,ncols=140,mininterval=0.3)
            for inputs in epoch_iterator:
                Config.global_step+=1
                if not self.model.training:
                    self.model.train()
                contex=autocast if Config.fp16 else nullcontext
                with contex():
                    if Config.multi_gpu==True:
                        p1,p2,p_neg=self.model(inputs)#
                        labels=torch.arange(0,p1.shape[0],device='cuda')
                        sims=torch.cat((self.model.module.retr_cal_score_for_vecs(p1,p2),self.model.module.retr_cal_score_for_vecs(p1,p_neg)),dim=1)#
                        sims=sims*20
                        cl_loss=self.model.module.loss_for_cl(sims,labels)#
                    else:cl_loss=self.model(inputs)
                    mlm_loss=self.model.mlm_cal_loss(inputs) if Config.loss_weights['mlm']!=0 else torch.tensor(0,device='cuda')
                    weighted_loss=cl_loss*Config.loss_weights['cl']+mlm_loss*Config.loss_weights['mlm']
                    weighted_loss/=(2 if Config.fgm_e!=0 else 1)#若加对抗，两次需平均
                    self.losses.add_accum_loss([cl_loss.item()/Config.grad_accum_steps,mlm_loss.item()/Config.grad_accum_steps])
                scaler.scale(weighted_loss).backward() if Config.fp16 else weighted_loss.backward()#反向传播

                if Config.fgm_e!=0:
                    self.fgm.attack(epsilon=Config.fgm_e)
                    with contex():
                        if Config.multi_gpu == True:
                            p1, p2, p_neg = self.model(inputs)  # loss 不能放在forward里面计算  在计算交叉熵的时候实际的batch只用了一半
                            labels = torch.arange(0, p1.shape[0], device='cuda')
                            sims = torch.cat((self.model.module.retr_cal_score_for_vecs(p1, p2), (self.model.module.retr_cal_score_for_vecs(p1,p_neg) + self.model.module.retr_cal_score_for_vecs(p2, p_neg)) / 2),dim=1)
                            sims = sims * 20
                            cl_loss = self.model.module.loss_for_cl(sims,labels)

                        else:
                            cl_loss = self.model(inputs)
                        mlm_loss = self.model.mlm_cal_loss(inputs) if Config.loss_weights['mlm'] != 0 else torch.tensor(
                            0, device='cuda')
                        weighted_loss = cl_loss * Config.loss_weights['cl'] + mlm_loss * Config.loss_weights['mlm']
                        weighted_loss /= (2 if Config.fgm_e != 0 else 1)  # 若加对抗，两次需平均
                        self.losses.add_accum_loss(
                            [cl_loss.item() / Config.grad_accum_steps, mlm_loss.item() / Config.grad_accum_steps])
                    self.fgm.restore()

                if Config.global_step%Config.grad_accum_steps==0:#需要更新了
                    (scaler.step(self.opt),scaler.update()) if Config.fp16 else self.opt.step()#
                    self.opt.zero_grad()#
                    weighted_loss=0
                    (cl_show_loss,mlm_show_loss),weighted_show_loss=self.losses.update_and_get_ema_losses()
                    epoch_iterator.set_description_str(f"epoch：{epoch+1}",refresh=False)
                    epoch_iterator.set_postfix_str("weighted_loss：{:.3e}，cl_loss：{:.3e}，mlm_loss：{:.3e}，lr：{:.3e}".format(
                        weighted_show_loss,cl_show_loss,mlm_show_loss,self.scheduler.get_last_lr()[0],refresh=False))#
                self.scheduler.step()
                #验证

                if Config.global_step%Config.val_step==0 and Config.global_step >= Config.start_val_steps:#进行验证
                    metrics=train_test_eval.eval_in_train(self.model,Config,self.tk)

                    if metrics>Config.best_val_score:
                        Config.best_val_score=metrics
                        Config.last_better_step=Config.global_step
                        Config.bestWeights={k:v.cpu().clone() for k, v in self.model.state_dict().items()}#保存最佳权重
                        Config.best_cp_loss=self.losses.total_weighted_loss
                        print(f"\n找到更佳模型，当前得分：{Config.best_val_score}\n")
                    else:
                        print(f"\n未改进，当前得分：{metrics}，最佳得分：{Config.best_val_score}\n")
                        if Config.global_step-Config.last_better_step>=Config.early_stop_step:
                            self.save_model(Config.save_path+'last/',Config.global_step/len(self.train_dataLoader),self.losses.total_weighted_loss,metrics,clear_path=True)#最后模型备份一下
                            print(f"\n长时间未改进，提前停止训练\n")
                            if Config.multi_gpu==True:
                                self.model.load_state_dict(Config.bestWeights)   
                            else:self.model.load_state_dict(Config.bestWeights)#先加载最佳权重

                            self.save_model(Config.save_path,Config.last_better_step/len(self.train_dataLoader),Config.best_cp_loss,Config.best_val_score)
                            return
                #备份
                if Config.global_step%Config.bak_step==0:
                    self.save_model(Config.save_path+'bak/',Config.global_step/len(self.train_dataLoader),self.losses.total_weighted_loss,0,clear_path=False)
                    print("\n备份成功\n")
            epoch_iterator.close()
        #能训练到终止
        self.save_model(Config.save_path+'last/',Config.global_step/len(self.train_dataLoader),self.losses.total_weighted_loss,0,clear_path=True)#最后模型备份一下
        print(f"\n训练完成\n")
        if Config.multi_gpu==True:
            self.model.load_state_dict(Config.bestWeights)   
        else:self.model.load_state_dict(Config.bestWeights)#先加载最佳权重
        self.save_model(Config.save_path,Config.last_better_step/len(self.train_dataLoader),Config.best_cp_loss,Config.best_val_score)

    def save_model(self,path,epoch,loss,metrics,clear_path=False):#
        if clear_path and os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        savePath=path+f"epoch-{epoch:.1f}-loss-{loss:.3e}-metrics-{metrics:.3e}/"
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        if Config.multi_gpu==True:
            torch.save(self.model.state_dict(),savePath+'pytorch_model.bin')
            self.model.module.bert.config.save_pretrained(savePath)
        else:
            torch.save(self.model.state_dict(),savePath+'pytorch_model.bin')
            self.model.bert.config.save_pretrained(savePath)
        self.tk.save_vocabulary(savePath)

