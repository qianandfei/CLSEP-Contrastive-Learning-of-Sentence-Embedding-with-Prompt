# -*- coding: utf-8 -*-
# @Time    : 2021/9/25 16:04
# @Author  : wangqian
import torch

def gausssiam_white(sentence):#√
    snr=torch.randint(6,12,(sentence.shape[:2])).cuda()[:,:,None]
    P_signal=(sentence*sentence).sum(dim=-1,keepdim=True)/sentence.shape[-1]
    P_noise = P_signal / (10 ** (snr / 10.0))
    return sentence+torch.randn(sentence.shape).cuda()*torch.sqrt(P_noise)

def fft_ifft(sentence):#
    fft_a = torch.fft.fft(sentence-torch.mean(sentence))
    a_f = torch.fft.ifft(fft_a)
    return a_f.real

def add_noise(sentence):#对
    max_a,_=torch.max(sentence,dim=-1,keepdim=True)
    min_a,_=torch.min(sentence,dim=-1,keepdim=True)
    samples=torch.rand(sentence.shape).cuda()
    scale=(max_a*0.8-min_a)
    samples=samples*scale+min_a
    s_with_bg = sentence+samples*torch.Tensor(1).uniform_(0, 0.01).cuda()
    return s_with_bg

def wangqian(sentence,p1=0.2,p2=0.2,p3=0.3,p4=0.3):
    res1,res2,res3,res4=gausssiam_white(sentence),fft_ifft(sentence),add_noise(sentence),torch.dropout(sentence,p=0.3,train=True)
    mask=torch.rand(res1.shape[:2]).cuda()[:,:,None]
    mask1,mask2,mask3,mask4=(0<=mask)&(mask<p1),(p1<=mask)&(mask<p1+p2),(p1+p2<=mask)&(mask<p1+p2+p3),(p1+p2+p3<=mask)&(mask<=p1+p2+p3+p4)
    return  res1*mask1+res2*mask2+res3*mask3+res4*mask4

def select_aug(sentence,p1=0.4):

    inputs_embeds=sentence
    s1=wangqian(inputs_embeds)#过两次增强
    mask=torch.rand(s1.shape[:2]).cuda()[:,:,None]
    mask1=(0<=mask)&(mask<=p1)#mask表示选择未增强的
    original=inputs_embeds*mask1
    s1=s1*(~mask1)
    s1=original+s1
    return s1


if __name__ == "__main__":
    sentence=torch.Tensor(100).uniform_(1,2)
    s1=gausssiam_white(sentence)
    print()