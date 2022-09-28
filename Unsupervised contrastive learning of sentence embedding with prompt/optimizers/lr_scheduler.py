import torch
import numpy as np
from Config import Config

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr):
        self.base_lr = base_lr
        self.iter_per_epoch=iter_per_epoch
        self.constant_predictor_lr = constant_predictor_lr
        self.warmup_iter = int(iter_per_epoch * warmup_epochs)
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, self.warmup_iter)
        decay_iter = iter_per_epoch * int(num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        small_lr_after_warup=final_lr+0.05*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))#0.02
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))##cosine_lr_schedule
        self.optimizer = optimizer
        self.iter = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = Config.lr_begin#1
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
            if self.iter<self.warmup_iter:param_group['momentum']=0.9#warmup_iter
            else:param_group['momentum']=0.9#
        self.iter =self.iter+Config.grad_accum_steps
        return lr

if __name__ == "__main__":
    import torchvision
    model = torchvision.models.resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    epochs = 100
    n_iter = 1000
    scheduler = LR_Scheduler(optimizer, 10, 1, epochs, 3, 0, n_iter)
    import matplotlib.pyplot as plt
    lrs = []
    for epoch in range(epochs):
        for it in range(n_iter):
            lr = scheduler.step()
            lrs.append(lr)
    plt.plot(lrs)
    plt.show()