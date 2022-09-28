from .lars import LARS
from .lars_simclr import LARS_simclr
from .larc import LARC
import torch
from Config import Config
from .lr_scheduler import LR_Scheduler
from transformers.optimization import AdamW

def get_optimizer(name, model, lr, momentum, weight_decay):

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr,
        'weight_decay': Config.weight_decay
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr,
        'weight_decay': 0.01
    }]#,{'name': 'decay',
     #  'params':[p for n, p in model.named_parameters() if not any(nd in n for nd in Config.no_decay)], 'weight_decay': Config.weight_decay},
    #{'name': 'no_decay',
     #'params':[p for n, p in model.named_parameters() if any(nd in n for nd in Config.no_decay)], 'weight_decay': 0.0}
    if name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,nesterov=True)
    elif   name=='adamw':
        #optimizer =torch.optim.Adam(parameters,lr=lr,betas=(0.9,0.999),weight_decay=weight_decay)#
        optimizer  =AdamW(parameters,lr=lr)

    elif name == 'lars_simclr': # Careful
        optimizer = LARS_simclr(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'larc':
        optimizer = LARC(
            torch.optim.SGD(
                parameters,
                lr=lr, 
                momentum=momentum, 
                weight_decay=weight_decay
            ),
            trust_coefficient=0.01,
            clip=False
        )
    else:
        raise NotImplementedError
    return optimizer



