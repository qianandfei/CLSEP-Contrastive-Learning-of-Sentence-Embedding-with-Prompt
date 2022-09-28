# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 16:04
# @Author  : wangqian


import sys
import logging
from prettytable import PrettyTable
import torch
from BertUtils import paddingList
from SentEval.senteval.engine import SE

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
sys.path.insert(0, PATH_TO_SENTEVAL)

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

@torch.no_grad()
def eval_in_train(model,Config,tk):
    # Set up the tasks
    model.eval()
    if Config.task_set == 'sts':
        Config.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif Config.task_set == 'transfer':
        Config.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif Config.task_set == 'full':
        Config.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        Config.tasks += [ 'SST2']

    # Set params for SentEval
    if Config.mode == 'dev' or Config.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 2}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 4}
    elif Config.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adamw', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        #get the embedding of sentences
        input_ids_dic = []
        for sentence in sentences:
            sentence = sentence.replace("\"", "")
            template = 'The meaning of " " is [MASK].'
            template_ids = tk.encode(template, max_length=Config.enc_maxlen, truncation=True)
            sentence_id = tk.encode(sentence, max_length=Config.enc_maxlen - len(template_ids), truncation=True,
                                    add_special_tokens=False)

            input_ids = template_ids[:5] + sentence_id + template_ids[5:]
            input_ids_dic.append({'input_ids': input_ids})

        now = dict()
        for k, padding_v in zip(['input_ids'], [0]):
            v = [i[k] for i in input_ids_dic]
            v = paddingList(v, padding_v, returnTensor=True)
            now[k] = v
        now['attention_mask'] = (now['input_ids'] != 0)


        input_ids = now['input_ids']
        attention_masks = now['attention_mask']
        with torch.no_grad():
            output=model.get_vecs(input_ids,attention_masks)
        return output.cpu()
    #get the results of senteval
    results = {}
    for task in Config.tasks:
        se = SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if Config.mode == 'dev':
        print("------ %s ------" % (Config.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16','STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))

            else:scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        # task_names = []
        # scores = []
        # for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
        #     task_names.append(task)
        #     if task in results:
        #         scores.append("%.2f" % (results[task]['devacc']))
        #     else:
        #         scores.append("0.00")
        # task_names.append("Avg.")
        # scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        # print_table(task_names, scores)
        #


    elif Config.mode == 'test' or Config.mode == 'fasttest':
        print("------ %s ------" % (Config.mode))
        task_names_sts = []
        scores_sts = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names_sts.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores_sts.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores_sts.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores_sts.append("0.00")
        task_names_sts.append("Avg.")
        scores_sts.append("%.2f" % (sum([float(score) for score in scores_sts]) / len(scores_sts)))
        print_table(task_names_sts, scores_sts)

        task_names_transfer = []
        scores_transfer = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names_transfer.append(task)
            if task in results:
                scores_transfer.append("%.2f" % (results[task]['acc']))
            else:
                scores_transfer.append("0.00")
        task_names_transfer.append("Avg.")
        scores_transfer.append("%.2f" % (sum([float(score) for score in scores_transfer]) / len(scores_transfer)))
        print_table(task_names_transfer, scores_transfer)
        scores=[float(score) for score in scores_sts]

    return float(scores[-3])

  



    


    














