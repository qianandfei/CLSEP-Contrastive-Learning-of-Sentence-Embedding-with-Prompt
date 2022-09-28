# -*- coding: utf-8 -*-
# @Time    : 2021/9/25 16:04
# @Author  : wangqian

import torch
from torch import nn
from torch.nn import functional
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings,BertModel,BertPreTrainingHeads,BertForPreTrainingOutput,BertForPreTraining
from transformers.models.bert import modeling_bert
from torch.utils.data import DataLoader
import scipy.stats
from text_aug import select_aug
from Config import Config
#重写bert embedding，方便加入自己的数据增强
class editedBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        print('成功使用自定义embedding！')
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.num_embedding=0
        self.aug1=0
        self.aug2=0
        self.is_aug=Config.is_aug


    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
  
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)


        ##################################
        if self.training:#数据增强只在训练阶段进行
            if self.is_aug:

                mask = []
                mask_index = torch.nonzero(input_ids == 101)
                for index in mask_index:
                    mask.append([1] * (index[1]) + [0] + [1] * (inputs_embeds.size(1) - index[1] - 2)+[0])#移除增强对cls的影响
                sen_mask = torch.tensor(mask, device='cuda')
                sen_mask = [sen_mask != 0][0]
                #print("the mask shape is",sen_mask.size())
               # print(inputs_embeds.size())
                input = inputs_embeds * ((~sen_mask)[:, :, None])

                self.aug1 = select_aug(inputs_embeds)
                inputs_embeds=self.aug1*sen_mask[:,:,None]+input


            else:pass
        ##################################
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        if not self.is_aug:
            embeddings = self.dropout(embeddings)
        return embeddings

modeling_bert.BertEmbeddings=editedBertEmbeddings#替换成自己的


class cls_BertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        print('cls_pretraining can work')
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()
   
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))   可删除 文档一些信息
    #@replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




class Paraphraser(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=4096):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class bert_cl_mlm(torch.nn.Module):
    def __init__(self,pretrained_path:str):
        super().__init__()
        cfg=BertConfig.from_pretrained(pretrained_path)
        self.bert=cls_BertForPreTraining(cfg)
        self.loss_for_cl=CrossEntropyLoss(ignore_index=-100)
        self.loss_for_mlm=CrossEntropyLoss(ignore_index=-100)
        self.Paraphraser=Paraphraser(self.model_opt.model_dim if Config.is_cnn else cfg.hidden_size, 4096)
        state_dict=torch.load(pretrained_path+'pytorch_model.bin')
        if 'bert.bert.embeddings.position_ids' in state_dict:
            print("加载权重：\n",self.load_state_dict(state_dict,strict=False))
        else:
            print("最初训练！")
            self.bert=cls_BertForPreTraining.from_pretrained(pretrained_path)


    #池化得到向量
    def get_vecs(self,input_ids,attention_mask): #使用最后一层prompt输出78.96



        last_hidden_state=self.bert(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True,return_dict=True)#['hidden_states'][-1],bert_outputs
        if Config.out_model=='cls':
            vecs = last_hidden_state['hidden_states'][-1][:,0,:]


        else:
            if self.training:
                last_hidden=last_hidden_state['hidden_states'][-1]/2+last_hidden_state['hidden_states'][0]/2+last_hidden_state['hidden_states'][-2]/3#padding部分向量置零  若采用此种方式就用first+last的平均
                last_state=last_hidden*attention_mask[:, :, None]#移除padding效果更好
                #get the sentence embedding by meaning the word vectors
                vecs = last_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                if Config.is_cnn:
                    vecs=last_hidden
            else:
                last_state = last_hidden_state['hidden_states'][-1] * attention_mask[:, :, None]
                vecs=last_state.sum(dim=1)/attention_mask.sum(dim=1,keepdim=True)#attention_mask.sum计算每个句子的长度然后求出对应的平均向量
        return vecs

    def retr_cal_score_for_vecs(self,query_vecs:torch.Tensor,document_vecs:torch.Tensor,every_pair=True):
        query_vecs=functional.normalize(query_vecs, p=2, dim=1)#归一化为单位向量[bs,hiden_len]
        document_vecs=functional.normalize(document_vecs, p=2, dim=1)#[bs,hiden_len]
        return query_vecs.matmul(document_vecs.T) if every_pair else (query_vecs*document_vecs).sum(dim=-1)


    #对比学习损失
    def forward(self,inputs:dict):
        #过两次bert
        vec1=self.get_vecs(inputs["s1_ids"],inputs["s1_attention_mask"])
        vec2=self.get_vecs(inputs["s2_ids"],inputs["s2_attention_mask"])
        vec_neg=self.get_vecs(inputs["neg_ids"],inputs["neg_attention_mask"])
        z1, z2,z_neg = self.Paraphraser(vec1), self.Paraphraser(vec2),self.Paraphraser(vec_neg)
        return z1, z2,z_neg
        

    #mlm损失
    def mlm_cal_loss(self,inputs:dict):
        input_ids,mlm_labels,attention_mask=inputs['mlm_input_ids'],inputs['mlm_labels'],inputs['attention_mask']
        logits=self.bert(input_ids=input_ids,attention_mask=attention_mask)['prediction_logits']
        loss=self.loss_for_mlm(logits.view(-1,logits.shape[-1]),mlm_labels.view(-1))
        return loss


    @torch.no_grad()
    def STS_val(self,val_dataLoader:DataLoader):
        if self.training:
            self.eval()
        true=[i[-1] for i in val_dataLoader.dataset.origin_data]
        pred=[]
        for i in val_dataLoader:
            s1,s2=i['s1'],i['s2']
            vec1=self.get_vecs(s1,s1!=0)
            vec2=self.get_vecs(s2,s2!=0)
            sim=torch.cosine_similarity(vec1,vec2,dim=-1)
            pred+=sim.tolist()
        return {'pred':pred,'metrics':scipy.stats.spearmanr(pred,true).correlation}




































































































