# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 16:04
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
        if self.training:#PWVA
            if self.is_aug:
                mask = []
                mask_index = torch.nonzero(input_ids == 103)
                for index in mask_index:
                    mask.append(
                        [1] * (index[1]) + [0] + [1] * (inputs_embeds.size(1) - index[1] - 1) )# remove the impact of token [CLS]
                sen_mask = torch.tensor(mask, device='cuda')
                sen_mask = [sen_mask != 0][0].cuda()
                input = inputs_embeds * ((~sen_mask)[:, :, None])

                if self.num_embedding != 1:
                    self.aug1, self.aug2 = select_aug(inputs_embeds)
                    inputs_embeds=self.aug1*sen_mask[:, :, None]+input
                    self.num_embedding+=1
                else:
                    inputs_embeds=self.aug2*sen_mask[:, :, None]+input
                    self.num_embedding=0
            else:pass
        ##################################
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        if not self.is_aug:#if the PWVA is not performed, the original dropout is executed
            embeddings = self.dropout(embeddings)
        return embeddings
modeling_bert.BertEmbeddings=editedBertEmbeddings #replace the bert's embedding layer with ours


class cls_BertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        print('cls_pretraining can work')
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

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

        outputs = self.bert(
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
            return ((total_loss,) + output) if total_loss is not None else output,pooled_output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ),pooled_output




class Paraphraser(nn.Module):  #Paraphraser structure
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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_dim=4096, hidden_dim=512, out_dim=4096): # bottleneck structure
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class bert_cl_mlm(torch.nn.Module):
    def __init__(self,pretrained_path:str):
        super().__init__()
        cfg=BertConfig.from_pretrained(pretrained_path)
        self.bert=cls_BertForPreTraining(cfg)
        self.loss_for_cl=CrossEntropyLoss(ignore_index=-100)
        self.loss_for_mlm=CrossEntropyLoss(ignore_index=-100)
        self.model_opt = TextCNN.ModelConfig()
        self.textcnn=TextCNN.ModelCNN(
                     kernel_num=self.model_opt.kernel_num,
                     kernel_sizes=self.model_opt.kernel_sizes,
                        model_dim=self.model_opt.model_dim
                             )
        self.Paraphraser=Paraphraser(self.model_opt.model_dim if Config.is_cnn else cfg.hidden_size,4096)
        self.Bottleneck=Bottleneck()
        self.is_aug = Config.is_aug
        state_dict=torch.load(pretrained_path+'pytorch_model.bin')
        if 'bert.bert.embeddings.position_ids' in state_dict:
            print("加载权重：\n",self.load_state_dict(state_dict,strict=False))
        else:
            print("最初训练！")
            self.bert=cls_BertForPreTraining.from_pretrained(pretrained_path)

    def get_vecs(self,input_ids,attention_mask):# get the sentence embedding
        last_hidden_state,pooler=self.bert(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        if Config.out_model=='cls':
            vecs = last_hidden_state['hidden_states'][-1][:,0,:]
        else: # prompt sentence embedding
            mask_id = torch.nonzero(input_ids == 103)
            mask_out = torch.tensor([]).cuda()
            for id in mask_id:
                mask_out = torch.cat((mask_out, last_hidden_state['hidden_states'][-1][id[0], id[1], :].unsqueeze(0)),dim=0)
            vecs=mask_out

        return vecs

    def retr_cal_score_for_vecs(self,query_vecs:torch.Tensor,document_vecs:torch.Tensor,every_pair=True):
        query_vecs=functional.normalize(query_vecs, p=2, dim=1)
        document_vecs=functional.normalize(document_vecs, p=2, dim=1)
        return query_vecs.matmul(document_vecs.T) if every_pair else (query_vecs*document_vecs).sum(dim=-1)

    #对比学习损失
    def cl_cal_loss(self,inputs:dict):

        input_ids,attention_mask=inputs['input_ids'],inputs['attention_mask']
        labels=torch.arange(0,input_ids.shape[0],device='cuda')
        #过两次bert
        vec1=self.get_vecs(input_ids,attention_mask)
        vec2=self.get_vecs(input_ids,attention_mask)

        z1, z2 = self.Paraphraser(vec1), self.Paraphraser(vec2)
        p1, p2 = self.Bottleneck(z1), self.Bottleneck(z2)
        sims=self.retr_cal_score_for_vecs(p1,p2)
        sims=sims*20#
        loss=self.loss_for_cl(sims,labels)
        return loss

    #mlm loss
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




































































































