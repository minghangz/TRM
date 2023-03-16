import torch
from torch import nn
from transformers import DistilBertModel
from torch.functional import F
from trm.data.datasets.utils import bert_embedding_batch
from transformers import DistilBertTokenizer
from torch.nn.utils.rnn import pad_sequence


class AttentivePooling(nn.Module):
    def __init__(self, feat_dim, att_hid_dim):
        super(AttentivePooling, self).__init__()

        self.feat_dim = feat_dim
        self.att_hid_dim = att_hid_dim

        self.feat2att = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        self.to_alpha = nn.Linear(self.att_hid_dim, 1, bias=False)
        # self.fc_phrase = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        # self.fc_sent = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)

    def forward(self, feats, global_feat, f_masks=None):
        """ Compute attention weights
        Args:
            feats: features where attention weights are computed; [num_sen, num_phr, D]
            global_feat: [num_sen, D]
            f_masks: mask for effective features; [num_sen, num_phr]
        """
        # check inputs
        assert len(feats.size()) == 3
        assert len(global_feat.size()) == 2
        assert f_masks is None or len(f_masks.size()) == 2

        # embedding feature vectors
        # feats = self.fc_phrase(feats)   # [num_sen,num_phr,hdim]
        # global_feat = self.fc_sent(global_feat).unsqueeze(-1) # [num_sen, hdim, 1]
        # alpha = torch.bmm(feats, global_feat) / math.sqrt(self.att_hid_dim) # [num_sen, num_phr, 1]
        # feats = torch.cat([global_feat.unsqueeze(1), feats], dim=1)
        attn_f = self.feat2att(feats)

        # compute attention weights
        dot = torch.tanh(attn_f)        # [num_sen,num_phr,hdim]
        alpha = self.to_alpha(dot)      # [num_sen,num_phr,1]
        if f_masks is not None:
            # alpha[:, 1:] = alpha[:, 1:].masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
            alpha = alpha.masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
        attw =  F.softmax(alpha, dim=1) # [num_sen, num_phr, 1]
        # attw = F.tanh(alpha.transpose(1,2), dim=2)
        attw = attw.squeeze(-1)

        return attw


class DistilBert(nn.Module):
    def __init__(self, joint_space_size, dataset, use_phrase, drop_phrase):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc_out1 = nn.Linear(768, joint_space_size)
        self.fc_out2 = nn.Linear(768, joint_space_size)
        self.dataset = dataset
        self.layernorm = nn.LayerNorm(768)
        self.aggregation = "avg"  # cls, avg
        self.use_phrase = use_phrase
        self.drop_phrase = drop_phrase
        if self.use_phrase:
            self.patt = AttentivePooling(joint_space_size, 128) # 128 is a magic number, remember to rewrite!
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.joint_space_size = joint_space_size
    
    def encode_single(self, query, word_len):
        N, word_length = query.size(0), query.size(1)
        attn_mask = torch.zeros(N, word_length, device=query.device)
        for i in range(N):
            attn_mask[i, :word_len[i]] = 1  # including [CLS] (first token) and [SEP] (last token)
        bert_encoding = self.bert(query, attention_mask=attn_mask)[0]  # [N, max_word_length, C]  .permute(2, 0, 1)
        if self.aggregation == "cls":
            query = bert_encoding[:, 0, :]  # [N, C], use [CLS] (first token) as the whole sentence feature
            query = self.layernorm(query)
            out_iou = self.fc_out1(query)
            out = self.fc_out2(query)
        elif self.aggregation == "avg":
            avg_mask = torch.zeros(N, word_length, device=query.device)
            for i in range(N):
                avg_mask[i, :word_len[i]] = 1       # including [CLS] (first token) and [SEP] (last token)
            avg_mask = avg_mask / (word_len.unsqueeze(-1))
            bert_encoding = bert_encoding.permute(2, 0, 1) * avg_mask  # use avg_pool as the whole sentence feature
            query = bert_encoding.sum(-1).t()  # [N, C]
            query = self.layernorm(query)
            out_iou = self.fc_out1(query)
            out = self.fc_out2(query)
        else:
            raise NotImplementedError
        return out, out_iou

    def forward(self, queries, wordlens):
        '''
        Average pooling over bert outputs among words to be sentence feature
        :param queries:
        :param wordlens:
        :param vid_avg_feat: B x C
        :return: list of [num_sent, C], len=Batch_size
        '''
        sent_feat = []
        sent_feat_iou = []
        for query, word_len in zip(queries, wordlens):  # each sample (several sentences) in a batch (of videos)
            out, out_iou = self.encode_single(query, word_len)
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        return sent_feat, sent_feat_iou
    
    
    def encode_sentences(self, sentences, phrases=None):
        sent_feat = []
        sent_feat_iou = []
        phrase_feat = []
        phrase_feat_iou = []
        phrase_weight = []
        phrase_masks = []

        stnc_query, stnc_len = bert_embedding_batch(sentences, self.tokenizer)
        for query, word_len in zip(stnc_query, stnc_len):  # each sample (several sentences) in a batch (of videos)
            out, out_iou = self.encode_single(query.cuda(), word_len.cuda())
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        if self.use_phrase == True:
            for bid, phrases_avid in enumerate(phrases):
                phrase_feat_avid = []
                phrase_feat_avid_iou = []
                
                phrase_query, phrase_len = bert_embedding_batch(phrases_avid, self.tokenizer)
                for query, word_len in zip(phrase_query, phrase_len):
                    out, out_iou = self.encode_single(query[:10].cuda(), word_len[:10].cuda())
                    pad_tensor = torch.zeros(10-len(out), self.joint_space_size) # this 10 is a magic number, remenber to rewrite!
                    pad_tensor = pad_tensor.to(out.device)
                    out = torch.cat((out, pad_tensor), 0)
                    out_iou = torch.cat((out_iou, pad_tensor), 0)

                    phrase_feat_avid.append(out)
                    phrase_feat_avid_iou.append(out_iou)

                phrase_feat_avid = pad_sequence(phrase_feat_avid, batch_first=True)
                phrase_feat_avid_iou = pad_sequence(phrase_feat_avid_iou, batch_first=True)
                phrase_mask = ((phrase_feat_avid != 0).long().sum(dim=-1) != 0).long().detach()
                
                if self.training and self.drop_phrase:
                    phrase_keep_weight = torch.zeros_like(phrase_mask).float().cuda()
                    for i, p in enumerate(phrases_avid):
                        for j, pp in enumerate(p[:10]):
                            phrase_keep_weight[i, j] = 0.9
                    drop_mask = torch.bernoulli(phrase_keep_weight)
                    phrase_mask = phrase_mask * drop_mask

                phrase_w = self.patt(phrase_feat_avid_iou, sent_feat_iou[bid], phrase_mask)
                
                phrase_feat.append(phrase_feat_avid)
                phrase_feat_iou.append(phrase_feat_avid_iou)
                phrase_weight.append(phrase_w)
                phrase_masks.append(phrase_mask)

            return sent_feat, sent_feat_iou, phrase_feat, phrase_feat_iou, phrase_weight, phrase_masks

        return sent_feat, sent_feat_iou


def build_text_encoder(cfg):
    joint_space_size = cfg.MODEL.TRM.JOINT_SPACE_SIZE
    dataset_name = cfg.DATASETS.NAME 
    use_phrase = cfg.MODEL.TRM.TEXT_ENCODER.USE_PHRASE
    drop_phrase = cfg.MODEL.TRM.TEXT_ENCODER.DROP_PHRASE
    return DistilBert(joint_space_size, dataset_name, use_phrase, drop_phrase)
