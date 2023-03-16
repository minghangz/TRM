import torch
from torch import nn
from torch.functional import F
from torch.nn.utils.rnn import pad_sequence
from .featpool import build_featpool  # downsample 1d temporal features to desired length
from .feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .loss import build_contrastive_loss
from .loss import build_bce_loss
from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv
import random


class TRM(nn.Module):
    def __init__(self, cfg):
        super(TRM, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.TRM.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.TRM.TEXT_ENCODER.NAME
        self.use_score_map_loss = cfg.MODEL.TRM.LOSS.USE_SCORE_MAP_LOSS
        self.cfg = cfg.MODEL.TRM
        self.thresh = cfg.MODEL.TRM.LOSS.THRESH

        self.w = self.cfg.RESIDUAL

    def forward(self, batches, cur_epoch=1):
        """
        Arguments:
            batches.all_iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
        """
        # backbone
        ious2d = batches.all_iou2d
        assert len(ious2d) == batches.feats.size(0)
        for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
            assert iou.size(0) == sent.size(0)
            assert iou.size(0) == batches.num_sentence[idx]
        # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128
        feats = self.featpool(batches.feats)
        # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d = self.feat2d(feats)
        # two visual features using in different branches, both [B*C*T*T]
        map2d, map2d_iou = self.proposal_conv(map2d)
        # two features using in different branches, both list(B)-[num_sent*C]
        if self.text_encoder.use_phrase:
            sent_feat, sent_feat_iou, phrase_feat, phrase_feat_iou, phrase_weight, phrase_mask = self.text_encoder.encode_sentences(batches.sentences, batches.phrase)
        else:
            sent_feat, sent_feat_iou = self.text_encoder(batches.queries, batches.wordlens)
        # inference
        contrastive_scores = []
        iou_scores = []
        phrase_iou_scores = []
        phrase_score_map = []
        phrase_score_map_mask = []
        _, T, _ = map2d[0].size()
        for i, sf_iou in enumerate(sent_feat_iou):  # sent_feat_iou: [num_sent x C] (len=B)
            # iou part
            if self.cfg.RESIDUAL and self.text_encoder.use_phrase:
                vid_feat_iou = map2d_iou[i]  # C x T x T
                vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)

                phrase_persent_iou_scores = []
                pf_feat_iou = phrase_feat_iou[i]
                for pf_iou_pers in pf_feat_iou:
                    pf_iou_pers_norm = F.normalize(pf_iou_pers, dim=1) # max_p * C
                    iou_score = torch.mm(pf_iou_pers_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T) # max_p x T x T
                    phrase_persent_iou_scores.append(iou_score)

                phrase_persent_iou_scores = torch.stack(phrase_persent_iou_scores) # num_sent*max_p*T*T
                phrase_score_map.append((phrase_persent_iou_scores*10).sigmoid() * self.feat2d.mask2d)
                phrase_score_map_mask.append(phrase_mask[i])

                T_num = phrase_persent_iou_scores.size(2)
                phrase_persent_iou_scores = phrase_persent_iou_scores.reshape(phrase_persent_iou_scores.size(0), phrase_persent_iou_scores.size(1), -1)
                phrase_w = phrase_weight[i].unsqueeze(1) # num_sent*1*max_p
                # phrase_w.clamp(0, 0.0001) / phrase_w.clamp(0, 0.0001).sum(dim=-1, keepdim=True)
                phrase_att_iou_scores = phrase_w @ phrase_persent_iou_scores
                phrase_att_iou_scores = phrase_att_iou_scores.reshape(phrase_att_iou_scores.size(0), T_num, -1) #num*T*T

                sf_iou_norm = F.normalize(sf_iou, dim=1) # num_sent x C
                sf_iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T

                sf_iou_score = ((sf_iou_score + self.w * phrase_att_iou_scores) * 10).sigmoid() * self.feat2d.mask2d 
                # sf_iou_score = (((1 - self.w) * sf_iou_score + self.w * phrase_att_iou_scores) * 10).sigmoid() * self.feat2d.mask2d
                # print(phrase_weight[i].shape)
                # print(sf_iou_score.shape)
                # print(phrase_att_iou_scores.shape)
                # sf_iou_score = phrase_weight[i][:, 0].view(-1, 1, 1) * sf_iou_score + phrase_att_iou_scores
                # sf_iou_score = (sf_iou_score * 10).sigmoid() * self.feat2d.mask2d
                iou_scores.append(sf_iou_score)
                
                # pf_feat_iou = (pf_feat_iou * phrase_weight[i]).sum(dim=1)
                # sf_iou_norm = F.normalize(sf_iou + pf_feat_iou, dim=1) # num_sent x C
                # sf_iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T

                # iou_scores.append((sf_iou_score*10).sigmoid() * self.feat2d.mask2d)
            else:
                vid_feat_iou = map2d_iou[i]  # C x T x T
                vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
                sf_iou_norm = F.normalize(sf_iou, dim=1) # num_sent x C
                iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
                iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)
            
                if self.text_encoder.use_phrase:
                    phrase_persent_iou_scores = []
                    pf_feat_iou = phrase_feat_iou[i]
                    for pf_iou_pers in pf_feat_iou:
                        pf_iou_pers_norm = F.normalize(pf_iou_pers, dim=1) # max_p * C
                        iou_score = torch.mm(pf_iou_pers_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T) # max_p x T x T
                        phrase_persent_iou_scores.append(iou_score * self.feat2d.mask2d)

                    phrase_persent_iou_scores = torch.stack(phrase_persent_iou_scores) # num_sent*max_p*T*T
                    phrase_score_map.append((phrase_persent_iou_scores*10).sigmoid() * self.feat2d.mask2d)
                    phrase_score_map_mask.append(phrase_mask[i])
                    T_num = phrase_persent_iou_scores.size(2)
                    phrase_persent_iou_scores = phrase_persent_iou_scores.reshape(phrase_persent_iou_scores.size(0), phrase_persent_iou_scores.size(1), -1)
                    
                    phrase_w = phrase_weight[i].unsqueeze(1) # num_sent*1*max_p
                    phrase_att_iou_scores = phrase_w @ phrase_persent_iou_scores
                    phrase_att_iou_scores = phrase_att_iou_scores.reshape(phrase_att_iou_scores.size(0), T_num, -1) #num*T*T

                    phrase_iou_scores.append((phrase_att_iou_scores * 10).sigmoid() * self.feat2d.mask2d)
                    # pf_feat_iou = (pf_feat_iou * phrase_weight[i]).sum(dim=1)
                    # pf_feat_iou_norm = F.normalize(pf_feat_iou, dim=1) # num_sent x C
                    # iou_score = torch.mm(pf_feat_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T) # max_p x T x T
                    # phrase_iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)

        # phrase contrastive

        if self.text_encoder.use_phrase and self.cfg.LOSS.CONTRASTIVE:
            contrast_neg_vid = []
            contrast_neg_phr = []
            contrast_neg_mask = []
            BS = len(sent_feat_iou)
            for i in range(BS):  # sent_feat_iou: [num_sent x C] (len=B)
                neg_idx = random.randint(0, BS-1)
                while neg_idx == i:
                    neg_idx = random.randint(0, BS-1)

                # negative video
                vid_feat = map2d_iou[neg_idx]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                phrase_persent_scores = []
                pf_feat = phrase_feat_iou[i]
                for pf_pers in pf_feat:
                    pf_pers_norm = F.normalize(pf_pers, dim=1) # max_p * C
                    score = torch.mm(pf_pers_norm, vid_feat_norm.reshape(vid_feat_norm.size(0), -1)).reshape(-1, T, T) # max_p x T x T
                    phrase_persent_scores.append(score)
                phrase_persent_scores = torch.stack(phrase_persent_scores) # num_sent*max_p*T*T
                contrast_neg_vid.append((phrase_persent_scores*10).sigmoid() * self.feat2d.mask2d)

                vid_feat = map2d_iou[i]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                phrase_persent_scores = []
                pf_feat = phrase_feat_iou[neg_idx]
                for pf_pers in pf_feat:
                    pf_pers_norm = F.normalize(pf_pers, dim=1) # max_p * C
                    score = torch.mm(pf_pers_norm, vid_feat_norm.reshape(vid_feat_norm.size(0), -1)).reshape(-1, T, T) # max_p x T x T
                    phrase_persent_scores.append(score)
                phrase_persent_scores = torch.stack(phrase_persent_scores) # num_sent*max_p*T*T
                contrast_neg_phr.append((phrase_persent_scores*10).sigmoid() * self.feat2d.mask2d)
                contrast_neg_mask.append(phrase_mask[neg_idx])

        if self.text_encoder.use_phrase and self.use_score_map_loss:
            phrase_score_map = pad_sequence(phrase_score_map, batch_first=True) # B, NS, NP, T, T
            B, NS, NP, T1, T2 = phrase_score_map.shape
            phrase_score_map_mask = pad_sequence(phrase_score_map_mask, batch_first=True) # B, NS, NP
            phrase_score_map_mask_exp = phrase_score_map_mask.reshape(B, NS, NP, 1, 1).expand(B, NS, NP, T1, T2)
            phrase_iou2d = pad_sequence(ious2d, batch_first=True) # B, NS, T, T
            phrase_iou2d_exp = phrase_iou2d.unsqueeze(2).expand(B, NS, NP, T1, T2)

            scoremap_loss_pos, _ = phrase_score_map.masked_fill(phrase_iou2d_exp * self.feat2d.mask2d < self.thresh, -1e9).view(B, NS, NP, -1).max(dim=-1)
            # scoremap_loss_pos, idx = scoremap_loss_pos.masked_fill(phrase_score_map_mask==0, 1e9).min(dim=-1)
            if self.cfg.LOSS.USE_FOCAL_LOSS:
                scoremap_loss_pos = -scoremap_loss_pos.log() * (1 - scoremap_loss_pos).pow(2)
            else:
                scoremap_loss_pos = -scoremap_loss_pos.log()
            scoremap_loss_pos = torch.where(torch.isnan(scoremap_loss_pos), torch.full_like(scoremap_loss_pos, 0), scoremap_loss_pos)
            scoremap_loss_pos = torch.where(torch.isinf(scoremap_loss_pos), torch.full_like(scoremap_loss_pos, 0), scoremap_loss_pos)
            scoremap_loss_pos = torch.sum(scoremap_loss_pos * phrase_score_map_mask) / torch.sum(phrase_score_map_mask)
            # mask = phrase_score_map_mask.gather(index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
            # scoremap_loss_pos = torch.sum(scoremap_loss_pos * mask) / torch.sum(mask)

            if self.cfg.LOSS.CONTRASTIVE:
                contrast_neg_vid = pad_sequence(contrast_neg_vid, batch_first=True) # B, NS, NP, T, T
                B, NS, NP, T1, T2 = contrast_neg_vid.shape
                scoremap_loss_neg_vid, _ = contrast_neg_vid.masked_select(self.feat2d.mask2d==1).view(B, NS, NP, -1).max(dim=-1)
                # scoremap_loss_neg_vid, idx = scoremap_loss_neg_vid.masked_fill(phrase_score_map_mask==0, 1e9).min(dim=-1)
                if self.cfg.LOSS.USE_FOCAL_LOSS:
                    scoremap_loss_neg_vid = -(1 - scoremap_loss_neg_vid).log() * scoremap_loss_neg_vid.pow(2)
                else:
                    scoremap_loss_neg_vid = -(1 - scoremap_loss_neg_vid).log()
                scoremap_loss_neg_vid = torch.where(torch.isnan(scoremap_loss_neg_vid), torch.full_like(scoremap_loss_neg_vid, 0), scoremap_loss_neg_vid)
                scoremap_loss_neg_vid = torch.where(torch.isinf(scoremap_loss_neg_vid), torch.full_like(scoremap_loss_neg_vid, 0), scoremap_loss_neg_vid)
                scoremap_loss_neg_vid = torch.sum(scoremap_loss_neg_vid * phrase_score_map_mask) / torch.sum(phrase_score_map_mask)
                # mask = phrase_score_map_mask.gather(index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
                # scoremap_loss_neg_vid = torch.sum(scoremap_loss_neg_vid * mask) / torch.sum(mask)

                contrast_neg_phr = pad_sequence(contrast_neg_phr, batch_first=True) # B, NS, NP, T, T
                B, NS, NP, T1, T2 = contrast_neg_phr.shape
                contrast_neg_mask = pad_sequence(contrast_neg_mask, batch_first=True) # B, NS, NP
                scoremap_loss_neg_phr, _ = contrast_neg_phr.masked_select(self.feat2d.mask2d==1).view(B, NS, NP, -1).max(dim=-1)
                # scoremap_loss_neg_phr, idx = scoremap_loss_neg_phr.masked_fill(contrast_neg_mask==0, 1e9).min(dim=-1)
                if self.cfg.LOSS.USE_FOCAL_LOSS:
                    scoremap_loss_neg_phr = -(1 - scoremap_loss_neg_phr).log() * scoremap_loss_neg_phr.pow(2)
                else:
                    scoremap_loss_neg_phr = -(1 - scoremap_loss_neg_phr).log()
                scoremap_loss_neg_phr = torch.where(torch.isnan(scoremap_loss_neg_phr), torch.full_like(scoremap_loss_neg_phr, 0), scoremap_loss_neg_phr)
                scoremap_loss_neg_phr = torch.where(torch.isinf(scoremap_loss_neg_phr), torch.full_like(scoremap_loss_neg_phr, 0), scoremap_loss_neg_phr)
                scoremap_loss_neg_phr = torch.sum(scoremap_loss_neg_phr * contrast_neg_mask) / torch.sum(contrast_neg_mask)
                # mask = contrast_neg_mask.gather(index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
                # scoremap_loss_neg_phr = torch.sum(scoremap_loss_neg_phr * mask) / torch.sum(mask)

                scoremap_loss_neg = (scoremap_loss_neg_vid + scoremap_loss_neg_phr) / 2
            # else:
                scoremap_loss_exc, _ = phrase_score_map.masked_fill(phrase_score_map_mask_exp == 0, 1e9).min(dim=2)
                if self.cfg.LOSS.USE_FOCAL_LOSS:
                    scoremap_loss_exc = - (1 - scoremap_loss_exc).log() * scoremap_loss_exc.pow(2)
                else:
                    scoremap_loss_exc = - (1 - scoremap_loss_exc).log()
                scoremap_loss_exc = torch.where(torch.isnan(scoremap_loss_exc), torch.full_like(scoremap_loss_exc, 0), scoremap_loss_exc)
                scoremap_loss_exc = torch.where(torch.isinf(scoremap_loss_exc), torch.full_like(scoremap_loss_exc, 0), scoremap_loss_exc)
                scoremap_loss_exc = torch.sum(scoremap_loss_exc * (phrase_iou2d < self.thresh).long() * self.feat2d.mask2d) / torch.sum((phrase_iou2d < self.thresh).long() * self.feat2d.mask2d)
        else:
            scoremap_loss_pos = torch.tensor(0.0).cuda()
            scoremap_loss_neg = torch.tensor(0.0).cuda()
            scoremap_loss_exc = torch.tensor(0.0).cuda()
        # scoremap_loss_pos = torch.tensor(0.0).cuda()
        # scoremap_loss_neg = torch.tensor(0.0).cuda()
        # scoremap_loss_exc = torch.tensor(0.0).cuda()

        # loss
        # if self.training:
        # import pdb
        # pdb.set_trace()
        if self.text_encoder.use_phrase and not self.cfg.RESIDUAL:
            loss_iou_phrase = self.iou_score_loss(torch.cat(phrase_iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            if not self.cfg.LOSS.PHRASE_ONLY:
                loss_iou_stnc = self.iou_score_loss(torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            else:
                loss_iou_stnc = torch.tensor(0.0).cuda()
        else:
            loss_iou_stnc = self.iou_score_loss(torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            loss_iou_phrase = torch.tensor(0.0).cuda()
        
        loss_vid, loss_sent = self.contrastive_loss(map2d, sent_feat, ious2d, None, None, batches.moments)
        if self.training:
            return loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc
        else:
            loss = (loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc)
            for i, sf in enumerate(sent_feat):
                # contrastive part
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)

                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)
            
            if self.text_encoder.use_phrase and not self.cfg.RESIDUAL:
                # return map2d_iou, sent_feat_iou, contrastive_scores, phrase_iou_scores  # first two maps for visualization
                if self.cfg.LOSS.PHRASE_ONLY:
                    return map2d_iou, sent_feat_iou, contrastive_scores, phrase_iou_scores, loss  # first two maps for visualization
                else:
                    # return map2d_iou, sent_feat_iou, contrastive_scores, [(s + p) / 2 for s, p in zip(iou_scores, phrase_iou_scores)], loss # first two maps for visualization
                    return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores, loss # first two maps for visualization
            else:
                return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores, loss
