from packaging import version
import torch
from torch import nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, mask = None):
        batchSize = feat_q.shape[0] # BS = 256
        dim = feat_q.shape[1] # 256 by mlp
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # (256,1,256) * (256, 256, 1)  >> (256, 1, 1 )
        l_pos = l_pos.view(batchSize, 1) # (BS, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch: # negative from other samples, default False
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size # 1

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim) # (1, 256, 256)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim) # (1, 256, 256)
        npatches = feat_q.size(1) #256
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) # 1, 256, 256

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :] # uint8, >> [1,256,256]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) # exp 나중에 해줌
        l_neg = l_neg_curbatch.view(-1, npatches) # C, BS >> 256,256
    
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T # 256, 256+1

        if mask is not None:
            mask = (mask>0)*9.0 +1
            logsoft_out = F.log_softmax(out) # 256,257
            weighted_loss = -logsoft_out[:,0] *mask
            return weighted_loss
        else:
            loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
            return loss
        # for i in range(out.size(0)):
        #     if weighted_loss[i] != loss[i]:
        #         print(weighted_loss[i], loss[i])

        
