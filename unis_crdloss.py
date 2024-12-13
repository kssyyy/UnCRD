import torch
from torch import nn
import math
import torch.nn.functional as F

eps = 1e-7

class Unis_CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self,s_dim=128,t_dim=128,feat_dim=128,n_data=1470,nce_k=140,nce_t=0.07,nce_m=0.05,teach_modal="MLP"):
        super(Unis_CRDLoss, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        # self.predictor = self._build_mlp(2, feat_dim, 128, feat_dim, False)
        # self.predictor1 = self._build_mlp(2, feat_dim, 128, feat_dim, False)
        
        self.contrast = ContrastMemory(feat_dim, n_data,nce_k,nce_t, nce_m)
        # self.criterion_t = ContrastLoss(n_data)
        # self.criterion_s = ContrastLoss(n_data)
        self.softmax=False
        self.criterion_t = SoftmaxLoss() if self.softmax else ContrastLoss(n_data)
        self.criterion_s = SoftmaxLoss() if self.softmax else ContrastLoss(n_data)
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)
    
    def forward(self,args, f_s, f_t,p_s,p_t, idx, contrast_idx=None,labels=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """



        f_s=self.embed_s(f_s)
        f_t=self.embed_t(f_t)
        # f_s = self.predictor(f_s)
        # f_t = self.predictor(f_t)
        # print(f_t.shape)
        #(bz,k+1,1)

        def Entropy(input_):
            bs = input_.size(0)
            epsilon = 1e-5
            entropy = -input_ * torch.log(input_ + epsilon)
            entropy = torch.sum(entropy, dim=1)
            return entropy
        s_softmax_out_temp = nn.Softmax(dim=1)(p_t)
        s_entropy_weight = Entropy(s_softmax_out_temp)
        # s_entropy_weight=nn.CrossEntropyLoss()(s_softmax_out_temp,labels)
        s_entropy_weight = 1 + torch.exp(-s_entropy_weight)
        s_entropy_weight = 32* s_entropy_weight / torch.sum(s_entropy_weight)
        con=s_entropy_weight.reshape(-1, 1, 1)
        


        p_s = torch.argmax(F.softmax(p_s, dim=1), dim=1)
        p_t = torch.argmax(F.softmax(p_t, dim=1), dim=1)
        
        temp_s,temp_t,mask = self.UniSMMConLoss(f_s, f_t, p_s, p_t, labels)
        mask = torch.where(mask == 0, torch.tensor(-1), mask)
        # con=mask* s_entropy_weight
        


        
        if args.loss_type==1:
            logits = torch.div(torch.matmul(temp_s, temp_t.T), 0.07)
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)

            # compute log_prob
            # exp_logits = torch.exp(logits-logits_max.detach())[0]
            exp_logits = torch.diag(torch.exp(logits-logits_max.detach()))
            loss = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum())
        else:
            out_s, out_t = self.contrast(temp_s, temp_t, idx,contrast_idx)
            out_s = out_s*con
            out_t = out_t*con
            s_loss = self.criterion_s(out_s,mask)
            t_loss = self.criterion_t(out_t,mask)
            loss = s_loss + t_loss
        return loss
    
    def UniSMMConLoss(self, feature_a, feature_b, predict_a, predict_b, labels):
        # 文本特征，图像特征，文本预测，图像预测，真实标签，温度
        feature_a_ = feature_a.detach()
        feature_b_ = feature_b.detach()

        a_pre = predict_a.eq(labels)  # a True or not
        a_pre_ = ~a_pre
        b_pre = predict_b.eq(labels)  # b True or not
        b_pre_ = ~b_pre

        a_b_pre = torch.gt(a_pre | b_pre, 0)  # For mask ((P: TT, nP: TF & FT)=T, (N: FF)=F)
        a_b_pre_ = torch.gt(a_pre & b_pre, 0) # For computing nP, ((P: TT)=T, (nP: TF & FT, N: FF)=F)
        #a_pre为a的正确预测 a_pre_为a的错误预测 a_b_pre为a和b的只要有一个正确预测 a_b_pre_为a和b全正确的预测
        a_ = a_pre_ | a_b_pre_  # For locating nP not gradient of a
        b_ = b_pre_ | a_b_pre_  # For locating nP not gradient of b

        if True not in a_b_pre:
            a_b_pre = ~a_b_pre
            a_ = ~a_
            b_ = ~b_
        mask = a_b_pre.float()
#
        feature_a_f = [feature_a[i].clone() for i in range(feature_a.shape[0])]
        for i in range(feature_a.shape[0]):
            if not a_[i]:# “a” 正确预测且 “a” 和 “b” 并非全正确预测的样本
                feature_a_f[i] = feature_a_[i].clone()
        feature_a_f = torch.stack(feature_a_f)

        feature_b_f = [feature_b[i].clone() for i in range(feature_b.shape[0])] # feature_b  # [[0,1]])
        for i in range(feature_b.shape[0]):
            if not b_[i]:# “b” 正确预测且 “a” 和 “b” 并非全正确预测的样本
                feature_b_f[i] = feature_b_[i].clone()
        feature_b_f = torch.stack(feature_b_f)
        return feature_a_f,feature_b_f,b_pre.float()


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
            
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        # print(v2.shape)
        # print(weight_v1.shape)
        # print(v2)
        # print(weight_v1)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
            
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))
        
        if math.isnan(Z_v1):
                Z_v1 = Z_v2
        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2      

class SoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss
       
class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x,mask):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        mask1=mask.reshape(-1, 1)
        # print(log_D1.sum(0))
        log_D1=mask1*log_D1
        # print(log_D1.sum(0))
        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        # print(log_D0.view(-1, 1).sum(0))
        mask2=mask.reshape(-1, 1, 1)
        log_D0=mask2*log_D0   
        # print(log_D0.view(-1, 1).sum(0))

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz
        # print(loss)
        return loss

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x
      
class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss
    
class ConvReg(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(448,256)  # 隐藏层

    def forward(self, X):
        return F.relu(self.hidden(X))
    
class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj
    
    