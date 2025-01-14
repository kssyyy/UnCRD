import argparse
import torch
from torch.utils.data import Dataset
from torch import nn
import random
import numpy as np
import os
import math
import itertools
import sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import time

from modal import DNN_Net,adapt_state_dict,EEGNet_mini
from load_data import load_SEED_V_cross_subject_data,load_SEED_V_cross_session_data,load_SEED_V_within_session_data
from CRDLoss import CRDLoss
from unis_crdloss import Unis_CRDLoss

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from DGCNN import DGCNN
from BiHDM import BiHDM

ch_names = ['Fp1', 'FPZ', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
lh_chs = ['Fp1', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT7', 'FC5', 'FC3', 'FC1', 
          'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1', 
          'PO7', 'PO3', 'O1']
rh_chs = ['Fp2', 'AF4', 'F8', 'F6', 'F4', 'F2', 'FT8', 'FC6', 'FC4', 'FC2', 
          'T8', 'C6', 'C4', 'C2','TP8', 'CP6', 'CP4', 'CP2', 'P8', 'P6', 'P4', 'P2', 
          'PO8', 'PO4', 'O2']
lv_chs = ['Fp1', 'F7', 'FT7', 'T7', 'TP7', 'P7', 'PO7', 'AF3', 'F5', 'FC5', 
          'C5', 'CP5', 'P5', 'O1', 'F3', 'FC3', 'C3', 'CP3', 'P3', 'PO3', 'F1', 'FC1', 
          'C1', 'CP1', 'P1']
rv_chs = ['Fp2', 'F8', 'FT8', 'T8', 'TP8', 'P8', 'PO8', 'AF4', 'F6', 'FC6', 
          'C6', 'CP6', 'P6', 'O2', 'F4', 'FC4', 'C4', 'CP4', 'P4', 'PO4', 'F2', 'FC2', 
          'C2', 'CP2', 'P2']

lh_stream_ = [list(ch_names).index(ch) for ch in lh_chs]
rh_stream_ = [list(ch_names).index(ch) for ch in rh_chs]
lv_stream_ = [list(ch_names).index(ch) for ch in lv_chs]
rv_stream_ = [list(ch_names).index(ch) for ch in rv_chs]


class MyDataset_CRD(Dataset):
    def __init__(self, data,k=1000,mode='exact',percent=1):
        super().__init__()
        self.X = data[0]
        self.E = data[1]
        self.y = np.squeeze(data[2])
        
        num_samples=self.y.shape[0]
        self.k = k
        self.mode = mode
        num_classes=5
        
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[self.y[i]].append(i)
        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i] ) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive,dtype=object)
        self.cls_negative = np.asarray(self.cls_negative,dtype=object)
        
    def __getitem__(self, index):
        x = self.X[index].astype('float32')
        e = self.E[index].astype('float32')
        y = self.y[index].astype('int64')
        
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[y], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)
        replace = True if self.k > len(self.cls_negative[y]) else False
        if len(self.cls_negative[y]) > 0:
            neg_idx = np.random.choice(self.cls_negative[y], self.k, replace=replace)
        else:
            neg_idx=np.array([])
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return x, e,y, index, sample_idx
    
    def __len__(self):
        return self.y.shape[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',default=[2022,2023,2024], nargs='+', type=int,
                        help='set seeds for multiple runs!')
    
    parser.add_argument('--data_name', type=str,default='seed')
    parser.add_argument('--model_name', type=str, default='EYE1_within_test')#DGEYE1_4
    parser.add_argument('--session',type=int,default=4)

    parser.add_argument('--patience', type=int,default=15)
    parser.add_argument('--n_epochs', type=int,default=40)
    parser.add_argument('--batch_size', type=int,default=32)

    

    parser.add_argument('--model',type=str,default="MLP")#MLP,DGCNN,EEGNet,BiHDM
    parser.add_argument('--mcc',type=bool,default=False)
    parser.add_argument('--kd',type=bool,default=False)
    

    parser.add_argument('--best_acc_test', type=list, default=[0]*16)
    parser.add_argument('--best_bca_test', type=list, default=[0]*16)
    parser.add_argument('--device_ids',default=[7], nargs='+', type=int)

    parser.add_argument('--loss_type', type=int,default=0)
    parser.add_argument('--loss', type=str,default='unis_crd')#unis_crd crd kd hint nst similarity rkd pkt  
                                                        #attention kdsvd vid
    return parser.parse_args()

def seed_torch(seed=2024):
    """
    @description  : fix the random seed
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Set Random Seed: {}".format(seed))
    
def load_data_SEED(args):
    train=[]
    eval=[]
    test=[]
    for subject in range(1,16+1):
        if "within" in args.model_name:
            train_data, train_label, test_data, test_label = load_SEED_V_within_session_data(subject, session=args.session)
        else:
            train_data, train_label, test_data, test_label = load_SEED_V_cross_subject_data(subject, session=args.session)
        zscore = preprocessing.StandardScaler()  # data preprocess: zscore, 源域、目标域分别zscore()
        # train_data=train_data[:,310:341]
        # test_data=test_data[:,310:341]

        # train_data, eval_data, train_label, eval_label = train_test_split(train_data, train_label,test_size=0.2, random_state=2024)

        train_data_eeg=train_data[:,:310]
        # eval_data_eeg=eval_data[:,:310]
        test_data_eeg=test_data[:,:310]

        train_data_eye=train_data[:,310:]
        # eval_data_eye=eval_data[:,310:]
        test_data_eye=test_data[:,310:]
        
        # print(train_data_eeg.shape,train_data_eye.shape)
        # print(test_data_eeg.shape,test_data_eye.shape)

        train_data_eeg = zscore.fit_transform(train_data_eeg)
        test_data_eeg = zscore.fit_transform(test_data_eeg)
        train_data_eye = zscore.fit_transform(train_data_eye)
        test_data_eye = zscore.fit_transform(test_data_eye)

        

        if args.session == 4:
            train_set = MyDataset_CRD([train_data_eeg, train_data_eye, train_label],k=1000)
            # eval_set = MyDataset_CRD([eval_data_eeg, eval_data_eye, eval_label],k=1000)
            test_set = MyDataset_CRD([test_data_eeg,test_data_eye, test_label],k=1000)
        else:
            train_set = MyDataset_CRD([train_data_eeg, train_data_eye, train_label])
            # eval_set = MyDataset_CRD([eval_data_eeg, eval_data_eye, eval_label])
            test_set = MyDataset_CRD([test_data_eeg,test_data_eye, test_label])
        print(len(train_set),len(test_set))
        train_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=4)
        
        eval_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=0)

        test_loader = torch.utils.data.DataLoader(test_set,
                                batch_size=args.batch_size,
                                num_workers=0)
        train.append(train_loader)
        eval.append(eval_loader)
        test.append(test_loader)
    return train,test,eval


def test(model, data_loader, args):
    model.eval()
    predict = []
    groundtruth = []
    loss_all = 0
    loss_fuc = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for (batch_id, data) in enumerate(data_loader):
            x_eeg = data[0].cuda()
            x_eye = data[1].cuda()
            label = data[2].cuda()

            predicts,_=model(x_eeg)  
            
            loss = loss_fuc(predicts, label)

            loss_all += loss.data.cpu().numpy()

            result = torch.max(predicts, 1)[1].data.cpu().numpy()
            result = np.reshape(result, newshape=-1)
            y_data = np.reshape(label.data.cpu().numpy(), newshape=-1)

            predict.append(result)
            groundtruth.append(y_data)

    predict = np.concatenate(predict,axis=0)
    groundtruth = np.concatenate(groundtruth,axis=0)
    bca = balanced_accuracy_score(groundtruth, predict)
    acc = accuracy_score(groundtruth, predict)
    return  acc,bca

def feature_extract(model, data_loader,subject,type,args):
    predict = []
    groundtruth = []
    feature=[]
    loss_all = 0
    loss_fuc = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for (batch_id, data) in enumerate(data_loader):
            x_eeg = data[0].cuda()
            x_eye = data[1].cuda()
            label = data[2].cuda()

            
            predicts,temp=model(x_eeg) 

            # loss_all += loss.data.cpu().numpy()

            result = torch.max(predicts, 1)[1].data.cpu().numpy()
            result = np.reshape(result, newshape=-1)
            y_data = np.reshape(label.data.cpu().numpy(), newshape=-1)
            # feature.append(temp[-1].data.cpu().numpy())
            feature.append(predicts.data.cpu().numpy())
            predict.append(result)
            groundtruth.append(y_data)

    predict = np.concatenate(predict,axis=0)
    groundtruth = np.concatenate(groundtruth,axis=0)

    feature = np.concatenate(feature, axis=0)
    print(feature.shape)
    print(groundtruth.shape)
    load='./feature3/'
    if not os.path.exists(load+'{}'.format(args.model_name,type)):
        os.makedirs(load+'{}'.format(args.model_name,type))
    np.save(load+'{}/{}_feature{}.npy'.format(args.model_name,type,subject), feature)
    np.save(load+'{}/{}_label{}.npy'.format(args.model_name,type,subject), groundtruth)
    print(load+'{}/{}_feature{}.npy'.format(args.model_name,type,subject))
    return 


def train(train_loader,test_loader,eval_loader,subject,args):
    if args.model=='DGCNN':
        model_s=DGCNN(in_channels=5,num_electrodes=62,k_adj=2,out_channels=16,num_classes=5).cuda()
        model_t=DGCNN(in_channels=1,num_electrodes=33,k_adj=2,out_channels=16,num_classes=5).cuda()
    elif args.model=='BiHDM':
        model_s = BiHDM(lh_stream_,rh_stream_, lv_stream_, rv_stream_, n_classes=5,d_input=5, 
                    d_stream=32, d_pair=32, 
                    d_global=32, d_out=16, k=6, a=0.01, 
                    pairwise_operation='subtraction', output_domain=False,
                    rnn_stream_kwargs={}, 
                    rnn_global_kwargs={}).cuda()
        model_t=BiHDM([0],[0], [0], [0], n_classes=3,d_input=33, 
                    d_stream=16, d_pair=16, 
                    d_global=16, d_out=8, k=6, a=0.01, 
                    pairwise_operation='subtraction', output_domain=False,
                    rnn_stream_kwargs={}, 
                    rnn_global_kwargs={}).cuda()
    elif args.model=="EEGNet":
        model_t=EEGNet_mini(classes_num=5, in_channels=1, time_step=33).cuda()
        model_s=EEGNet_mini(classes_num=5, in_channels=62, time_step=5).cuda()

    else:
        model_t=DNN_Net(features=33, hiddens=128, classes=5).cuda()
        model_s=DNN_Net(features=310, hiddens=128, classes=5).cuda()

    model_t.eval()
    model_s.eval()
    _,feat_t = model_t(torch.randn(32, 33).cuda())
    _,feat_s= model_s(torch.randn(32, 310).cuda())
    
    # model_t=model_t
    # checkpoint = torch.load("./ModelSave/{}/{}_eye.pth".format(args.model_name,subject),map_location='cpu')
    # adapted_state_dict = adapt_state_dict(checkpoint)
    # model_t.load_state_dict(adapted_state_dict)
    
    model_t.load_state_dict(torch.load("./ModelSave/{}/{}_eye.pth".format(args.model_name,subject),map_location='cpu'))

    model_s=model_s
    # model_s=MulT(20,1500).cuda()
    
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    # print(len(train_loader))
    criterion_ce = nn.CrossEntropyLoss()
    # criterion_div = DistillKL(5)
    # criterion_kd = Unis_CRDLoss(n_data=12000,nce_k=9000)
    criterion_dis=DistillKL(4)

    if "within" in arg.model_name:
        n_data=1300
        nce_k=900
    else:
        n_data=28000
        nce_k=24000
    n_data=1200
    nce_k=1000
    if args.loss=='crd':
        if args.model=="DGCNN":
            criterion_kd = CRDLoss(s_dim=992,t_dim=528,feat_dim=648,n_data=n_data,nce_k=nce_k)
        elif args.model=="EEGNet":
            criterion_kd = CRDLoss(s_dim=80,t_dim=528,feat_dim=128,n_data=n_data,nce_k=nce_k)
        elif args.model=="BiHDM":
            criterion_kd = CRDLoss(s_dim=96,t_dim=48,n_data=n_data,nce_k=nce_k)
        else:
            criterion_kd = CRDLoss(n_data=n_data,nce_k=nce_k)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        # module_list.append(criterion_kd.predictor)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif args.loss=='kd':
        criterion_kd = DistillKL(4)
    elif args.loss == 'hint':
        if args.model=="DGCNN":
            criterion_kd = HintLoss(s_dim=992,t_dim=528,feat_dim=648)
        elif args.model=="BiHDM":
            criterion_kd = HintLoss(s_dim=96)
        else:
            criterion_kd = HintLoss()
        # regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        # module_list.append(regress_s)
        # trainable_list.append(regress_s)
    elif args.loss == 'attention':
        criterion_kd = Attention()#bad
    elif args.loss == 'nst':
        criterion_kd = NSTLoss()
    elif args.loss == 'similarity':
        criterion_kd = Similarity()
    elif args.loss == 'rkd':
        criterion_kd = RKDLoss()
    elif args.loss == 'pkt':
        criterion_kd = PKT()
    elif args.loss == 'kdsvd':#bad
        criterion_kd = KDSVD()
    elif args.loss == 'correlation':
        criterion_kd = Correlation()
    elif args.loss == 'vid':#bad
        s_n = [f.shape[1] for f in feat_s]
        t_n = [f.shape[1] for f in feat_t]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)


    else:
        if args.model=="DGCNN":
            criterion_kd = Unis_CRDLoss(s_dim=992,t_dim=528,feat_dim=648,n_data=n_data,nce_k=nce_k)
        elif args.model=="EEGNet":
            criterion_kd = Unis_CRDLoss(s_dim=80,t_dim=528,feat_dim=128,n_data=n_data,nce_k=nce_k)
        elif args.model=="BiHDM":
            criterion_kd = Unis_CRDLoss(s_dim=96,t_dim=48,n_data=n_data,nce_k=nce_k)
        else:
            criterion_kd = Unis_CRDLoss(n_data=n_data,nce_k=nce_k)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        # module_list.append(criterion_kd.predictor)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    # trainable_list.append(criterion_kd.predictor)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_ce)    # classification loss
    # criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd) 

    
    
    lr=1e-3
    optimizer = torch.optim.AdamW(trainable_list.parameters(), lr=lr, weight_decay=0.001)
    
    module_list.append(model_t)
    best_bca = 0
    best_acc=0
    last_best_epoch = 5
        
    module_list.cuda()
    criterion_list.cuda()

    # feature_extract(model_s,test_loader,subject,'MLP',args)

    for epoch in range(args.n_epochs):
        for module in module_list:
            module.train()
        module_list[-1].eval()
        
        loss_kd_epoch = torch.zeros((1, )).cuda()
        loss_ce_epoch = torch.zeros((1, )).cuda()
        loss_mcc_epoch = torch.zeros((1, )).cuda()

        train_pred = []
        train_true = []

        test_iter = iter(test_loader)
        cycle_test_iter = itertools.cycle(test_iter)

        iters_per_epoch = len(train_loader)
        elapsed_time=0
        start_time = time.time()
        for batch_id, data in enumerate(train_loader):
            unsup_batch=[t.cuda() for t in next(cycle_test_iter)]
            adjust_learning_rate(optimizer, epoch + batch_id/iters_per_epoch, lr)
            x_eeg = data[0].cuda()
            x_face=data[1].cuda()
            label = data[2].cuda()
            index = data[3].cuda()
            contrast_index=data[4].cuda()

            # 
            output_s,f_s=model_s(x_eeg)
            # end_time = time.time()
            # elapsed_time += (end_time - start_time)
            with torch.no_grad():
                output_t,f_t=model_t(x_face)
                
            loss_ce= criterion_ce(output_s, label)
            
            # loss_div = criterion_div(output_s, output_t)
            if args.mcc==True:
                def Entropy(input_):
                    bs = input_.size(0)
                    epsilon = 1e-5
                    entropy = -input_ * torch.log(input_ + epsilon)
                    entropy = torch.sum(entropy, dim=1)
                    return entropy

                unsup_x_eeg =unsup_batch[0].cuda()
                unsup_output_s,_=model_s(unsup_x_eeg) 
                output_s_temp = unsup_output_s/2.5
                s_softmax_out_temp = nn.Softmax(dim=1)(output_s_temp)
                s_entropy_weight = Entropy(s_softmax_out_temp).detach()
                s_entropy_weight = 1 + torch.exp(-s_entropy_weight)
                s_entropy_weight = 64* s_entropy_weight / torch.sum(s_entropy_weight)
                cov_matrix_t = s_softmax_out_temp.mul(s_entropy_weight.view(-1,1)).transpose(1,0).mm(s_softmax_out_temp)
                cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
                loss_mcc = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / 5
            else:
                loss_mcc=0

            if args.loss=='crd':
                loss_kd = 0.02*criterion_kd(f_s[-1], f_t[-1], index,contrast_index)
            elif args.loss=='kd':
                loss_kd = 10*criterion_kd(output_s, output_t)
            elif args.loss=='hint':
                loss_kd = 1*criterion_kd(f_s[-1], f_t[-1])
            elif args.loss == 'attention':
                loss_group = criterion_kd(f_s, f_t)
                loss_kd = 1000*sum(loss_group)
            elif args.loss == 'nst':
                g_s = f_s[:]
                g_t = f_t[:]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = 10*sum(loss_group)
            elif args.loss == 'similarity':
                g_s = [f_s[-2]]
                g_t = [f_t[-2]]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = 100*sum(loss_group)
            elif args.loss == 'rkd':
                loss_kd = criterion_kd(f_s[-1], f_t[-1])
            elif args.loss == 'pkt':
                loss_kd = 1000*criterion_kd(f_s[-1], f_t[-1])
            elif args.loss == 'kdsvd':
                g_s = f_s[:]
                g_t = f_t[:]
                loss_group = criterion_kd(g_s,g_t)
                loss_kd = sum(loss_group)
            elif args.loss == 'correlation':
                loss_kd = 0.1*criterion_kd(f_s[-1], f_t[-1])
            elif args.loss == 'vid':
                g_s = f_s
                g_t = f_t
                loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
                loss_kd = sum(loss_group)
            
            else:
                loss_kd = 0.02*criterion_kd(args,f_s[-1], f_t[-1],output_s,output_t, index,contrast_index,label)
            loss_mcc = 100*loss_mcc
            loss=loss_ce  +loss_kd+loss_mcc

            if args.kd==True and args.loss!='kd':
                loss=loss+criterion_dis(output_s, output_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_ce_epoch+=loss_ce
            loss_kd_epoch+=loss_kd
            loss_mcc_epoch+=loss_mcc

            result = torch.max(output_s, 1)[1].data.cpu().numpy()
            result = np.reshape(result, newshape=-1)
            y_data = np.reshape(label.data.cpu().numpy(), newshape=-1)
            
            train_pred.append(result)
            train_true.append(y_data)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        print(f"模型训练时间: {elapsed_time:.4f} 秒")
        train_pred = np.concatenate(train_pred,axis=0)
        train_true = np.concatenate(train_true,axis=0)       
        train_bca = balanced_accuracy_score(train_true, train_pred)
        train_acc = accuracy_score(train_true, train_pred)

        
        model_s.eval()
        tacc,tbca = test(model_s,test_loader,args)
        eacc,ebca = test(model_s,eval_loader,args)
        model_s.train()
        
        loss_ce_epoch = loss_ce_epoch / (batch_id + 1)
        loss_kd_epoch = loss_kd_epoch / (batch_id + 1)
        loss_mcc_epoch = loss_mcc_epoch / (batch_id + 1)
        print("epoch:{} loss:[{:.5f} {:.5f} {:.5f}]  test:[{:.5f} {:.5f}] train:[{:.5f} {:.5f}]"
            .format(epoch, loss_ce_epoch.item(),loss_kd_epoch.item(),loss_mcc_epoch.item(),tacc,tbca,train_acc,train_bca))
        
        if ebca > best_bca:
            best_bca = ebca
            if ebca>args.best_bca_test[subject-1]:
                args.best_bca_test[subject-1]=ebca
                # feature_extract(model_s,test_loader,subject,'CRD',args)

        if eacc > best_acc:
            best_acc = eacc
            if eacc>args.best_acc_test[subject-1]:
                args.best_acc_test[subject-1]=eacc
                # feature_extract(model_s,test_loader,subject,'CRD',args)




    return best_acc,best_bca
    # return temp1,temp2

def adjust_learning_rate(optimizer, epoch,lr):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < 10:
        lr_temp = lr * epoch / 10 
    else:
        lr_temp = lr * 0.5 * (1. + math.cos(math.pi * (epoch - 10) / (40 - 10)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_temp

if __name__ == '__main__':
    os.chdir(sys.path[0])
    args = parse_args()
    print(args)
    device = torch.device('cuda:{}'.format(args.device_ids[0]))
    torch.cuda.set_device('cuda:{}'.format(args.device_ids[0]))
    args.device = args.device_ids[0]

    if args.data_name=='seed':
        train_data,test_data,eval_data=load_data_SEED(args)
    num =len(train_data)
    final_acc = []
    final_bca=[]
    for seed in args.seeds:
        seed_torch(seed)
        acc_list=[]
        bca_list=[]
        for i in range(num):
            acc,bca=train(train_data[i],test_data[i],eval_data[i],i+1,args)
            print('subject: {}, acc: {:.5f} '.format(i+1, acc))
            acc_list.append(acc)
            bca_list.append(bca)
        final_acc.append(acc_list)
        final_bca.append(bca_list)
    print("session={},mcc={},save_name={}".format(args.session,args.mcc,args.model_name))
    for acc_list in final_acc:
        for acc1 in acc_list:
            print(acc1)
        acc_mean = np.mean(acc_list)
        acc_list.append(acc_mean)
        print(acc_mean)
        print("============================================")
    for bca_list in final_bca:
        for bca1 in bca_list:
            print(bca1)
        bca_mean = np.mean(bca_list)
        bca_list.append(bca_mean)
        print(bca_mean)
        print("============================================")
    for bca1 in args.best_bca_test:
        print(bca1)
    bca_mean = np.mean(args.best_bca_test)
    args.best_bca_test.append(bca_mean)
    print(bca_mean)
    print("============================================")

    for acc1 in args.best_acc_test:
        print(acc1)
    acc_mean = np.mean(args.best_acc_test)
    args.best_acc_test.append(acc_mean)
    print(acc_mean)
    print("============================================")    
    print(args)
