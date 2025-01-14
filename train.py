import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import argparse
from sklearn import preprocessing
import numpy as np
import os
import itertools
import random

from sklearn.model_selection import train_test_split
from modal import DNN_Net,EEGNet_mini
from load_data import load_SEED_V_cross_subject_data,load_SEED_V_cross_session_data,load_SEED_V_within_session_data
from DGCNN import DGCNN
from sognn import SOGNN
from BiHDM import BiHDM

ch_names = ['Fp1', 'FPZ', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 
            'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 
            'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 
            'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 
            'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 
            'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
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

class MyDataset(Dataset):
    def __init__(self, data,aug="false"):
        super().__init__()
        self.X = data[0]
        self.Y = data[1]

    def __getitem__(self, idx):
        x = self.X[idx].astype('float32')
        y = self.Y[idx].astype('int64')

        return x,y

    def __len__(self):
        return self.Y.shape[0]

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
            feature.append(temp[-1].data.cpu().numpy())
            predict.append(result)
            groundtruth.append(y_data)

    predict = np.concatenate(predict,axis=0)
    groundtruth = np.concatenate(groundtruth,axis=0)

    feature = np.concatenate(feature, axis=0)
    print(feature.shape)
    print(groundtruth.shape)
    if not os.path.exists('./feature/{}'.format(args.model_name,type)):
        os.makedirs('./feature/{}'.format(args.model_name,type))
    np.save('./feature/{}/{}_feature{}.npy'.format(args.model_name,type,subject), feature)
    np.save('./feature/{}/{}_label{}.npy'.format(args.model_name,type,subject), groundtruth)
    print('./feature/{}/{}_feature{}.npy'.format(args.model_name,type,subject))
    return 


def main(arg,seed):
    # torch.random.manual_seed(args.seed)
    acc_list=[]
    bca_list=[]
    for subject in range(1, 16+1):
        if "within" in arg.save_name:
            train_data, train_label, test_data, test_label = load_SEED_V_within_session_data(subject, session=arg.session)
        else:
            train_data, train_label, test_data, test_label = load_SEED_V_cross_subject_data(subject, session=args.session)
        zscore = preprocessing.StandardScaler()  # data preprocess: zscore, 源域、目标域分别zscore()
        if "EEG" in args.save_name:
            train_data=train_data[:,:310]
            test_data=test_data[:,:310]
            if args.model=="DGCNN":
                model = DGCNN(in_channels=5,num_electrodes=62,k_adj=2,out_channels=16,num_classes=5).cuda()
            elif args.model=="SOGNN":
                model = SOGNN().cuda()
            elif args.model=="EEGNet":
                model = EEGNet_mini(classes_num=5, in_channels=62, time_step=5).cuda()
            elif args.model=="BiHDM":
                lh_stream_ = [list(ch_names).index(ch) for ch in lh_chs]
                rh_stream_ = [list(ch_names).index(ch) for ch in rh_chs]
                lv_stream_ = [list(ch_names).index(ch) for ch in lv_chs]
                rv_stream_ = [list(ch_names).index(ch) for ch in rv_chs]
                model = BiHDM(lh_stream_,rh_stream_, lv_stream_, rv_stream_, n_classes=5,d_input=5, 
                                d_stream=32, d_pair=32, 
                                d_global=32, d_out=16, k=6, a=0.01, 
                                pairwise_operation='subtraction', output_domain=False,
                                rnn_stream_kwargs={}, 
                                rnn_global_kwargs={}).cuda()
            else:
                model = DNN_Net(features=310, hiddens=128, classes=5).cuda()
            print("EEG")
        else:
            train_data=train_data[:,310:]
            test_data=test_data[:,310:]
            if args.model=="DGCNN":
                model = DGCNN(in_channels=1,num_electrodes=33,k_adj=2,out_channels=16,num_classes=5).cuda()
            elif args.model=="SOGNN":
                model = SOGNN().cuda()
            elif args.model=="EEGNet":
                model = EEGNet_mini(classes_num=5, in_channels=1, time_step=33).cuda()
            elif args.model=="BiHDM":
                model_2 = BiHDM([0],[0], [0], [0], n_classes=5,d_input=33, 
                    d_stream=16, d_pair=16, 
                    d_global=16, d_out=8, k=6, a=0.01, 
                    pairwise_operation='subtraction', output_domain=False,
                    rnn_stream_kwargs={}, 
                    rnn_global_kwargs={}).cuda()
            else:
                model = DNN_Net(features=33, hiddens=128, classes=5).cuda()
        # # train_data=train_data[:,:310]
        # # test_data=test_data[:,:310]
        # train_data=train_data[:,310:]
        # test_data=test_data[:,310:]
        train_data = zscore.fit_transform(train_data)
        test_data = zscore.fit_transform(test_data)

        # train_data, eval_data, train_label, eval_label = train_test_split(train_data, train_label,test_size=0.2, random_state=seed)

        # model = DNN_Net(features=31, hiddens=128, classes=4).cuda()
        acc,bca= train(model, subject, train_data, train_label, train_data, train_label,
                           test_data, test_label, args)
        acc_list.append(acc)
        bca_list.append(bca)
        print('subject: {}, acc: {:.4f}, bca: {:.4f} '.format(subject, acc,bca))

    print("session={}".format(args.session))
    # for acc1 in acc_list:
    #     print(acc1)
    # acc_mean = np.mean(acc_list)
    # acc_list.append(acc_mean)
    # print(acc_mean)
    return acc_list,bca_list


def train(model, subject, src_data, src_label, eval_data, eval_label, tar_data, tar_label, args):
    # source_loader = from_data_load_train_data(src_data, src_label, batch_size=args.batch_size, kwargs=kwargs)
    # source_loader_len = len(source_loader)
    # target_loader = from_data_load_train_data(tar_data, tar_label, batch_size=args.batch_size, kwargs=kwargs)
    # len_dataloader = source_loader_len
    temp1,temp2=0,0 
    train_set = MyDataset([src_data, src_label])
    eval_set = MyDataset([eval_data, eval_label])
    test_set = MyDataset([tar_data,tar_label])
    train_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=4)
        
    eval_loader = torch.utils.data.DataLoader(eval_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set,
                                batch_size=args.batch_size,
                                num_workers=0)
    
    # weights_init(model)
    # model = model.to(args.device)
    for p in model.parameters():
        p.requires_grad = True
    # mcc_loss = MinimumClassConfusionLoss()
    # modal.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    acc_eval_list = []
    acc_target_list = []
    loss_list = []
    best_acc = 0
    best_bca = 0
    last_best_epoch=0
    for epoch in range(args.n_epoch):
        # source_iter = iter(source_loader)
        # target_iter = iter(target_loader)
        # while i < len_dataloader:
        len_dataloader=len(train_loader)
        i=0

        test_iter = iter(test_loader)
        cycle_test_iter = itertools.cycle(test_iter)

        for batch_id, data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # source_data, source_label = source_iter.next()
            # try:
            #     target_data, target_label = target_iter.next()
            # except:
            #     target_iter = iter(target_loader)
            #     target_data, target_label = target_iter.next()

            # source_data = source_data.to(args.device)
            # source_label = source_label.to(args.device)
            # target_data = target_data.to(args.device)
            x_eeg = data[0].cuda()
            label = data[1].cuda()
            # print(x_eeg.shape)
            class_output,_ = model(x_eeg)
            # target_logits = model(target_data)
            # print(class_output.shape)
            # print(label.shape)
            loss = nn.CrossEntropyLoss()(class_output, label.squeeze(1)) 

            unsup_batch=[t.cuda() for t in next(cycle_test_iter)]
            if args.mcc==True:
                def Entropy(input_):
                    bs = input_.size(0)
                    epsilon = 1e-5
                    entropy = -input_ * torch.log(input_ + epsilon)
                    entropy = torch.sum(entropy, dim=1)
                    return entropy

                unsup_x_eeg =unsup_batch[0].cuda()
                unsup_output_s,_=model(unsup_x_eeg) 
                output_s_temp = unsup_output_s/2.5
                s_softmax_out_temp = nn.Softmax(dim=1)(output_s_temp)
                s_entropy_weight = Entropy(s_softmax_out_temp).detach()
                s_entropy_weight = 1 + torch.exp(-s_entropy_weight)
                s_entropy_weight = 32* s_entropy_weight / torch.sum(s_entropy_weight)
                cov_matrix_t = s_softmax_out_temp.mul(s_entropy_weight.view(-1,1)).transpose(1,0).mm(s_softmax_out_temp)
                cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
                loss_mcc = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / 5
            else:
                loss_mcc=0

            loss.backward()
            optimizer.step()
            loss_list.append(loss)
            i+=1
            # acc_eval = test(model, eval_loader, args)
            # acc_target = test(model, test_loader, args)
            # acc_eval_list.append(acc_eval)
            # acc_target_list.append(acc_target)
            # print("subject: {}, epoch: {}, iter: {}/{}, train loss: {:.4f}, acc_eval: {:.4f},  acc_target: {:.4f}".format(
                # subject, epoch + 1, i, len_dataloader, loss, acc_eval, acc_target))
            # print(log)
            # args.log_file.write(log + '\n')
            # args.log_file.flush()

        model.eval()
        acc_eval,bca_eval = test(model, eval_loader, args)
        acc_target,bca_target = test(model, test_loader, args)
        print("subject: {}, epoch: {}, train loss: {:.4f}, acc_eval: {:.4f},  acc_target: {:.4f}".format(
                subject, epoch + 1, loss, acc_eval, acc_target))
        
        if acc_eval > best_acc:
            
            best_acc = acc_eval
            if acc_eval>args.best_acc_test[subject-1]:
                args.best_acc_test[subject-1]=acc_eval
                if not os.path.exists(args.save_name):
                    os.makedirs(args.save_name)
                torch.save(model.state_dict(), args.save_name + '{}_eye.pth'.format(subject))
                
            last_best_epoch = epoch

        if bca_eval >best_bca:
            best_bca = bca_eval
            if bca_eval>args.best_bca_test[subject-1]:
                args.best_bca_test[subject-1]=bca_eval
                if not os.path.exists(args.save_name):
                    os.makedirs(args.save_name)
                torch.save(model.state_dict(), args.save_name + '{}_eye.pth'.format(subject))
            last_best_epoch = epoch




        if epoch - last_best_epoch >args.patient :
            print('early_stopping! curr epoch: {} last_best_epoch: {}'
                          .format(epoch, last_best_epoch))
            break
    # torch.save(model.state_dict(), '{}/{}_{}_subject_{}.pth'.format(args.model_dir, args.scenario, args.method, subject))
    # plt.plot(iteration, loss_list)
    # plt.plot(iteration, acc_eval_list)
    # plt.plot(iteration, acc_target_list)
    # plt.savefig('{}/{}_{}_subject_{}.png'.format(args.figure_dir, args.scenario, args.method, subject), format='png', dpi=300)
    # plt.show()
    return best_acc,best_bca
    # return args.best_acc[subject-1],args.best_bca[subject-1]


def test(model, data_loader, args):
    model.eval()
    predict = []
    groundtruth = []
    loss_all = 0
    loss_fuc = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for (batch_id, data) in enumerate(data_loader):
            x_eeg = data[0].cuda()
            label = data[1].cuda()

            predicts,_=model(x_eeg)  
            
            loss = loss_fuc(predicts, label.squeeze(1))

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
    model.train()
    return  acc,bca

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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-n_epoch', type=int, default=50)
    parser.add_argument('-n_iteration', type=int, default=500)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-l2_decay', type=float, default=5e-4)
    parser.add_argument('-gamma', type=float, default=0.95)  # 指数学习率衰减
    parser.add_argument('-seed', type=int, default=2024, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-cuda', default=True)
    parser.add_argument('-gpu_id', type=str, default='9')
    # parser.add_argument('-root', type=str, default='{}'.format(os.getcwd()))
    parser.add_argument('-save_name', type=str, default='./ModelSave/DGEEG2_within_test/')
    parser.add_argument('-session',type=int,default=4)

    parser.add_argument('-model',type=str,default="DGCNN")#MLP,DGCNN,SOGNN,EEGNet,BiHDM
    parser.add_argument('-mcc',type=bool,default=False)

    parser.add_argument('--best_acc_eval', type=list, default=[0]*16)
    parser.add_argument('--best_bca_eval', type=list, default=[0]*16)

    parser.add_argument('--best_acc_test', type=list, default=[0]*16)
    parser.add_argument('--best_bca_test', type=list, default=[0]*16)

    parser.add_argument('--patient', type=int, default=10)
    parser.add_argument('--seeds',default=[2020,2021,2022,2023,2024,2025], nargs='+', type=int,
                        help='set seeds for multiple runs!')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.device = torch.device('cuda:{}'.format(args.gpu_id))
    torch.cuda.set_device('cuda:{}'.format(args.gpu_id))
    
    # kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    print(args)
    final_acc = []
    final_bca=[]
    for seed in args.seeds:
        seed_torch(seed)
        acc,bca=main(args,seed)
        final_acc.append(acc)
        final_bca.append(bca)

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

    # for bca1 in args.best_bca_eval:
    #     print(bca1)
    # bca_mean = np.mean(args.best_bca_eval)
    # args.best_bca_eval.append(bca_mean)
    # print(bca_mean)
    # print("============================================")

    # for acc1 in args.best_acc_eval:
    #     print(acc1)
    # acc_mean = np.mean(args.best_acc_eval)
    # args.best_acc_eval.append(acc_mean)
    # print(acc_mean)
    # print("============================================")

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
    print("采用验证集早停")