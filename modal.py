import time
import torch
import torch.nn as nn
from typing import Optional

class DNN_Net(nn.Module):
    def __init__(self, features=341, hiddens=128, classes=4):
        super(DNN_Net, self).__init__()
        self.classifier1 = nn.Sequential(
            nn.Linear(features, hiddens),
            nn.BatchNorm1d(hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.BatchNorm1d(hiddens),
            nn.ReLU(),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(hiddens, hiddens),
            nn.BatchNorm1d(hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.BatchNorm1d(hiddens),
            nn.ReLU(),
        )
        self.cls=nn.Linear(hiddens, classes)

    def forward(self, input):
        f1 = self.classifier1(input)
        f2 = self.classifier2(f1)
        return  self.cls(f2),[f1, f2]

class EEGNet(nn.Module):
    def __init__(self,
                 classes_num: int,
                 in_channels: int,
                 time_step: int,
                 kernLenght: int = 64,
                 F1: int = 8,
                 D: int = 2,
                 F2: int = 16,
                 dropout_size: Optional[float] = 0.5,
                ):
        super().__init__()
        self.n_classes = classes_num
        self.Chans = in_channels
        self.Samples = time_step
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropout_size
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=x.unsqueeze(1)    
        output = self.block1(x)
        temp = output.reshape(output.size(0), -1)
        output = self.block2(output)
        output1 = output.reshape(output.size(0), -1)
        output = self.classifier_block(output1)
        return output,[temp, output1]
    
    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)


class EEGNet_mini(nn.Module):
    def __init__(self,
                 classes_num: int,
                 in_channels: int,
                 time_step: int,
                 kernLenght: int = 1,
                 F1: int = 8,
                 D: int = 2,
                 F2: int = 16,
                 dropout_size: Optional[float] = 0.5,
                ):
        super().__init__()
        self.n_classes = classes_num
        self.Chans = in_channels
        self.Samples = time_step
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropout_size
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            # nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            # nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 1),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            # nn.AvgPool2d((1, self.Samples)),
            nn.Dropout(self.dropoutRate))
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1]==310:
            x=x.reshape(x.size(0),62,5)
        else:
            x=x.unsqueeze(1)
        x=x.unsqueeze(1)    
        output = self.block1(x)
        temp = output.reshape(output.size(0), -1)
        output = self.block2(output)
        output1 = output.reshape(output.size(0), -1)
        output = self.classifier_block(output1)
        return output,[temp, output1]
    
    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)

def adapt_state_dict(old_state_dict):
    new_state_dict = {}
    for key, value in old_state_dict.items():
        # print(key)
        if "classifier" in key:
            layer_num = int(key.split('.')[1])
            if layer_num < 5:
                new_key = key.replace("classifier", "classifier1")
            else:
                new_key = key.replace("classifier", "classifier2")
                new_layer_num = layer_num - 6
                new_key = new_key.replace(f".{layer_num}", f".{new_layer_num}")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model_1 = DNN_Net(features=310, hiddens=128, classes=3).cuda()
    model_2 = DNN_Net(features=33, hiddens=128, classes=3).cuda()
    # print(count_parameters(model))
    # model = EEGNet_mini(classes_num=5, in_channels=1, time_step=33).cuda()#3053
    # print(count_parameters(model))
    start_time = time.time()
    for _ in range(32):
        x_1 = torch.randn(32, 310).cuda()
        x_2 = torch.randn(32, 33).cuda()
        y ,temp= model_1(x_1)
        y ,temp= model_2(x_2)
        y=torch.softmax(y,dim=-1)+torch.softmax(y,dim=-1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"模型训练时间: {elapsed_time:.4f} 秒")

    model_1 = DNN_Net(features=310, hiddens=128, classes=5).cuda()
    model_2 = DNN_Net(features=33, hiddens=128, classes=5).cuda()
    # print(count_parameters(model))
    # model = EEGNet_mini(classes_num=5, in_channels=1, time_step=33).cuda()#3053
    # print(count_parameters(model))
    start_time = time.time()
    for _ in range(32):
        x_1 = torch.randn(32, 310).cuda()
        x_2 = torch.randn(32, 33).cuda()
        y ,temp= model_1(x_1)
        y ,temp= model_2(x_2)
        y=torch.softmax(y,dim=-1)+torch.softmax(y,dim=-1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"模型训练时间: {elapsed_time:.4f} 秒")

    model = DNN_Net(features=310, hiddens=128, classes=5).cuda()
    # model=EEGNet(classes_num=5, in_channels=62, time_step=400).cuda()#3093
    print(count_parameters(model))
    x = torch.randn(32, 310).cuda()
    start_time = time.time()
    for _ in range(32):
        y ,temp= model(x)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"模型训练时间: {elapsed_time:.4f} 秒")
    
    model = DNN_Net(features=33, hiddens=128, classes=5).cuda()
    print(count_parameters(model))
    # model = EEGNet_mini(classes_num=5, in_channels=1, time_step=33).cuda()#3053
    # print(count_parameters(model))
    start_time = time.time()
    x = torch.randn(32, 33).cuda()
    for _ in range(32):   
        y ,temp= model(x)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"模型训练时间: {elapsed_time:.4f} 秒")

    
    # print(y.shape)
    # print(temp[-1].shape)
    # checkpoint = torch.load("./ModelSave/seed/EYE2/1_eye.pth")
    # adapted_state_dict = adapt_state_dict(checkpoint)
    # model.load_state_dict(adapted_state_dict)
    # model.load_state_dict(torch.load("./ModelSave/seed/EEG1/1_eye.pth",map_location='cpu'))
    