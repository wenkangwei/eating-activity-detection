
import torch
from torch import nn
from torch.nn import functional as F


class Discriminator_BN_Bias(nn.Module):
    def __init__(self, ngpu, input_shape , out_fea = 1, bias=True):
        super(Discriminator_BN_Bias, self).__init__()
        
        # (input_shape[0], input_shape[1] )=  (number of data point, 6 axis channel )
        in_channels, win_size = input_shape[0], input_shape[1]
        self.in_channels = in_channels
        self.ngpu = ngpu  
        self.out_fea = out_fea
        filter_num = 10
        
        self.conv0 = nn.Conv1d(in_channels = self.in_channels, 
                               out_channels = filter_num, 
                               kernel_size= 44,  stride= 2, padding=0, bias=bias)
        
        self.relu0= nn.ReLU()
        num_fea = (win_size-44)//2 +1
        self.conv1 = nn.Conv1d(filter_num,filter_num, kernel_size= 20,stride= 2, padding=0, bias=bias)
        self.relu1= nn.ReLU()
        num_fea = (num_fea-20)//2 +1
        
        self.bn1 = nn.BatchNorm1d(filter_num)
        
        self.conv2 = nn.Conv1d(filter_num,filter_num, kernel_size= 4, stride= 2, padding=0, bias=bias)
        self.relu2= nn.ReLU()
        
        num_fea = (num_fea-4)//2 +1
        self.bn2 = nn.BatchNorm1d(filter_num)
        
        
        self.avgpool = nn.AvgPool1d(kernel_size=10)
        self.flatten = nn.Flatten()
        self.linear1 = None 
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=200, out_features=out_fea, bias=bias)
        self.softmax = nn.Softmax(dim=out_fea)
        self.sigmoid = nn.Sigmoid()
        
        nn.init.normal_(self.conv0.weight.data, 0.0, 1.)
        nn.init.normal_(self.conv1.weight.data, 0.0, 1.)
        nn.init.normal_(self.conv2.weight.data, 0.0, 1.)
        nn.init.normal_(self.linear2.bias.data, 0.0, 1.)
        #nn.init.normal_(self.conv0.bias.data, 0.0, 1.)
        #nn.init.normal_(self.conv1.bias.data, 0.0, 1.)
        #nn.init.normal_(self.conv2.bias.data, 0.0, 1.)
#         nn.init.normal_(self.avgpool.weight.data, 0.0, 1.)
        
    def l1_loss(self,factor=0.01):
        l1_crit = nn.L1Loss(size_average=False)
        reg_loss = 0.
        loss = 0.
        layers = [self.conv0, self.conv1, self.conv2]
        for layer in layers:
            for p in layer.parameters():
                #print(p)
                reg_loss += l1_crit(p, torch.zeros(p.shape))

        loss = factor * reg_loss
        return loss

    def forward(self, input):

        x = input.permute(0,2,1)
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x=  self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x=  self.bn2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        #print("Pooling shape:",x.shape)
        x = self.flatten(x)
        if self.linear1 == None:
            self.linear1 = nn.Linear(in_features=x.shape[1], out_features=200, bias=True)
            nn.init.normal_(self.linear1.weight.data, 0.0, 1.)
            nn.init.normal_(self.linear1.bias.data, 0.0, 1.)
            
        x = self.relu4(self.linear1(x))
        out = self.linear2(x)

        return out



class BasicBlock(nn.Module):
    def __init__(self, in_channels ,bias= True, filter_num = 10):
        super(BasicBlock, self).__init__()       
        self.conv0 = nn.Conv1d(in_channels = in_channels, 
                               out_channels = filter_num, 
                               kernel_size= 3,  stride= 1, padding=1, bias=bias)
        self.bn0 = nn.BatchNorm1d(filter_num)
        self.relu0 = nn.ReLU()
        
        self.conv1 = nn.Conv1d(in_channels = in_channels, 
                               out_channels = filter_num, 
                               kernel_size= 3,  stride= 1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm1d(filter_num)
        self.relu1 = nn.ReLU()
        pass
    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu0(out)
        out = self.conv1(out)
        out = self.bn1(out)
        
        out += residual
        out = self.relu1(out)
        
        return out

class Discriminator_ResNet(nn.Module):
    def __init__(self, ngpu, input_shape , out_fea = 1):
        super(Discriminator_ResNet, self).__init__()
        
        # (input_shape[0], input_shape[1] )=  (number of data point, 6 axis channel )
        in_channels, win_size = input_shape[0], input_shape[1]
        self.in_channels = in_channels
        self.ngpu = ngpu  
        self.out_fea = out_fea
        filter_num = 10
        
        self.conv0 = nn.Conv1d(in_channels = self.in_channels, 
                               out_channels = filter_num, 
                               kernel_size= 44,  stride= 2, padding=0, bias=True)
        
        self.relu0= nn.ReLU()
        num_fea = (win_size-44)//2 +1
        self.conv1 = nn.Conv1d(filter_num,filter_num, kernel_size= 20,stride= 2, padding=0, bias=True)
        self.relu1= nn.ReLU()
        self.bn1 = nn.BatchNorm1d(filter_num)
        
        self.block1 =  BasicBlock( in_channels=filter_num  ,bias= True, filter_num = filter_num)
        self.block2 =  BasicBlock( in_channels=filter_num  ,bias= True, filter_num = filter_num)
        
        
        self.avgpool = nn.AvgPool1d(kernel_size=10)
        self.flatten = nn.Flatten()
        self.linear1 = None 
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=200, out_features=out_fea, bias=True)
        self.softmax = nn.Softmax(dim=out_fea)
        self.sigmoid = nn.Sigmoid()
        
        nn.init.normal_(self.conv0.weight.data, 0.0, 1.)
        nn.init.normal_(self.conv1.weight.data, 0.0, 1.)
        nn.init.normal_(self.linear2.bias.data, 0.0, 1.)
        
    def l1_loss(self,factor=0.01):
        l1_crit = nn.L1Loss(size_average=False)
        reg_loss = 0.
        loss = 0.
        layers = [self.conv0, self.conv1]
        for layer in layers:
            for p in layer.parameters():
                #print(p)
                reg_loss += l1_crit(p, torch.zeros(p.shape))

        loss = factor * reg_loss
        return loss

    
    def forward(self, input):

        x = input.permute(0,2,1)
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x=  self.bn1(x)
        x = self.relu1(x)
        x = self.block1(x)
        x = self.block2(x)

        
        x = self.avgpool(x)
        #print("Pooling shape:",x.shape)
        x = self.flatten(x)
        if self.linear1 == None:
            self.linear1 = nn.Linear(in_features=x.shape[1], out_features=200, bias=True)
            nn.init.normal_(self.linear1.weight.data, 0.0, 1.)
            nn.init.normal_(self.linear1.bias.data, 0.0, 1.)
            
        x = self.relu2(self.linear1(x))
        out = self.linear2(x)

        return out
