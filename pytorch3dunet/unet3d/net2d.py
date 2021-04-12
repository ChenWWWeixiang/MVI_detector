from __future__ import print_function, division, absolute_import
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import types
import torch,random
import re
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast as autocast

#################################################################
# You can find the definitions of those models here:
# https://github.com/pytorch/vision/blob/master/torchvision/models
#
# To fit the API, we usually added/redefined some methods and
# renamed some attributs (see below for each models).
#
# However, you usually do not need to see the original model
# definition from torchvision. Just use `print(model)` to see
# the modules and see bellow the `model.features` and
# `model.classifier` definitions.
#################################################################

__all__ = [
    'alexnet',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'inceptionv3',
    'squeezenet1_0', 'squeezenet1_1',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19'
]

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'densenet121': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth',
    'densenet169': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth',
    'densenet201': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth',
    'densenet161': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth',
    'inceptionv3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    # 'vgg16_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth',
    # 'vgg19_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth'
}

input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in ['inceptionv3']:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name] = [0.5, 0.5, 0.5]
    stds[model_name] = [0.5, 0.5, 0.5]

pretrained_settings = {}

for model_name in __all__:
    pretrained_settings[model_name] = {
        'imagenet': {
            'url': model_urls[model_name],
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 2
        }
    }

# for model_name in ['vgg16', 'vgg19']:
#     pretrained_settings[model_name]['imagenet_caffe'] = {
#         'url': model_urls[model_name + '_caffe'],
#         'input_space': 'BGR',
#         'input_size': input_sizes[model_name],
#         'input_range': [0, 255],
#         'mean': [103.939, 116.779, 123.68],
#         'std': [1., 1., 1.],
#         'num_classes': 1000
#     }

def update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict

def load_pretrained(model, num_classes, settings):
    #assert num_classes == settings['num_classes'], \
    #    "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    state_dict = model_zoo.load_url(settings['url'])
    state_dict = update_state_dict(state_dict)
    model.load_state_dict(state_dict,strict=False)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model




        # return loss/transposed.size(0)
#################################################################
# AlexNet

def modify_alexnet(model):
    # Modify attributs
    model._features = model.features
    del model.features
    model.dropout0 = model.classifier[0]
    model.linear0 = model.classifier[1]
    model.relu0 = model.classifier[2]
    model.dropout1 = model.classifier[3]
    model.linear1 = model.classifier[4]
    model.relu1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier


    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout0(x)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x=F.dropout(x,0.5)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def alexnet(num_classes=1000, pretrained='imagenet'):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    model = models.alexnet(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['alexnet'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_alexnet(model)
    return model

###############################################################
# DenseNets

def modify_densenets(model,USE_SUB):
    # Modify attributs
    model.USE_SUB=USE_SUB
    num_of_features=2208
    model.classifier = torch.nn.Linear(num_of_features,2)
    model.lstm=torch.nn.LSTM(num_of_features,1024,num_layers=1,bidirectional=True,batch_first=True)
    #model.lstm2 = torch.nn.LSTM(hidden*2, 256, bidirectional=True,batch_first=True)
    del model.classifier

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        d = x.view(x.size(0), -1)#padding,-1
        #x,_ = torch.max(x,0,keepdim=True)
        #x = F.dropout(x,0.5)
        x1 = self.classifier(d)
        if self.USE_SUB:
            x2 = self.classifier2(torch.cat([x,x1],-1))
            return x1,x2,d
        return x1,d
        return x1,x2,d
    def forward(self, input):
        #torch.squeeze()
        #input=input.squeeze(0).permute(1,0,2,3)
        with autocast():
            x = self.features(input)
            x1,x2,d = self.logits(x)
        return x1,d,x2

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    #model.lstm_logits = types.MethodType(lstm_logits, model)
    return model

def densenet121(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet121(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet121'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model,num_classes)
    return model

def densenet169(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet169(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet169'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model,num_classes)
    return model

def densenet201(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet201(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet201'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model,num_classes)
    return model

def densenet161(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet161(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet161'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model,num_classes)
    return model

###############################################################
# InceptionV3
def inceptionv3(num_classes=1000, pretrained='imagenet'):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    """
    model = models.inception_v3(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['inceptionv3'][pretrained]
        model = load_pretrained(model, num_classes, settings)

    # Modify attributs
    model.last_linear = model.fc
    del model.fc

    def features(self, input):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(input) # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x) # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x) # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x) # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x) # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 35 x 35 x 192
        x = self.Mixed_5b(x) # 35 x 35 x 256
        x = self.Mixed_5c(x) # 35 x 35 x 288
        x = self.Mixed_5d(x) # 35 x 35 x 288
        x = self.Mixed_6a(x) # 17 x 17 x 768
        x = self.Mixed_6b(x) # 17 x 17 x 768
        x = self.Mixed_6c(x) # 17 x 17 x 768
        x = self.Mixed_6d(x) # 17 x 17 x 768
        x = self.Mixed_6e(x) # 17 x 17 x 768
        if self.training and self.aux_logits:
            self._out_aux = self.AuxLogits(x) # 17 x 17 x 768
        x = self.Mixed_7a(x) # 8 x 8 x 1280
        x = self.Mixed_7b(x) # 8 x 8 x 2048
        x = self.Mixed_7c(x) # 8 x 8 x 2048
        return x

    def logits(self, features):
        x = F.avg_pool2d(features, kernel_size=8) # 1 x 1 x 2048
        x = F.dropout(x, training=self.training) # 1 x 1 x 2048
        x = x.view(x.size(0), -1) # 2048
        x = self.last_linear(x) # 1000 (num_classes)
        if self.training and self.aux_logits:
            aux = self._out_aux
            self._out_aux = None
            return x, aux
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

###############################################################
# ResNets
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
 
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
 
        out = self.gamma*out + x
        return out

class TextCNN(nn.Module):
    def __init__(self, num_classes, out_channels=100,sigmoid=False):
        super().__init__()
        self.features=models.resnet101(pretrained=False)
        settings = pretrained_settings['resnet101']['imagenet']
        self.features = load_pretrained(self.features, num_classes, settings)
       #self.features = modify_resnets(self.features,num_classes)
        del self.features.fc
        
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone=torch.nn.Sequential(self.features.conv1,
                                       self.features.bn1,self.features.relu,self.features.maxpool,
                                       self.features.layer1,self.features.layer2,self.features.layer3,
                                       #Self_Attn(1024,None),
                                       self.features.layer4,
                                       #Self_Attn(2048,None),
                                       self.features.avgpool)
        Ks=[2,4,6,8]
        Knum=out_channels
        self.convs=nn.ModuleList([nn.Conv2d(2,Knum,(K,2048)) for K in Ks])
        self.sigmoid=sigmoid
        self.dropout=nn.Dropout() 
        self.fc = nn.Linear(len(Ks)*Knum,num_classes) ##全连接层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,input,mask):
        with autocast():
            input=torch.transpose(input,2,1)#b*6*l*y*x->b*l*6*y*x
            b,l=input.shape[0],input.shape[1]
            input=input.reshape(-1,input.shape[2],input.shape[3],input.shape[4])#->bl*6*y*x
            features=torch.add(self.backbone(input[:,:3,:,:]),self.backbone(input[:,3:,:,:]))#->bl*1*c
            features=features.view(b,l,-1).unsqueeze(1)#->b*1*l*c
            mm=torch.zeros_like(features)
            mask=mask.unsqueeze(1).squeeze(-1)
            mask=mm*mask.unsqueeze(-1).repeat(1,1,1,2048)#b*1*l*c
            features=torch.cat([features,mask],1)
            features = [F.relu(conv(features)).squeeze(3) for conv in self.convs] # len(Ks)*(N,Knum,W)
            features = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in features]  # len(Ks)*(N,Knum)
            x= torch.cat(features,1) #(N,Knum*len(Ks))
            x = self.dropout(x)
            logit = self.fc(x)
            if self.sigmoid:
                return logit.log_softmax(-1)
            else:
                return logit

class ResLSTM(nn.Module):
    def __init__(self, num_classes, out_channels=256,sigmoid=False):
        super(ResLSTM,self).__init__()
        
        self.features=models.resnet101(pretrained=False)
        self.features.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        settings = pretrained_settings['resnet101']['imagenet']
        self.features = load_pretrained(self.features, num_classes, settings)
       # self.features = modify_resnets(self.features,num_classes)
        del self.features.fc
        self.features=torch.nn.Sequential(self.features.conv1,
                                       self.features.bn1,self.features.relu,self.features.maxpool,
                                       self.features.layer1,self.features.layer2,self.features.layer3,
                                       #Self_Attn(1024,None),
                                       self.features.layer4,
                                      # Self_Attn(2048,None),
                                       self.features.avgpool)
        self.z_use=nn.LSTM(2048,out_channels,1,True,True,0.5,bidirectional=True)
        self.fc=torch.nn.Sequential(torch.nn.Linear(out_channels*2,64),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(64,num_classes))
        self.sigmoid=sigmoid
    def forward(self,input,mask):
        with autocast():
            input=torch.transpose(input,2,1)#b*3*l*y*x->b*l*3*y*x
            b,l=input.shape[0],input.shape[1]
            input=input.reshape(-1,input.shape[2],input.shape[3],input.shape[4])#->bl*3*y*x
            features=self.features(input)#->bl*1*c
            features.view(features.size(0), -1)#->bl*c
            features= features.view(b,l,-1)#->b*l*c
            features, (hn, cn)=self.z_use(features)#->b*l*c
            out = torch.cat([hn[-1,:,:],hn[-2,:,:]],dim=-1)
            y1=self.fc(out)#
        
            if self.sigmoid:
                return y1.log_softmax(-1)
            else:
                return y1

def modify_resnets_r(model,num_of_cls,USE_25D,USE_SUB,USE_MC):
    # Modify attributs
    model.USE_SUB=USE_SUB
    fl=512
    model.classifier = torch.nn.Linear(2048+fl,num_of_cls)
    model.classifier2 = torch.nn.Linear(fl+2048+num_of_cls,17)
    model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       #Self_Attn(model.conv1.out_channels,None),
                                       model.layer1,
                                       Self_Attn(256,None),
                                       model.layer2,
                                       Self_Attn(512,None),
                                       model.layer3,
                                       Self_Attn(1024,None),
                                       model.layer4,
                                       Self_Attn(2048,None),
                                       model.avgpool)
    #model.bn=
    model.clinic=torch.nn.Sequential(
        torch.nn.BatchNorm1d(25),
        torch.nn.Linear(25,fl),
        nn.ReLU(),
        nn.Dropout())
    del model.fc

    def features_func(self, input):
        x = self.features(input)
        return x
    
    def forward(self, input,r,test=False):
        with autocast():
            x = self.features_func(input)
            x = x.view(x.size(0), -1)
            r=r.squeeze(0)
            r=self.clinic(r)
            #x=r
            x=x.cat([x,r],-1)
            d = x.max(0).values
            d = d.unsqueeze(0)
            x1 = self.classifier(x)
            if self.USE_SUB:
                x2 = self.classifier2(torch.cat([x,x1],-1))
                return x1,d,x2
        return x1,d
        
    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model

def modify_resnets(model,num_of_cls,USE_25D,USE_SUB,USE_MC):
    # Modify attributs
    model.USE_SUB=USE_SUB
    model.classifier = torch.nn.Linear(2048,num_of_cls)
    model.classifier2 = torch.nn.Linear(2048+num_of_cls,17)
    model.conv1 = nn.Conv2d(3+USE_MC*12, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       Self_Attn(model.conv1.out_channels,None),
                                       model.layer1,
                                       Self_Attn(256,None),
                                       model.layer2,
                                       Self_Attn(512,None),
                                       model.layer3,
                                       Self_Attn(1024,None),
                                       model.layer4,
                                       Self_Attn(2048,None),
                                       model.avgpool)
    del model.fc

    def features_func(self, input):
        x = self.features(input)
        return x

    def forward(self, input,test=False):
        with autocast():
            x = self.features_func(input)
            d = x.view(x.size(0), -1)
            if test:
                x=x.max(0).values
                x = x.unsqueeze(0)
            #d=x
            #d = x.max(0).values
            #d = d.unsqueeze(0)
            x1 = self.classifier(d)
            if self.USE_SUB:
                x2 = self.classifier2(torch.cat([d,x1],-1))
                return x1,d,x2
        return x1,d

    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model

def modify_resnets20(model,num_of_cls,USE_25D,USE_SUB,USE_MC,input_channel):
    # Modify attributs
    model.USE_SUB=USE_SUB
    model.classifier = torch.nn.Linear(512*2,num_of_cls)
    model.conv1=nn.Conv2d(input_channel,64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       model.layer1,
                                       model.layer2,
                                       model.layer3,
                                       model.layer4,
                                       model.avgpool)
    del model.fc
   # model.dropout=nn.Dropout(0.9)
    def features_func(self, input):
        temp=[]
        for i in range(6):
            temp.append(self.features(input[:,i,:,:,:]))
        x=torch.stack(temp,1)
        x=torch.mean(x,1)
        return x

    def forward(self, input,cropped):
        with autocast():
            x = self.features_func(input)
            x2 = self.features_func(cropped)
            d = x.view(x.size(0), -1)
            d2 = x.view(x2.size(0), -1)
            d=torch.cat([d,d2],-1)
            x1 = self.classifier(d)
        return x1,d

    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model


class AgentCLS(nn.Module):
    def __init__(self,input_f,num_of_cls):
        super(AgentCLS,self).__init__()
        self.dropout=nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(input_f,num_of_cls).cuda()
        self.classifier2 = torch.nn.Linear(input_f+num_of_cls,16).cuda()
    def forward(self,d):
        d=self.dropout(d)
        x1 = self.classifier(d)
        x2 = self.classifier2(torch.cat([d,F.relu(x1)],-1))
        return x1.log_softmax(-1),d,x2.log_softmax(-1)
class ForestCLF(nn.Module):
    def __init__(self,num_trees,model,num_features=1024,num_of_cls=3):
        super(ForestCLF,self).__init__()
        self.cls_agents=[]
        self.num_trees=num_trees
        self.num_features=num_features
        self.num_of_cls=num_of_cls
        #self.model=model
        self.model = models.resnet152(pretrained=False)
        #if pretrained is not None:
        settings = pretrained_settings['resnet152']['imagenet']
        self.model = load_pretrained(self.model, 1000, settings)
        self.model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       #Self_Attn(model.conv1.out_channels,None),
                                       model.layer1,
                                       #Self_Attn(256,None),
                                       model.layer2,
                                       #Self_Attn(512,None),
                                       model.layer3,
                                       #Self_Attn(1024,None),
                                       model.layer4,
                                      # Self_Attn(2048,None),
                                       model.avgpool)
        del self.model.fc
        self.cc=[]
        self.idx=[]
        self.clc=[]
        f_lists=np.arange(0,2048)
        c_lists=FILTERLIST1
        for i in range(num_trees):
            self.cls_agents.append(AgentCLS(num_features,num_of_cls).cuda())
            self.idx.append(np.array(random.sample(f_lists.tolist(), self.num_features)))
            self.cc.append(np.array(random.sample(c_lists, self.num_of_cls)))
            self.clc.append(nn.Linear(self.num_features,2).cuda())
        self.lossfunction=nn.NLLLoss().cuda()
    def get_pre1(self,gt):
        output_numpy=np.zeros((self.result[0].shape[0],5))
        output_numpy2=np.zeros((self.result[0].shape[0],16))
        count=np.zeros((self.result[0].shape[0],5))
        count2=np.zeros((self.result[0].shape[0],16))
        for ar,ac,aw,ar2 in zip(self.result,self.cc,self.w,self.result2):
            output_numpy[:,ac]+=torch.exp(ar).detach().cpu().numpy()*torch.exp(aw[:,1:2]).detach().cpu().numpy()
            count[:,ac]+=torch.exp(aw[:,1:2]).detach().cpu().numpy()
            count2+=torch.exp(aw[:,1:2]).detach().cpu().numpy()
            output_numpy2+=torch.exp(ar2).detach().cpu().numpy()*torch.exp(aw[:,1:2]).detach().cpu().numpy()
        output_numpy=output_numpy/(count)
        output_numpy2=output_numpy2/(count2)
        #output_numpy2=torch.stack(self.result2).mean(0).detach().cpu().numpy()
        output_numpy[np.isnan(output_numpy)]=0
        output_numpy=output_numpy/output_numpy.sum(-1,keepdims=True)
        output_numpy2=output_numpy2/output_numpy2.sum(-1,keepdims=True)
        pre=np.argmax(output_numpy,-1)
        pre2=np.argmax(output_numpy2,-1)
        return pre,pre2
    def get_output(self,x):
        with autocast():
            d=self.model.features(x)
            d=d.view(d.shape[0],-1)
            valid=torch.where((~torch.isnan(d)).any(1))
            d=d[valid]
            #target=target[valid]
            #target2=target2[valid]
            #loss1_s=[]
            self.result=[]
            self.result2=[]
            self.w=[]
            loss2=[]
            loss1=[]
            for i in range(self.num_trees):
                w=self.clc[i](d[:,self.idx[i]]).log_softmax(-1)##TODO add weight!
                y1,df,y2=self.cls_agents[i](d[:,self.idx[i]])
                self.result.append(y1)
                self.result2.append(y2)
                self.w.append(w)
            output_numpy=np.zeros((self.result[0].shape[0],5))
            output_numpy2=np.zeros((self.result[0].shape[0],16))
            count=np.zeros((self.result[0].shape[0],5))
            count2=np.zeros((self.result[0].shape[0],16))
            for ar,ac,aw,ar2 in zip(self.result,self.cc,self.w,self.result2):
                output_numpy[:,ac]+=torch.exp(ar).detach().cpu().numpy()*torch.exp(aw[:,1:2]).detach().cpu().numpy()
                count[:,ac]+=torch.exp(aw[:,1:2]).detach().cpu().numpy()
                count2+=torch.exp(aw[:,1:2]).detach().cpu().numpy()
                output_numpy2+=torch.exp(ar2).detach().cpu().numpy()*torch.exp(aw[:,1:2]).detach().cpu().numpy()
            
            
            output_numpy=output_numpy/count
            output_numpy2=output_numpy2/count2
            output_numpy[np.isnan(output_numpy)]=0
        return output_numpy,output_numpy2

    def forward(self,x,target,target2):
        with autocast():
            d=self.model.features(x)
            d=d.view(d.shape[0],-1)
            valid=torch.where((~torch.isnan(d)).any(1))
            d=d[valid]
            target=target[valid]
            target2=target2[valid]
            #loss1_s=[]
            self.result=[]
            self.result2=[]
            loss2=[]
            loss1=[]
            loss3=[]
            self.w=[]
            for i in range(self.num_trees):
                w=self.clc[i](d[:,self.idx[i]]).log_softmax(-1)#
                self.w.append(w)
                y1,df,y2=self.cls_agents[i](d[:,self.idx[i]])
                self.result.append(y1)
                self.result2.append(y2)
                
                fake=torch.ones_like(target.squeeze(-1))*-1
                #count=torch.zeros_like(target.squeeze(-1))
                for j in range(target.shape[0]):
                    if target[j] in self.cc[i].tolist():
                        fake[j]=self.cc[i].tolist().index(target[j])
                        #count[j]+=1
                idx_used=torch.where(fake>=0)
                if len(idx_used)>0:
                    loss2.append(torch.exp(w[:,1:2])*self.lossfunction(y2[idx_used],target2.squeeze(-1)[idx_used]))
                    loss1.append(torch.exp(w[:,1:2])*self.lossfunction(y1[idx_used],fake[idx_used]))
                w_gt=(fake>=0).long()
                loss3.append(F.nll_loss(w,w_gt))
            loss1=torch.stack(loss1)
            loss2=torch.stack(loss2)
            loss3=torch.stack(loss3)
            if torch.isnan(loss1).any():
                valid=torch.where((~torch.isnan(loss1)))
                loss1=loss1[valid]
            loss1=torch.mean(loss1)
            loss2=torch.mean(loss2)
            loss3=torch.mean(loss3)
            loss=loss1*0.7+loss2*0.3+loss3*0.5
            if torch.isnan(loss):
                a=1
        return loss,loss1,loss2,loss3,df

def modify_resnets_plus2(model,num_of_cls,asinput=False,USE_25D=False):
    # Modify attributs
    if USE_25D: 
        model.classifier = torch.nn.Linear(2048 + 7, num_of_cls)
    else:
        model.fusing = torch.nn.Linear(4096, 1024)
        model.classifier = torch.nn.Linear(1024+6,num_of_cls)
        model.droplayer=torch.nn.Dropout(0.5)
    if not asinput:
        model.classifier_gender = torch.nn.Linear(2048, 2)
        model.classifier_age = torch.nn.Linear(1024, 5)
        model.regress_pos = torch.nn.Linear(1024, 1)


    model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool)
    del model.fc

    def features_func(self, input):
        x = self.features(input)
        return x

    def forward(self, input,ipos=None,igender=None,iage=None):
        x = self.features_func(input)
        x = x.view(x.size(0), -1)

        gender = self.classifier_gender(x).log_softmax(-1)
        x=torch.cat([gender[:,0:1].exp()*x,gender[:,1:2].exp()*x],-1)
        cc=self.droplayer(self.fusing(x).relu())
        age = self.classifier_age(cc).log_softmax(-1)
        pos = self.regress_pos(cc).sigmoid()
        f=torch.cat([age.exp(),pos, cc], -1)
        y = self.classifier(f).log_softmax(-1)
        return y, gender, age, pos, f
    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model

def modify_resnets_plus(model,num_of_cls,asinput=False,USE_25D=False):
    # Modify attributs
    if USE_25D:
        model.classifier = torch.nn.Linear(2048 + 7, num_of_cls)
    else:
        model.classifier = torch.nn.Linear(2048+8,num_of_cls)
    if not asinput:
        model.classifier_gender = torch.nn.Linear(2048, 2)
        model.classifier_age = torch.nn.Linear(2048, 5)
        model.regress_pos = torch.nn.Linear(2048, 1)

    model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool)
    del model.fc

    def features_func(self, input):
        x = self.features(input)
        return x

    def forward(self, input,ipos=None,igender=None,iage=None):
        x = self.features_func(input)
        x = x.view(x.size(0), -1)
        if USE_25D:
            x = x.max(0)
        if not asinput:
            pos=self.regress_pos(x).sigmoid()
            gender = self.classifier_gender(x)
            age = self.classifier_age(x)
        else:
            pos=ipos
            #gender=igender
            gender = torch.zeros(igender.shape[0], 2).cuda().scatter_(1, igender, 1)
            age = torch.zeros(iage.shape[0], 5).cuda().scatter_(1, iage//20, 1)
           # age=iage.float()
        if USE_25D:
            cc=torch.cat([gender.relu(),age.relu(),x],-1)
            y = self.classifier(cc).log_softmax(-1)
            return y, gender.log_softmax(-1), age.log_softmax(-1), cc, cc
        else:
            cc = torch.cat([gender.relu(), age.relu(),pos, x], -1)
            y = self.classifier(cc).log_softmax(-1)
            return y, gender.log_softmax(-1), age.log_softmax(-1), pos, cc
    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model

    def __init__(self,inner_fs,o_fs):
        super(Decoder,self).__init__()
        self.deconv1=Up(inner_fs[0]+inner_fs[1],o_fs[0])
        self.deconv2=Up(inner_fs[2]+o_fs[0],o_fs[1])
        self.deconv3 = Up(inner_fs[3]+o_fs[1], o_fs[2])
        self.deconv4 = Up(inner_fs[4]+o_fs[2], o_fs[3])

    def forward(self,Fs):
        x=self.deconv1(Fs[0],Fs[1])
        x = self.deconv2(x, Fs[2])
        x = self.deconv3(x, Fs[3])
        x = self.deconv4(x, Fs[4])
        return x

def resnet18(input_channel,num_classes=1000,pretrained='imagenet'):
    """Constructs a ResNet-18 model.
    """
    model = models.resnet18(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet18'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets20(model,num_classes,USE_25D=False,
        USE_SUB=False,USE_MC=False,input_channel=input_channel)
    return model

def resnet34(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-34 model.
    """
    model = models.resnet34(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet34'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model

def resnet50(num_classes=1000, pretrained='imagenet',USE_25D=False):
    """Constructs a ResNet-50 model.
    """
    model = models.resnet152(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets20(model,num_classes,USE_25D=False,
        USE_SUB=False,USE_MC=False)
    return model

def resnet101(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-101 model.
    """
    model = models.resnet101(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet101'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model,num_classes)
    return model

def resnet152(num_classes=1000, pretrained='imagenet',setting=None):
    """Constructs a ResNet-152 model.
    """
    model = models.resnet152(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets20(model,num_classes,USE_25D=False,
        USE_SUB=False,USE_MC=False)
    return model
def multiagentres152(num_classes=1000, pretrained='imagenet',setting=None):
    """Constructs a ResNet-152 model.
    """
    model = models.resnet152(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = ForestCLF(20,model,num_features=1024,num_of_cls=2)
    return model

def resUnet152(num_classes=1000, pretrained='imagenet',setting=None):
    model = models.resnet152(pretrained=False)
    settings = pretrained_settings['resnet152'][pretrained]
    model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model,num_classes)
    return model

def resnet152_plus(num_classes=1000, pretrained='imagenet',asinput=False,USE_25D=False):
    model = models.resnet152(pretrained=False)
    settings = pretrained_settings['resnet152'][pretrained]
    model = load_pretrained(model, num_classes, settings)
    model = modify_resnets_plus2(model,num_classes,asinput=asinput,USE_25D=USE_25D)
    return model
def resnet152_R(num_classes=1000, pretrained='imagenet',setting=None):
    model = models.resnet152(pretrained=False)
    settings = pretrained_settings['resnet152'][pretrained]
    model = load_pretrained(model, num_classes, settings)
    model = modify_resnets_r(model,num_classes,USE_25D=setting['general']['use25d'],USE_SUB=setting['model']['subcls'],USE_MC=setting['general']['mc'])
    return model

###############################################################
# SqueezeNets

def modify_squeezenets(model,numclas):
    # /!\ Beware squeezenets do not have any last_linear module

    # Modify attributs
    model.dropout = model.classifier[0]
    #model.last_conv = model.classifier[1]
    model.relu = model.classifier[2]
    model.avgpool = model.classifier[3]
    model.last_conv = torch.nn.Conv2d(512, numclas, kernel_size=1)
    del model.classifier


    def logits(self, features):
        x = self.dropout(features)
        x = self.last_conv(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x).squeeze(-1).squeeze(-1)
        x=F.log_softmax(x,1)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def squeezenet1_0(num_classes=1000, pretrained='imagenet'):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    model = models.squeezenet1_0(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['squeezenet1_0'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model

def squeezenet1_1(num_classes=1000, pretrained='imagenet'):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = models.squeezenet1_1(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['squeezenet1_1'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model,num_classes)
    return model

###############################################################
# VGGs

def modify_vggs(model,num_of_cls):
    # Modify attributs
    #model._features = model.features
    #del model.features
    #model.linear0 = model.classifier[0]
    #model.relu0 = model.classifier[1]
    #model.dropout0 = model.classifier[2]
    #model.linear1 = model.classifier[3]
    #model.relu1 = model.classifier[4]
    #model.dropout1 = model.classifier[5]
    #model.last_linear =
    model.classifier[6]=torch.nn.Linear(4096,num_of_cls)

    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), -1)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout0(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.dropout1(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x=x.view(x.size(0), -1)
        x = self.classifier(x)
        x=F.log_softmax(x,-1)
        return x

    # Modify methods
    #model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def vgg11(num_classes=1000, pretrained='imagenet'):
    """VGG 11-layer model (configuration "A")
    """
    model = models.vgg11(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg11'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg11_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    model = models.vgg11_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg11_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg13(num_classes=1000, pretrained='imagenet'):
    """VGG 13-layer model (configuration "B")
    """
    model = models.vgg13(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg13'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg13_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 13-layer model (configuration "B") with batch normalization
    """
    model = models.vgg13_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg13_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg16(num_classes=1000, pretrained='imagenet'):
    """VGG 16-layer model (configuration "D")
    """
    model = models.vgg16(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg16'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg16_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    model = models.vgg16_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg16_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg19(num_classes=1000, pretrained='imagenet'):
    """VGG 19-layer model (configuration "E")
    """
    model = models.vgg19(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg19'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model,num_classes)
    return model

def vgg19_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 19-layer model (configuration 'E') with batch normalization
    """
    model = models.vgg19_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg19_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model,num_classes)
    return model