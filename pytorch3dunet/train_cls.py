import importlib
import torch
import os,yaml
import sys,logging
import warnings 
warnings.filterwarnings('ignore')
import numpy as np 
sys.path.append('/mnt/data9/deep_R/pytorch-3dunet/')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3dunet.datasets.mvi_dataset import MviSet2D as MviSet
from pytorch3dunet.datasets.mvi_dataset import SetWarpper
import torch.cuda.amp as amp
from pytorch3dunet.unet3d.vit2 import ViT3D
from pytorch3dunet.unet3d.transunet import VisionTransformer,CONFIGS,FocalLoss 
from pytorch3dunet.unet3d.net2d import ResLSTM,TextCNN,resnet50,resnet18
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES']='1'
config_file='pytorch3dunet/train_cls.yaml'
loggers = {} 

#mix + focal = 80 epoch 0.617(0.9356+0.1025)
#mix + nll = 100 pocjh 0.607(0.83+19.45)
def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger

def main():
    options=yaml.safe_load(open(config_file, 'r'))
    #create model
    # model = ViT3D(
    #     image_size = (32, 224, 224),
    #     patch_size = (8, 16, 16),
    #     num_classes = 2,
    #     dim = 1024,
    #     depth = 7,
    #     heads = 32,
    #     mlp_dim = 1024,
    #     dropout = 0.1,
    #     emb_dropout = 0.3
    # )
    
    #model=VisionTransformer(CONFIGS[options['model']['configs']],[224,224,32])
    if options['model']['use2d']:
        model=resnet18(input_channel=len(options['thsets']), num_classes=2)
    else:
        model=TextCNN(2)
    print(model)
    os.makedirs(options['trainer']['checkpoint_dir']+'/saves',exist_ok=True)
    os.makedirs(options['trainer']['checkpoint_dir']+'/logs',exist_ok=True)
    logger = get_logger('Train')
    if(options["model"]["loadpretrainedmodel"]):
        # remove paralle module
        if os.path.exists(options["model"]["pretrainedmodelpath"]):
            pretrained_dict = torch.load(options["model"]["pretrainedmodelpath"])
            
            #options['trainer']['start_epoch']=pretrained_dict['epoch']
            # load only exists weights
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
            #print('matched keys:',len(pretrained_dict))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('Load pretrained OK!')
            
    model=model.cuda()
    #create dataloader
    train_dataset=MviSet(options['loader']['train']['data_list'],True,options)
    test_dataset=MviSet(options['loader']['test']['data_list'],False,options)
    ttv=SetWarpper(options['loader']['test']['data_list'],False,options)
    train_dataset.load_things()
    test_dataset.load_things()
    print('Loading dataset OK!')
    train_loader=DataLoader(train_dataset,batch_size=options["loader"]['train']["batchsize"],
                                    shuffle=options["loader"]['train']["shuffle"],
                                    num_workers=options["loader"]['train']["numworkers"],
                                    drop_last=False,
                                    #sampler=sampler,
                                    pin_memory=True,)
    test_loader=DataLoader(test_dataset,batch_size=options["loader"]['test']["batchsize"],
                                    shuffle=options["loader"]['test']["shuffle"],
                                    num_workers=options["loader"]['test']["numworkers"],
                                    drop_last=False,
                                    #sampler=sampler,
                                    pin_memory=True,)
    v_loader=DataLoader(ttv,batch_size=1,
                                    shuffle=options["loader"]['test']["shuffle"],
                                    num_workers=options["loader"]['test']["numworkers"],
                                    drop_last=False,
                                    #sampler=sampler,
                                    pin_memory=True,)
    optimizer = optim.Adam(model.parameters(),lr = options["trainer"]['lr'],amsgrad=True,weight_decay=options["trainer"]['weight_decay'])
    TOTALITER=0
    schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',
                                                                 patience=10, factor=.1, threshold=1e-3, verbose=True)
    acct=0
    
    for epoch in range(options['trainer']['start_epoch'],options['trainer']['max_epoch']):
       # break
        train_gogogo(model,train_loader,optimizer,options,TOTALITER)
        acc,acc0,acc1=val_gogogo(model,test_loader,options,schedule)
        print('ACC for eopch: ',epoch,acc,acc0,acc1)
        saving(model,'latest',',mix2dfocal',options)
        if acc>acct:
            acct=acc
            saving(model,'best','mix2dfocal',options)
        if epoch%1==0:
            acc,acc0,acc1=test_gogogo(model,v_loader,options)
            print('Epoch Acc: ',epoch,acc,acc0,acc1)
    
    print('Final Acc: ',epoch,acc,acc0,acc1)
def mixup_data(x,y,c,alpha):
    lam = np.random.beta(alpha, 1-alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index] 
    mixed_c = lam * c + (1 - lam) * c[index,:]
    return mixed_x, y_a, y_b,mixed_c, lam
    
def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def saving(model,epoch,others,options):
    tt=model.state_dict()
    os.makedirs(options['trainer']['checkpoint_dir']+'/saves',exist_ok=True)
    #tt['epoch']=epoch
    torch.save(tt, "{}/{}_epoch{}.pt".format(options['trainer']['checkpoint_dir']+'/saves',others,epoch))
@torch.no_grad()
def test_gogogo(model,test_loader,options):
    model.eval()
    acc1=0
    acc0=0
    acc=0
    cnt=0
    cnt1=0
    cnt0=0
    for idx,(data,c,gt) in enumerate(test_loader):
        data,c,gt=data[0].cuda().float(),c[0].cuda().float(),gt[0].numpy()
        #gt=torch.Tensor(gt)
        #data,c=torch.Tensor(data).cuda().float(),torch.Tensor(c).cuda().float()
        pred,_=model(data,c)
        #pred=pred.log_softmax(-1)
        p=[]
        for i in range(2):
            t=pred[:,i].cpu().numpy().tolist()
            t.sort()
            p.append(np.mean(t[-5:]))
        pred=np.array(p)
        #print(pred,gt)
        pred=np.argmax(pred,-1)
        #acc=(pred==gt).mean()
        cnt+=gt.shape[0]
        acc+=np.sum(pred==gt)*1.0
        acc1+=np.sum((pred==gt)*(gt==1))*1.0
        cnt0+=np.sum(gt==0)
        acc0+=np.sum((pred==gt)*(gt==0))*1.0
        cnt1+=np.sum(gt==1)
        
        a=1
    return acc/cnt,acc0/cnt0,acc1/cnt1
@torch.no_grad()
def val_gogogo(model,loader,options,schedule):
    model.eval()
    acc1=0
    acc0=0
    acc=0
    cnt=0
    cnt1=0
    cnt0=0
    criti=FocalLoss()
    losses=[]
    for idx,(data,c,gt) in enumerate(loader):
        
        data,c,gt=data.cuda().float(),c.cuda().float(),gt[0].cuda()
        gtt=gt.cpu().numpy()
       
        pred,_=model(data,c)
        pred=pred.log_softmax(-1)
        loss=criti(pred,gt).mean().detach().cpu().numpy()
        losses.append(loss)
        #gt=gt.cpu().numpy()
        # p=[]
        # for i in range(mask.shape[0]):
        #     p.append(torch.argmax(pred[i,mask[i]].mean(0),-1).detach().cpu().numpy())
        # pred=np.array(p)
        pred=torch.argmax(pred,-1).detach().cpu().numpy()
        #print(pred,gt)
        acc+=np.sum(pred==gtt)*1.0
        cnt+=gt.shape[0]
        acc1+=np.sum((pred==gtt)*(gtt==1))*1.0
        cnt0+=np.sum(gtt==0)
        acc0+=np.sum((pred==gtt)*(gtt==0))*1.0
        cnt1+=np.sum(gtt==1)
        if idx>options['trainer']['validation_iters']:
            break
    losses=np.array(losses)
    schedule.step(losses.mean())
    return acc/cnt,acc0/cnt0,acc1/cnt1
        
        
def train_gogogo(model,loader,optimizer,options,TOTALITER):
    #criti=nn.NLLLoss(torch.Tensor([0.3,0.7]).cuda().half())
    criti=FocalLoss()
    model.train()
    scaler = amp.GradScaler()
    optimizer.zero_grad()
    mm=0
    for idx,(data,c,gt) in enumerate(loader):
        data,c,gt=data.cuda().float(),c.cuda().float(),gt[0].cuda()
        pred,d=model(data,c)
        pred=pred.log_softmax(-1)
        loss=criti(pred,gt)
        if options['trainer']['mixup']:
            inputs, targets_a, targets_b,mixed_c, lam = mixup_data(data, gt,c, 0.5)
            #映射为Variable
            inputs, targets_a, targets_b,mixed_c = map(Variable, (inputs,targets_a,targets_b,mixed_c))
            #抽取特征，BACKBONE为粗特征抽取网络
            outputs=model(inputs,mixed_c)[0].log_softmax(-1)
            loss_func = mixup_criterion(targets_a, targets_b, lam)  
            loss += loss_func(criti, outputs)
        #loss.backward()
        mm+=loss.item()
        scaler.scale(loss).backward()
        #mask=mask.squeeze(-1).bool()
        if idx%1==0:                            
            scaler.step(optimizer)
            
            # 更新scalar的缩放信息
            scaler.update()

            #optimizer.step()

            optimizer.zero_grad()
            TOTALITER+=1
        if TOTALITER%options['trainer']['log_after_iters']==0:
           # x=mm/options['trainer']['log_after_iters']
            print(f'iters '+str(TOTALITER)+f':loss {loss.item()}')
           # mm=0
        
    
if __name__ == '__main__':
    
    main()
