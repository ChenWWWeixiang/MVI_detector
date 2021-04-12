from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC,SVR
import  pandas as pd
import numpy as np
import sys,random
sys.path.append('/mnt/data9/radiomics_work')
from toolss import EarlyStopping
from sklearn.feature_selection import RFE,RFECV
import sklearn.neural_network as n
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import time,joblib,os,xlrd,logging
import scipy.stats as stats
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch.nn
import torch.optim
from sklearn import decomposition
log_path='/mnt/data9/deep_R/pytorch-3dunet/radiomics_work/mnoc.txt'
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
if os.path.exists(log_path):
    os.remove(log_path)
handler = logging.FileHandler(log_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)

pca = decomposition.PCA(n_components=20)
no_use_feature=['CTC(循环肿瘤细胞)','乙肝病毒DNA','丙肝病毒DNA']
isPCA=False
def read_xlsx(workbook,cls):
    booksheet = workbook.sheet_by_name(cls)
    p = dict()
    for row in range(2,booksheet.nrows,7):
        #row_data = dict()
        id=booksheet.cell(row, 0).value
        f=[]
        for i in range(1,41):
            featurename=booksheet.cell(1, i).value
            if featurename in no_use_feature:
                continue
            value=booksheet.cell(row, i).value
            if value=='*':
                value=0
            if value=='':
                value=0
            if value=='＜20':
                value=0
            if value=='＜100':
                value=50
            if value=='＜500':
                value=250
            if value=='＞24200':
                value=20000
            f.append(value)
        for i in range(44,48):
            value=float(booksheet.cell(row, i).value)
            f.append(value)
        for i in range(52,56):
            value=float(booksheet.cell(row, i).value)
            f.append(value)
        p[id]=f
    return p
def read_xlm_data():
    path='/mnt/data1/mvi2/clinic.xlsx'
    workbook = xlrd.open_workbook(path)
    mvi = read_xlsx(workbook, 'MVI')
    Nmvi = read_xlsx(workbook, 'not MVI')
    mvi.update(Nmvi.items())
    return mvi
clinics=read_xlm_data()
class SmallModel(torch.nn.Module):
    def __init__(self,hl,inf):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.BatchNorm1d(inf),#14736
                                      torch.nn.Linear(inf, hl),  #8
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(0.3),
                                      torch.nn.BatchNorm1d(hl),)
        self.output=torch.nn.Sequential(torch.nn.Linear(256+hl, 512),  # 128
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(0.5),
                                      torch.nn.BatchNorm1d(512),
                                      torch.nn.Linear(512, 2))
        self.clinic=torch.nn.Sequential(torch.nn.BatchNorm1d(6),
                                      torch.nn.Linear(6, 256),  #8
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(0.2),
                                      torch.nn.BatchNorm1d(256))
        self.recon=torch.nn.Sequential(torch.nn.BatchNorm1d(hl),
                                      torch.nn.Linear(hl, inf),  #8
                                      torch.nn.ReLU())

        #pass
    def forward(self, intput,clinic):
        f1=self.fc(intput)
        rec=self.recon(f1)
        f2=self.clinic(clinic)
        return self.output(torch.cat([f1,f2],-1)),rec
       # return self.output(f1),self.output(f1)
        #return self.output(f2),self.output(f2)
#SVM=False
def filter_t(dataset):
    for item in dataset.keys():
        if not '-original' in item:
            dataset.pop(item)
    return dataset
def filter_name(dataset,name):
    for item in dataset.keys():
        if not item+'\n' in name:
            dataset.pop(item)
    return dataset
def filter_mod(dataset,mod):
    for item in dataset.keys():
        if isinstance(mod,list):
            flag = 0
            for i in mod:
                if i in item:
                    flag=1
                    break
            if flag==0:
                dataset.pop(item)
        else:
            if mod not in item:
                dataset.pop(item)
    return dataset
def filter_clinic(C,clinicidx):
    C=C[:,clinicidx]
    return C
def filter_size(dataset,clinic,y_test,r):
    C=[]
    D=[]
    Y=[]
    for item in range(clinic.shape[0]):
        if np.max(clinic[item,-3:]) >= r[0] and np.max(clinic[item,-3:])<=r[1]:
            D.append(dataset[item,:])
            C.append(clinic[item, :])
            Y.append(y_test[item])
    D=np.array(D)
    C=np.array(C)
    Y = np.array(Y)
    return D,C,Y
sr=[6,20]#[3,6],[6,20]
Mod=['_','dwi','t1_PRE','t1_A','t1_V','t1_POST','T2']#T1PRE,
#Mod=[['t1_PRE','T2'],['t1_PRE','t1_POST'],['T2','t1_POST'],['T2','t1_POST','t1_PRE']]
#Mod=[['t1_POST','t1_PRE']]
#Mod=Mod+[['t1_#POST','T2']]
Mod=['t1_PRE']
#name=open('names.txt','r')
#name=name.readlines()
#name=name[:200]
MM=0
clinicidx=[4,43,22,14,1,2]
#print('m')
def remap_train_test(X_train,X_test,seed=938):
    X=pd.concat([X_train,X_test],0)
    X1=X[X['label']==1]
    X0=X[X['label']==0]
    #Y=np.concatenate([y_train,y_test],0)
    X1_train, X1_test=train_test_split(X1, test_size=0.5,random_state=seed)#22:75#1:74#2170
    X0_train, X0_test = train_test_split(X0, test_size=0.5, random_state=seed)
    return pd.concat([X1_train,X0_train],0),pd.concat([X1_test,X0_test],0)

for mod in Mod:
    Acc2 = []
    AUC = []
    sen = []
    spe = []
    Acc_t = []
    AUC_t = []
    sen_t = []
    spe_t = []

    mAcc2 = []
    mAUC = []
    msen = []
    mspe = []
    mAcc_t = []
    mAUC_t = []
    msen_t = []
    mspe_t = []
    PP=[]
    GG=[]
    for turns in range(50):
        tt=[]
        seed=random.randint(1,1000)
        #print('seed',seed)
        #print('auto')
        X_train = pd.read_csv(open('/mnt/data9/deep_R/pytorch-3dunet/radiomics_work/m_results_train2.csv', 'r'))
        X_test = pd.read_csv(open('/mnt/data9/deep_R/pytorch-3dunet/radiomics_work/m_results_test2.csv', 'r'))
        X_train, X_test = remap_train_test(X_train, X_test,seed)

        y_train = X_train.pop('label').astype(int)
        train_id = X_train.pop('Patient')
        X_train=filter_mod(X_train, mod)
        #X_train = filter_t(X_train)
        #X_train = filter_name(X_train, name)
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-5)

        y_test = X_test.pop('label').astype(int)
        test_id = X_test.pop('Patient')

        X_test = filter_mod(X_test, mod)

        #X_test=filter_t(X_test)
        #X_test = filter_name(X_test, name)
        X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + 1e-5)
        #X_test=X_test.fillna(0)
        X_test=X_test._values
        #X_test[X_test==np.nan]=0
        X_train = X_train._values
        y_train = y_train._values
        y_test = y_test._values
        if isPCA:
            TT=pca.fit(X_train).components_
            X_train=pca.fit_transform(X_train)
            X_test=np.matmul(X_test,TT.T)
        #X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + 1e-5)
        #X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-5)
        C_train=[]
        inf =X_train.shape[1]
        #print(inf)
        for id,item in enumerate(train_id._values):
            clinic_f=clinics[item]
            C_train.append(clinic_f)
        C_train=np.array(C_train)
        C_train=filter_clinic(C_train, clinicidx)
        C_test=[]
        for id,item in enumerate(test_id._values):
            clinic_f=clinics[item]
            C_test.append(clinic_f)
        C_test=np.array(C_test)

        X_test,C_test,y_test=filter_size(X_test,C_test,y_test,sr)
        #print(y_test.shape[0],np.sum(y_test==1))
        C_test = filter_clinic(C_test, clinicidx)
        
        C_test = (C_test - C_test.min(axis=0)) / (C_test.max(axis=0) - C_test.min(axis=0) + 1e-5)
        C_train = (C_train - C_train.min(axis=0)) / (C_train.max(axis=0) - C_train.min(axis=0) + 1e-5)
        # X_train = torch.tensor(X_train).float().cuda()
        # y_train = torch.tensor(y_train).long().cuda()
        # C_train = torch.tensor(C_train).float().cuda()
        C_test = torch.tensor(C_test).float().cuda()
        X_test = torch.tensor(X_test).float().cuda()
        y_test = torch.tensor(y_test).long().cuda()
        for hl in [512]:
            try:
                model = SmallModel(hl,inf).cuda()
                early_stopping = EarlyStopping(50, verbose=False)
                cri = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
                Xtrain_fold, Xval_fold, Ctrain_fold, Cval_fold,Ytrain_fold, Yval_fold = train_test_split(X_train, C_train,y_train, test_size=0.2)
                Xtrain_fold = torch.tensor(Xtrain_fold).float().cuda()
                Ytrain_fold = torch.tensor(Ytrain_fold).long().cuda()
                Ctrain_fold = torch.tensor(Ctrain_fold).float().cuda()
                Cval_fold = torch.tensor(Cval_fold).float().cuda()
                Xval_fold = torch.tensor(Xval_fold).float().cuda()
                Yval_fold = torch.tensor(Yval_fold).long().cuda()
                last=0
                weight=10
                for turn in range(5000):
                    model.train()
                    optimizer.zero_grad()
                    pred,rec = model(Xtrain_fold,Ctrain_fold)
                    loss=cri(pred, Ytrain_fold)
                    #loss = (cri(pred, Ytrain_fold)+torch.nn.functional.smooth_l1_loss(rec,Xtrain_fold)*weight)/(1+weight)
                    loss.backward()
                    optimizer.step()
                    #if turn%100==0:
                    #    print(loss)
                    weight=weight*0.9
                    model.eval()
                    pred,rec = model(Xval_fold, Cval_fold)
                    loss = cri(pred, Yval_fold)
                    #loss=loss.detach().cpu().numpy()
                    early_stopping(loss)
                    if early_stopping.early_stop:
                        #print("Early stopping")
                        break
                    #last=loss
                model.eval()
                pred ,rec= model(Xval_fold,Cval_fold)
                pred=pred.softmax(-1)
                pred = pred[:, 1].detach().cpu().numpy()
                pred1 = pred > 0.5
                gt = Yval_fold.cpu().numpy()
                Acc2.append(np.mean(pred1 == gt))
                sen.append(np.sum((pred1 == gt) * (gt == 1)) / (np.sum((gt == 1)) + 1e-5))
                spe.append(np.sum((pred1 == gt) * (gt == 0)) / (np.sum((gt == 0)) + 1e-5))
                AUC.append(metrics.roc_auc_score(gt, pred))
 
                pred,rec = model(X_test, C_test)
                pred=pred.softmax(-1)
                pred = pred[:, 1].detach().cpu().numpy()
                pred1 = pred > 0.5
                gt = y_test.cpu().numpy()
                Acc_t.append(np.mean(pred1 == gt))
                sen_t.append(np.sum((pred1 == gt) * (gt == 1)) / (np.sum((gt == 1)) + 1e-5))
                spe_t.append(np.sum((pred1 == gt) * (gt == 0)) / (np.sum((gt == 0)) + 1e-5))
                AUC_t.append(metrics.roc_auc_score(gt, pred))
                PP+=pred.tolist()
                GG+=gt.tolist()
            except Exception:
                #print('error!',Exception)
                continue
            
            tt.append(np.mean(AUC_t))
        #aa=AUC_t
        #continue
       # print('man')
        if mod=='T1_PRE21':
            X_train = pd.read_csv(open('/mnt/data9/deep_R/pytorch-3dunet/radiomics_work/m_results_train2.csv', 'r'))
            X_test = pd.read_csv(open('/mnt/data9/deep_R/pytorch-3dunet/radiomics_work/m_results_test2.csv', 'r'))
            X_train, X_test = remap_train_test(X_train, X_test, seed)

            y_train = X_train.pop('label').astype(int)
            train_id = X_train.pop('Patient')
            X_train = filter_mod(X_train, mod)
            # X_train = filter_t(X_train)
            # X_train = filter_name(X_train, name)
            X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-5)

            y_test = X_test.pop('label').astype(int)
            test_id = X_test.pop('Patient')

            X_test = filter_mod(X_test, mod)

            # X_test=filter_t(X_test)
            # X_test = filter_name(X_test, name)
            X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + 1e-5)
            # X_test=X_test.fillna(0)
            X_test = X_test._values
            # X_test[X_test==np.nan]=0
            X_train = X_train._values
            y_train = y_train._values
            y_test = y_test._values
            if isPCA:
                TT = pca.fit(X_train).components_
                X_train = pca.fit_transform(X_train)
                X_test = np.matmul(X_test, TT.T)
            # X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + 1e-5)
            # X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-5)
            C_train = []
            inf = X_train.shape[1]
            #print(inf)
            for id, item in enumerate(train_id._values):
                clinic_f = clinics[item]
                C_train.append(clinic_f)
            C_train = np.array(C_train)
            C_test = []
            for id, item in enumerate(test_id._values):
                clinic_f = clinics[item]
                C_test.append(clinic_f)
            C_test = np.array(C_test)
            C_train=filter_clinic(C_train, clinicidx)
            C_test=filter_clinic(C_test, clinicidx)
            
            X_test,C_test,y_test=filter_size(X_test,C_test,y_test,sr)
            # print(X_test.shape[0])

            C_test = (C_test - C_test.min(axis=0)) / (C_test.max(axis=0) - C_test.min(axis=0) + 1e-5)
            C_train = (C_train - C_train.min(axis=0)) / (C_train.max(axis=0) - C_train.min(axis=0) + 1e-5)
            # X_train = torch.tensor(X_train).float().cuda()
            # y_train = torch.tensor(y_train).long().cuda()
            # C_train = torch.tensor(C_train).float().cuda()
            C_test = torch.tensor(C_test).float().cuda()
            X_test = torch.tensor(X_test).float().cuda()
            y_test = torch.tensor(y_test).long().cuda()
            for hl in [256]:
                try:
                    model = SmallModel(hl, inf).cuda()
                    early_stopping = EarlyStopping(50, verbose=False)
                    cri = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
                    Xtrain_fold, Xval_fold, Ctrain_fold, Cval_fold, Ytrain_fold, Yval_fold = train_test_split(X_train,
                                                                                                                C_train,
                                                                                                                y_train,
                                                                                                                test_size=0.2)
                    Xtrain_fold = torch.tensor(Xtrain_fold).float().cuda()
                    Ytrain_fold = torch.tensor(Ytrain_fold).long().cuda()
                    Ctrain_fold = torch.tensor(Ctrain_fold).float().cuda()
                    Cval_fold = torch.tensor(Cval_fold).float().cuda()
                    Xval_fold = torch.tensor(Xval_fold).float().cuda()
                    Yval_fold = torch.tensor(Yval_fold).long().cuda()
                    last = 0
                    weight = 10
                    for turn in range(5000):
                        model.train()
                        optimizer.zero_grad()
                        pred, rec = model(Xtrain_fold, Ctrain_fold)
                        loss = cri(pred, Ytrain_fold)
                        # loss = (cri(pred, Ytrain_fold)+torch.nn.functional.smooth_l1_loss(rec,Xtrain_fold)*weight)/(1+weight)
                        loss.backward()
                        optimizer.step()
                        # if turn%100==0:
                        #    print(loss)
                        weight = weight * 0.99
                        model.eval()
                        pred, rec = model(Xval_fold, Cval_fold)
                        loss = cri(pred, Yval_fold)
                        # loss=loss.detach().cpu().numpy()
                        early_stopping(loss)
                        if early_stopping.early_stop:
                            # print("Early stopping")
                            break
                        # last=loss
                    model.eval()
                    pred, rec = model(Xval_fold, Cval_fold)
                    pred = pred[:, 1].detach().cpu().numpy()
                    pred1 = pred > 0.5
                    gt = Yval_fold.cpu().numpy()
                    mAcc2.append(np.mean(pred1 == gt))
                    msen.append(np.sum((pred1 == gt) * (gt == 1)) / (np.sum((gt == 1)) + 1e-5))
                    mspe.append(np.sum((pred1 == gt) * (gt == 0)) / (np.sum((gt == 0)) + 1e-5))
                    mAUC.append(metrics.roc_auc_score(gt, pred))

                    pred, rec = model(X_test, C_test)
                    pred = pred[:, 1].detach().cpu().numpy()
                    pred1 = pred > 0.5
                    gt = y_test.cpu().numpy()
                    mAcc_t.append(np.mean(pred1 == gt))
                    msen_t.append(np.sum((pred1 == gt) * (gt == 1)) / (np.sum((gt == 1)) + 1e-5))
                    mspe_t.append(np.sum((pred1 == gt) * (gt == 0)) / (np.sum((gt == 0)) + 1e-5))
                    mAUC_t.append(metrics.roc_auc_score(gt, pred))
                except:
                    continue
            
    logger.info('auto: '+str(hl)+'\t'+str(mod)+'\t'+str(sr)+'\tvalidation\t'+str(np.mean(Acc2).round(4))+'+-'+str(np.std(Acc2).round(4))+'\t'+
        str(np.mean(AUC).round(4)) + '+-' + str(np.std(AUC).round(4))+'\t'+
        str(np.mean(sen).round(4)) + '+-' + str(np.std(sen).round(4))+'\t'+
        str(np.mean(spe).round(4)) + '+-' + str(np.std(spe).round(4))+'\t'+
        'test\t'+str(np.mean(Acc_t).round(4))+'+-'+str(np.std(Acc_t).round(4))+'\t'+
        str(np.mean(AUC_t).round(4)) + '+-' + str(np.std(AUC_t).round(4))+'\t'+
        str(np.mean(sen_t).round(4)) + '+-' + str(np.std(sen_t).round(4))+'\t'+
        str(np.mean(spe_t).round(4)) + '+-' + str(np.std(spe_t).round(4)))
    if mod=='t1_PRE21':
        logger.info('mannual: '+str(hl)+'\t'+str(mod)+'\t'+str(sr)+'\tvalidation\t'+str(np.mean(mAcc2).round(4)) + '+-' + str(np.std(mAcc2).round(4))+'\t'+
            str(np.mean(mAUC).round(4)) + '+-' + str(np.std(mAUC).round(4))+'\t'+
            str(np.mean(msen).round(4)) + '+-' + str(np.std(msen).round(4))+'\t'+
            str(np.mean(mspe).round(4)) + '+-' + str(np.std(mspe).round(4))+'\t'+
            'test\t'+str(np.mean(mAcc_t).round(4)) + '+-' + str(np.std(mAcc_t).round(4))+'\t'+
            str(np.mean(mAUC_t).round(4)) + '+-' + str(np.std(mAUC_t).round(4))+'\t'+
            str(np.mean(msen_t).round(4)) + '+-' + str(np.std(msen_t).round(4))+'\t'+
            str(np.mean(mspe_t).round(4)) + '+-' + str(np.std(mspe_t).round(4)))
#print(idx,MM)
f=open('s.txt','w')
f.writelines('Pred Gt\n')
for pp,gg in  zip(PP,GG):
    f.writelines(str(pp)+' '+str(gg)+'\n')
#