import xlrd
from scipy import stats
import numpy as np
import pandas as pd
no_use_feature=['CTC(循环肿瘤细胞)','乙肝病毒DNA','丙肝病毒DNA']
def read_xlsx(workbook,cls):
    X_train = pd.read_csv(open('m_results_train2.csv', 'r'))
    X_test = pd.read_csv(open('m_results_test2.csv', 'r'))
    train_id = X_train.pop('Patient')._values.tolist()
    test_id = X_test.pop('Patient')._values.tolist()
    nn_list=train_id+test_id
    booksheet = workbook.sheet_by_name(cls)
    p = dict()
    for row in range(2,booksheet.nrows,7):
        #row_data = dict()
        id=booksheet.cell(row, 0).value
        if not id in nn_list:
            continue
        f=[cls=='MVI']
        for i in range(1,41):
            value=booksheet.cell(row, i).value
            featurename=booksheet.cell(1, i).value
            if featurename in no_use_feature:
                continue
            if value=='*':
                value=-1
            if value=='':
                value=-1
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
group1=np.array([clinics[k][1:] for k in clinics.keys() if clinics[k][0]==True])
group0=np.array([clinics[k][1:] for k in clinics.keys() if clinics[k][0]==False])
g0m,g0d1=[],[]
g1m,g1d1=[],[]
pearsonr=[]

for i in range(len(group1[0])):
    if i >=6 and not i== len(group1[0])-8:
        t = stats.ttest_ind(group1[group1[:, i]>-1, i], group0[group0[:, i]>-1, i],equal_var=False).pvalue
        pearsonr.append(t)
        g1m.append(np.mean(group1[group1[:, i]>-1, i]))
        g0m.append(np.mean(group0[group0[:, i]>-1,i]))
        g1d1.append(np.std(group1[group1[:, i]>-1, i]))
        g0d1.append(np.std(group0[group0[:, i]>-1, i]))
    else:
        try:
            t = stats.ttest_ind(group1[group1[:, i]>-1, i], group0[group0[:, i]>-1, i],equal_var=False).pvalue
            pearsonr.append(t)
        except:
            a=1
        g1m.append(np.sum(group1[group1[:, i]>-1, i]==1))
        g0m.append(np.sum(group0[group0[:, i]>-1, i]==1))
        g0d1.append(np.sum(group0[group0[:, i]>-1, i] != 1))
        g1d1.append(np.sum(group1[group1[:, i]>-1, i] != 1))

title='乙肝表面抗原（1，阳性。0，阴性）	乙肝表面抗体（1，阳性。0，阴性）	乙肝e抗体（1，阳性。0，阴性）	乙肝e抗原（1，阳性。0，阴性）	乙肝核心抗体（1，阳性。0，阴性）	丙肝抗体（1，阳性。0，阴性）	癌胚抗原（0-5）	甲胎蛋白（0-20）	CA125（0.1-35）	CA19-9（0.1-37）	CA15-3（0.1-30）	CA724（0.1-10）	丙氨酸氨基转氨酶（0-40）	天冬酸氨基转氨酶（0-40）	总蛋白（55-80）	血清白蛋白（35-50）	总胆红素（0-21）	直接胆红素（0-8.6）	碱性磷酸酶（0-130）	γ—谷氨酰基转移酶（0-50）	α-L-岩藻糖苷酶（＜40）	凝血酶时间测定（15-21）	血浆活化部分部分凝血活酶时间测定（30-45）	血浆凝血酶原时间测定（11-15）	血浆凝血酶原活动度测定（70-150）	国际标准化比值	血浆纤维蛋白原测定（2-4）	血浆D-二聚体（0-0.5）	血红蛋白	红细胞	白细胞（3.5-10）	中性粒细胞（0.5-0.7）	淋巴细胞（0.2-0.4）	单核细胞（0.03-0.08）	嗜酸性粒细胞（0.01-0.05）	嗜碱性粒细胞（0-0.01）	血小板计数（100-300）'
title=title.split('\t')+'1男2女	年龄	身高cm	体重kg\t肿瘤数量\t肿瘤长\t肿瘤宽\t肿瘤高'.split('\t')
data=np.array([g0m,g0d1,g1m,g1d1])
data=np.concatenate([np.array(title)[np.newaxis,:],data,np.array(pearsonr)[np.newaxis,:]]).transpose([1,0])
fisrtline=np.array(['feature_name','mean for not MVI/ number of value 1','std for not MVI/ number of value 0',
                      'mean for MVI/ number of value 1',  'std for MVI/ number of value 0','p value'])
data=np.concatenate([fisrtline[np.newaxis,:],data],0)
df=pd.DataFrame(data)
df.to_excel('univariate_new.xls')