import xlrd
from scipy import stats
import numpy as np
import pandas as pd
X_train = pd.read_csv(open('m_results_train.csv', 'r'))
train_id = X_train.pop('Patient')

#X_test = pd.read_csv(open('a_results_test.csv', 'r'))
#test_id = X_test.pop('Patient')
#xe=X_test._values
X=X_train._values
#X=np.concatenate([xt,xe],0)

group1=X[X[:,0]==1,1:]
group0=X[X[:,0]==0,1:]

g0m,g0d1,g0d2=[],[],[]
g1m,g1d1,g1d2=[],[],[]
pearsonr=[]

for i in range(len(group1[0])):
    t = stats.ttest_ind(group1[:, i], group0[:, i]).pvalue
    pearsonr.append(t)
    g1m.append(np.mean(group1[:, i]))
    g0m.append(np.mean(group0[:,i]))
    g1d1.append(np.percentile(group1[:, i], 25))
    g1d2.append(np.percentile(group1[:, i], 75))
    g0d1.append(np.percentile(group0[:, i], 25))
    g0d2.append(np.percentile(group0[:, i], 75))

title=X_train.keys()[1:]
data=np.array([g0m,g0d1,g0d2,g1m,g1d1,g1d2])
data=np.concatenate([np.array(title)[np.newaxis,:],data,np.array(pearsonr)[np.newaxis,:]]).transpose([1,0])
fisrtline=np.array(['feature_name','mean for not MVI','1/4 for not MVI','3/4 for not MVI',
                      'mean for MVI', '1/4 for MVI', '3/4 for MVI','p value'])
data=np.concatenate([fisrtline[np.newaxis,:],data],0)
df=pd.DataFrame(data)
df.to_excel('r_uni_m.xls')