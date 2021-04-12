import pandas as pd

liver = pd.read_csv(open('mb_results_train2.csv', 'r'))
name1 = set(liver.pop('Patient')._values)
aliver = pd.read_csv(open('m_results_train.csv', 'r'))
name2 = set(aliver.pop('Patient')._values)
a = pd.read_csv(open('al_results_train.csv', 'r'))
name3 = set(a.pop('Patient')._values)
a=1