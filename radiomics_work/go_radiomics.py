from radiomics.featureextractor import RadiomicsFeatureExtractor
import os,csv,six,glob,h5py
import numpy as np
from radiomics_work.utils_sitk import *
mask_path='/mnt/data1/mvi2/pred_train'
data_path='/mnt/data1/mvi2/h5_set_new3'
outputfile='train_part_newsetting.csv'
extractor = RadiomicsFeatureExtractor('RadiomicsParams.yaml')
files=glob.glob(os.path.join(mask_path,'*.nrrd'))
patients=list(set([f.split('/')[-1].split('_')[1] for f in files]))
MOD=['dwi_ADC','t1_PRE','t1_A','t1_V','t1_POST','T2']
flag=0
with open(outputfile, 'w+', newline='') as f:
    writer = csv.writer(f)
    for i,name in enumerate(patients):
        row = ['name', 'label']
        data=glob.glob(os.path.join(data_path,'*'+name+'.h5'))
        seg = glob.glob(os.path.join(mask_path, '*'+name + '_predictions.nrrd'))
        ismvi=seg[0].split('/')[-1].split('_')[0]
        row_next = [name, ismvi*1]
        f = h5py.File(data[0], 'r')
        datas=f['raw']
        if datas.shape[0]<6:
            continue
        for idx in range(6):
            data=datas[idx,:,:,:]
            imageName=sitk.GetImageFromArray(data)
            maskName = sitk.ReadImage(seg[0])
            try:
                result = extractor.execute(imageName, maskName)
                for j, (key, val) in enumerate(six.iteritems(result)):
                    if j<11:
                        continue
                    #if 'interpolated' in key:
                    #    continue
                    if not isinstance(val,(float,int,np.ndarray)):
                        continue
                    if np.isnan(val):
                        val=0
                       # print(val)
                    row.append(MOD[idx]+':'+key)
                    row_next.append(val)
                result = extractor.execute(imageName,get_margin(maskName))
                #result = extractor.execute(imageName, maskName)
                for j, (key, val) in enumerate(six.iteritems(result)):
                    if j<11:
                        continue
                    #if 'interpolated' in key:
                    #    continue
                    if not isinstance(val,(float,int,np.ndarray)):
                        continue
                    if np.isnan(val):
                        val=0
                       # print(val)
                    row.append(MOD[idx]+'-margin:'+key)
                    row_next.append(val)

            except:
                print(imageName)
        if len(row_next)<112*12:
            continue
        if flag == 0:
            writer.writerow(row)
            flag=1
        writer.writerow(row_next)