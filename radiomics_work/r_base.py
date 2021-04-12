from radiomics import featureextractor
import os,csv,six,glob
import numpy as np
from radiomics_work.utils_sitk import *
mask_path='/mnt/data1/mvi2/seg.1'
#mask2_path='/mnt/data1/mvi2/pred2'
mvi_data_path='/mnt/data1/mvi2/data2019/MVI'
notmvi_data_path='/mnt/data1/mvi2/data2019/not MVI'
outputfile='all_raw.csv'
extractor = featureextractor.RadiomicsFeatureExtractor('RadiomicsParams.yaml')
files=os.listdir(mask_path)
patients=list(set([f.split('_')[0] for f in files]))
MOD=['dwi_ADC','t1_PRE','t1_A','t1_V','t1_POST','T2']
flag=0
with open(outputfile, 'w+', newline='') as f:
    writer = csv.writer(f)
    for i,name in enumerate(patients):
        row = ['id', 'label']
        data_r = glob.glob(os.path.join(mvi_data_path, name)) + glob.glob(os.path.join(notmvi_data_path, name))
        ismvi = len(glob.glob(os.path.join(mvi_data_path, name))) > 0
        seg1 = glob.glob(os.path.join(mask_path, name + '_t1_pre*'))
        seg2 = glob.glob(os.path.join(mask_path, name + '_t1_post*'))
        seg3 = glob.glob(os.path.join(mask_path, name + '_t2*'))
        row_next = [name, ismvi*1]
        for idx, mod in enumerate(MOD):
            data = glob.glob(os.path.join(data_r[0], name + '_*.nrrd'))
            data = [d for d in data if d.split('/')[-1].split(name)[1].split('.nrrd')[0][1:] == mod or
                    d.split('/')[-1].split(name)[1].split('.nrrd')[0][1:].upper() == mod.upper()]
            if len(data)==0:
                break
            imageName=os.path.join(data[0])
            if idx==1:
                maskName = seg1[0]
            elif idx==4:
                maskName = seg2[0]
            elif idx==5:
                maskName = seg3[0]
            else:
                maskName = seg2[0]
            try:
                result = extractor.execute(imageName, maskName)
                for idx, (key, val) in enumerate(six.iteritems(result)):
                    if idx<11:
                        continue
                    if 'interpolated' in key:
                        continue
                    if not isinstance(val,(float,int,np.ndarray)):
                        continue
                    if np.isnan(val):
                        val=0
                       # print(val)
                    row.append(mod+':'+key)
                    row_next.append(val)
                result = extractor.execute(imageName, get_margin(maskName))
                for jj, (key, val) in enumerate(six.iteritems(result)):
                    if jj < 11:
                        continue
                    if 'interpolated' in key:
                        continue
                    if not isinstance(val, (float, int, np.ndarray)):
                        continue
                    if np.isnan(val):
                        val = 0
                    # print(val)
                    row.append(mod + '.margin:' + key)
                    row_next.append(val)
            except:
                print('r1')
        if len(row_next)<1300:
            print('r2')
            continue
        if flag == 0:
            writer.writerow(row)
            flag=1
        writer.writerow(row_next)