from radiomics import featureextractor
import os,csv,six,glob
import numpy as np
from radiomics_work.utils_sitk import *
mask_path='/mnt/data1/mvi2/nrrd_set_rawres'
#mask2_path='/mnt/data1/mvi2/pred2'
data_path='/mnt/data1/mvi2/nrrd_set_rawres'
def minus_r(data1,data2,mask,row,row_next,name,ismvi):
    ddata1 = os.path.join(data_path, name + '_'+data1 +'_'+ str(ismvi) + '.nrrd')
    ddata2 = os.path.join(data_path, name + '_'+data2 +'_'+ str(ismvi) + '.nrrd')
    ddata1=sitk.GetArrayFromImage(sitk.ReadImage(ddata1))
    ddata2=sitk.GetArrayFromImage(sitk.ReadImage(ddata2))
    delta=sitk.GetImageFromArray((ddata2-ddata1).astype(np.int16))
    mask = sitk.ReadImage(mask)
    delta.CopyInformation(mask)
    result = extractor.execute(delta, mask)
    for idx, (key, val) in enumerate(six.iteritems(result)):
        if idx < 11:
            continue
        if not isinstance(val, (float, int, np.ndarray)):
            continue
        if np.isnan(val):
            val = 0
        # print(val)
        row.append(data2+'-'+data1 + ':' + key)
        row_next.append(val)
    return row, row_next
outputfile='new3.csv'
extractor = featureextractor.RadiomicsFeatureExtractor('RadiomicsParams.yaml')
files=glob.glob(os.path.join(mask_path,'*_segs_*.nrrd'))
patients=list(set([f.split('/')[-1].split('_')[0] for f in files]))
MOD=['dwi_ADC','t1_PRE','t1_A','t1_V','t1_POST','t2']
flag=0
with open(outputfile, 'w+', newline='') as f:
    writer = csv.writer(f)
    for i,name in enumerate(patients):
        row = ['id', 'label']

        now_seg=glob.glob(os.path.join(mask_path,name+'_segs_*.nrrd'))
        ismvi=int(now_seg[0].split('_')[-1].split('.')[0])
        row_next = [name, ismvi*1]
        for idx, mod in enumerate(MOD):
            data = os.path.join(data_path, name + '_'+mod+'_'+str(ismvi)+'.nrrd')
            if not os.path.exists(data):
                break
            imageName=os.path.join(data)
            maskName = now_seg[0]

            try:
                result = extractor.execute(imageName, maskName)
                for idx, (key, val) in enumerate(six.iteritems(result)):
                    if idx<11:
                        continue
                    if not isinstance(val,(float,int,np.ndarray)):
                        continue
                    if np.isnan(val):
                        val=0
                       # print(val)
                    row.append(mod+':'+key)
                    row_next.append(val)
            except Exception:
                print('r1')
            result = extractor.execute(imageName, get_margin(maskName))
            for jj, (key, val) in enumerate(six.iteritems(result)):
                if jj < 11:
                    continue
                if not isinstance(val, (float, int, np.ndarray)):
                    continue
                if np.isnan(val):
                    val = 0
                # print(val)
                row.append(mod + '.margin:' + key)
                row_next.append(val)

        row, row_next=minus_r('t1_PRE', 't1_A',maskName, row, row_next,name,ismvi)
        row, row_next = minus_r('t1_A', 't1_V', maskName, row, row_next,name,ismvi)
        row, row_next = minus_r('t1_V', 't1_POST', maskName, row, row_next,name,ismvi)
        row, row_next = minus_r('t1_PRE', 't1_POST', maskName, row, row_next,name,ismvi)
        row, row_next = minus_r('t1_POST', 't2', maskName, row, row_next,name,ismvi)
        if len(row_next)<1900:
            print('r2')
            continue
        if flag == 0:
            writer.writerow(row)
            flag=1
        writer.writerow(row_next)