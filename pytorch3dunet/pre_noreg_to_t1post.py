import SimpleITK as sitk
import numpy as np
import os,glob,cv2,h5py
from tqdm import tqdm
from pytorch3dunet.datasets.utils_sitk import *
raw_path='/mnt/data1/mvi2/cyst_seg'
mvi_data_path='/mnt/data1/mvi2/data2019/MVI'
notmvi_data_path='/mnt/data1/mvi2/data2019/not MVI'

output_path='/mnt/data1/mvi2/h5_t1post_3mod_noreg'
output_path2='/mnt/data1/mvi2/nrrd_t1post_3mod_noreg'
os.makedirs(output_path,exist_ok=True)
os.makedirs(output_path2,exist_ok=True)
files=os.listdir(raw_path)
patients=list(set([f.split('_')[0] for f in files]))
MOD=['t1_post','t1_PRE','t2']

crop=False

for p in tqdm(patients):
    C = []
    segs=[]
    for mod in MOD:
    # if os.path.exists( os.path.join(output_path2, p + '.seg.nrrd')):
    #     continue
        seg1=glob.glob(os.path.join(raw_path,p+'_'+mod+'*'))+\
             glob.glob(os.path.join(raw_path,p+'_'+mod.upper()+'*'))+ \
             glob.glob(os.path.join(raw_path, p + '_' + mod.lower() + '*'))
        if len(seg1)==0:
            print(p,mod,'no seg!')
            continue
        seg1 = sitk.ReadImage(seg1[0])
        segs.append(seg1)
        data_r = glob.glob(os.path.join(mvi_data_path, p)) + glob.glob(os.path.join(notmvi_data_path, p))
        if len(data_r)==0:
            print(p,mod,'no seg!')
            continue
        ismvi=len(glob.glob(os.path.join(mvi_data_path, p)))>0
        #elastixImageFilter = sitk.ElastixImageFilter()
        data=glob.glob(os.path.join(data_r[0], p+'_'+mod+'*.nrrd'))+\
             glob.glob(os.path.join(data_r[0], p+'_'+mod.upper()+'*.nrrd'))+ \
             glob.glob(os.path.join(data_r[0], p + '_' + mod.lower() + '*.nrrd'))
        if len(data)==0:
            print(p,mod,'no data')
            continue
        try:
            data_this = sitk.ReadImage(data[0])
        except:
            print(p, 'not such '+mod+' file')
            break
        C.append(data_this)
    if len(C)<3:
        continue
    #fix, _, seg1, _=get_resampled_with_segs(fix, seg1)
    #elastixImageFilter.SetFixedImage(fix)
    resampled_data, moving_resampled_ar, resampled_mask, mask_map_ar = get_resampled_with_batches(C[0], segs[0],C[1:],segs[1:])
    MM,mm=moving_resampled_ar.max(),moving_resampled_ar.min()


    moving_resampled_ar=(moving_resampled_ar-mm)/(MM-mm)
    mask_map_ar=mask_map_ar>0



    f = h5py.File(os.path.join(output_path,str(int(ismvi))+'_'+p+'.h5'), 'w')
    f.create_dataset(name='raw',data=np.array(moving_resampled_ar, dtype='float32'))
    f.create_dataset(name='label', data=np.array(mask_map_ar, dtype='uint8'))
    f.close()
    sitk.WriteImage(resampled_data,os.path.join(output_path2,p+'.data.nrrd'))
    sitk.WriteImage(resampled_mask, os.path.join(output_path2, p + '.seg.nrrd'))

