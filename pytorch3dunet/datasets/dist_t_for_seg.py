import SimpleITK as sitk
import numpy as np
import os,glob,cv2,h5py
from tqdm import tqdm
from multiprocessing.dummy import  Pool as threadpool
import scipy.ndimage as snd
from pytorch3dunet.datasets.utils_sitk import *
from scipy.ndimage import distance_transform_edt

raw_path='/mnt/data1/mvi2/cyst_seg'
mvi_data_path='/mnt/data1/mvi2/data2019/MVI'
notmvi_data_path='/mnt/data1/mvi2/data2019/not MVI'

output_path='/mnt/data1/mvi2/h5_3mod_512'
#output_path2='/mnt/data1/mvi2/nrrd_t1post_3mod_noreg'
os.makedirs(output_path,exist_ok=True)
#os.makedirs(output_path2,exist_ok=True)
files=os.listdir(raw_path)
patients=list(set([f.split('_')[0] for f in files]))
MOD=['t1_post','t1_PRE','t2']
pools=threadpool(16)
crop=False
def process(p):
    C = []
    segs = []
    for mod in MOD:
        seg1 = glob.glob(os.path.join(raw_path, p + '_' + mod + '*')) + \
               glob.glob(os.path.join(raw_path, p + '_' + mod.upper() + '*')) + \
               glob.glob(os.path.join(raw_path, p + '_' + mod.lower() + '*'))
        if len(seg1) == 0:
            print(p, mod, 'no seg!')
            continue
        seg1 = sitk.ReadImage(seg1[0])
        segs.append(seg1)
        data_r = glob.glob(os.path.join(mvi_data_path, p)) + glob.glob(os.path.join(notmvi_data_path, p))
        if len(data_r) == 0:
            print(p, mod, 'no seg!')
            continue
        ismvi = len(glob.glob(os.path.join(mvi_data_path, p))) > 0
        #if os.path.exists(os.path.join(output_path, str(int(ismvi)) + '_' + p + '.h5')):
        #    return
        # elastixImageFilter = sitk.ElastixImageFilter()
        data = glob.glob(os.path.join(data_r[0], p + '_' + mod + '*.nrrd')) + \
               glob.glob(os.path.join(data_r[0], p + '_' + mod.upper() + '*.nrrd')) + \
               glob.glob(os.path.join(data_r[0], p + '_' + mod.lower() + '*.nrrd'))
        if len(data) == 0:
            print(p, mod, 'no data')
            continue
        try:
            data_this = sitk.ReadImage(data[0])
        except:
            print(p, 'not such ' + mod + ' file')
            break
        C.append(data_this)
    if len(C) < 3:
        return
    allI = []
    allR = []
    allG=[]
    for c, s in zip(C, segs):
        resampled_data, moving_resampled_ar, resampled_mask, mask_map_ar = get_resampled_with_segs(c, s, newth=60)
        MM, mm = moving_resampled_ar.max(), moving_resampled_ar.min()
        moving_resampled_ar = (moving_resampled_ar - mm) / (MM - mm)

        #I = (mask_map_ar > 0).astype(np.uint8)
        #J = mask_map_ar
        J=distance_transform_edt(mask_map_ar, sampling=[4,1,1])

        Jm = -distance_transform_edt(1-mask_map_ar, sampling=[4, 1, 1])
        I=J/J.max()-Jm/Jm.min()
        #a=1
        rshape = I.shape
        rate = [64 / rshape[0], 352 / rshape[1], 352 / rshape[2]]
        I = snd.zoom(I, rate)[:64, :352, :352]

        moving_resampled_ar = snd.zoom(moving_resampled_ar, rate)[:64, :352, :352]
        mask_map_ar = snd.zoom(mask_map_ar, rate,order=1)[:64, :352, :352]
        # I=I.max()-I
        allI.append(I)
        allG.append(mask_map_ar)
        allR.append(moving_resampled_ar)
    allI = np.stack(allI, -1)
    allR = np.stack(allR, -1)
    allG = np.stack(allG, -1)
    f = h5py.File(os.path.join(output_path, str(int(ismvi)) + '_' + p + '.h5'), 'w')
    f.create_dataset(name='raw', data=np.array(allR, dtype='float32'))
    f.create_dataset(name='label', data=np.array(allI, dtype='float32'))
    f.create_dataset(name='raw-label', data=np.array(allG, dtype='uint8'))
    f.close()
    print(p,'OK!')

#for p in tqdm(patients):
pools.map(process,patients)
pools.close()
pools.join()



