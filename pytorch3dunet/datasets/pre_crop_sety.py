import SimpleITK as sitk
import numpy as np
import os,glob,cv2,h5py
from pytorch3dunet.datasets.utils_sitk import *
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
raw_path='/mnt/data1/mvi2/seg.1'
mvi_data_path='/mnt/data1/mvi2/data2019/MVI'
notmvi_data_path='/mnt/data1/mvi2/data2019/not MVI'
output_path='/mnt/data1/mvi2/h5_croped+mask'
output_path2='/mnt/data1/mvi2/nrrd_croped+mask'
os.makedirs(output_path,exist_ok=True)
os.makedirs(output_path2,exist_ok=True)
files=os.listdir(raw_path)
patients=list(set([f.split('_')[0] for f in files]))
error=['B758047','K0356066','Y2308371','Y3889340','Y3694856','Y3680536','Y3618222']
dark=['F837180','G163438','K0242432','K0251050','Y3194079','Y3211079','Y3214716','Y3233897','Y3297829','Y3298933'
      ,'Y3347050','Y3380149','Y3382710','Y3391616','Y3399889','Y3418999','Y3439756','Y3448012','Y3490714','Y3538908',
      'Y3548056','Y3589049','Y3762428']
MOD_else=['dwi_ADC','t1_A','t1_V','t1_POST','t2']
MOD_all=['t1_PRE','dwi_ADC','t1_A','t1_V','t1_POST','t2']
crop=False
for p in patients:
    if p in dark:
        darkmod=2
    elif p in error:
        darkmod = 4
    else:
        darkmod=3
    C=[]
    seg1=glob.glob(os.path.join(raw_path,p+'_t1_pre*'))
    seg1 = sitk.ReadImage(seg1[0])
    data_r = glob.glob(os.path.join(mvi_data_path, p)) + glob.glob(os.path.join(notmvi_data_path, p))
    ismvi=len(glob.glob(os.path.join(mvi_data_path, p)))>0
    elastixImageFilter = sitk.ElastixImageFilter()
    data=glob.glob(os.path.join(data_r[0], p+'_*.nrrd'))
    mod='t1_PRE'
    data=[d for d in data if d.split('/')[-1].split(p)[1].split('.nrrd')[0][1:]==mod or
          d.split('/')[-1].split(p)[1].split('.nrrd')[0][1:].upper()==mod.upper()]
    try:
        fix=sitk.ReadImage(data[0])
    except:
        print(seg1,'not such T1_PRE file')
        continue
    #fix, _, seg1, _=get_resampled_with_segs(fix, seg1)
    elastixImageFilter.SetFixedImage(fix)
    C.append(fix)

    for idx,mod in enumerate(MOD_else):
        data=glob.glob(os.path.join(data_r[0], p+'_*.nrrd'))
        data=[d for d in data if d.split('/')[-1].split(p)[1].split('.nrrd')[0][1:]==mod or
              d.split('/')[-1].split(p)[1].split('.nrrd')[0][1:].upper()==mod.upper()]
        if len(data)>0:
            move = sitk.ReadImage(data[0])
        else:
            print(seg1, 'not such'+mod+' file')
            break
        elastixImageFilter.SetMovingImage(move)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
        elastixImageFilter.Execute()

        moved=elastixImageFilter.GetResultImage()
        #moved=sitk.GetArrayFromImage(moved)
        moved.CopyInformation(fix)
        C.append(moved)

    if len(C) < 6:
        print( 'jump' + p )
        continue
    for i in range(6):
        if i==0:
            rs, s = get_matched_segs(C[i], seg1)
            #S=s
            S = cv2.morphologyEx(s, cv2.MORPH_DILATE, kernel, iterations=2)
            rs.CopyInformation(C[i])
            #sitk.WriteImage(rs, os.path.join(output_path2, p + '_segs_' + str(int(ismvi)) + '.nrrd'))
            #sitk.WriteImage(C[i], os.path.join(output_path2, p + '_' + MOD_all[i] + '_' + str(int(ismvi)) + '.nrrd'))
        #else:
            #rc, C[i], rs, s = get_resampled_with_segs(C[i], seg1,must_shape=rc0)
            #sitk.WriteImage(C[i],os.path.join(output_path2,p+'_'+MOD_all[i]+'_'+str(int(ismvi))+'.nrrd'))
        C[i] = sitk.GetArrayFromImage(C[i])
        pad_data = (remap_gray(C[i], i, darkmod) * 255).astype(np.uint8)
        rawshape = pad_data.shape
        pad_data = np.reshape(pad_data, (rawshape[0], rawshape[1] * rawshape[2]))
        pad_data = cv2.equalizeHist(pad_data)
        temp = np.reshape(pad_data, (rawshape[0], rawshape[1], rawshape[2]))
        xx,yy,zz=np.where(S>0)
        temp = temp * S
        temp=temp[xx.min():xx.max(),yy.min():yy.max(),zz.min():zz.max()]
        C[i]=temp

    C = np.array(C)
    f = h5py.File(os.path.join(output_path,str(int(ismvi))+'_'+p+'.h5'), 'w')
    f.create_dataset(name='raw',data=np.array(C, dtype='uint8'))
    f.create_dataset(name='label', data=np.array(S, dtype='uint8'))
    f.close()
