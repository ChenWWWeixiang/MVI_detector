import SimpleITK as sitk
import numpy as np
import os,glob,cv2,h5py
from utils_sitk import *
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
raw_path='/mnt/data1/mvi2/data2019/labels'
mvi_data_path='/mnt/data1/mvi2/data2019/MVI'
notmvi_data_path='/mnt/data1/mvi2/data2019/not MVI'
output_path='/mnt/data1/mvi2/croped_2_328'

os.makedirs(output_path,exist_ok=True)
#os.makedirs(output_path2,exist_ok=True)
files=os.listdir(raw_path)
patients=list(set([f.split('_')[0] for f in files]))
error=['B758047','K0356066','Y2308371','Y3889340','Y3694856','Y3680536','Y3618222']
dark=['F837180','G163438','K0242432','K0251050','Y3194079','Y3211079','Y3214716','Y3233897','Y3297829','Y3298933'
      ,'Y3347050','Y3380149','Y3382710','Y3391616','Y3399889','Y3418999','Y3439756','Y3448012','Y3490714','Y3538908',
      'Y3548056','Y3589049','Y3762428']
MOD_else=['dwi_ADC','t1_A','t1_V','t1_PRE','t2']
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
    seg1=glob.glob(os.path.join(raw_path,p+'_T1_post*'))
    seg1 = sitk.ReadImage(seg1[0])
    data_r = glob.glob(os.path.join(mvi_data_path, p)) + glob.glob(os.path.join(notmvi_data_path, p))
    ismvi=len(glob.glob(os.path.join(mvi_data_path, p)))>0
    #if os.path.exists(os.path.join(output_path,str(int(ismvi))+'_'+p+'.h5')):
    #    continue
    elastixImageFilter = sitk.ElastixImageFilter()
    data=glob.glob(os.path.join(data_r[0], p+'_*.nrrd'))
    mod='t1_POST'
    data=[d for d in data if d.split('/')[-1].split(p)[1].split('.nrrd')[0][1:]==mod or
          d.split('/')[-1].split(p)[1].split('.nrrd')[0][1:].upper()==mod.upper()]
    try:
        fix=sitk.ReadImage(data[0],sitk.sitkFloat32)
    except:
        print(seg1,'not such t1_POST file')
        continue
    #fix, _, seg1, _=get_resampled_with_segs(fix, seg1)
    elastixImageFilter.SetFixedImage(fix)
    C.append(fix)

    for idx,mod in enumerate(MOD_else):
        data=glob.glob(os.path.join(data_r[0], p+'_*.nrrd'))
        data=[d for d in data if d.split('/')[-1].split(p)[1].split('.nrrd')[0][1:]==mod or
              d.split('/')[-1].split(p)[1].split('.nrrd')[0][1:].upper()==mod.upper()]
        if len(data)>0:
            try:
                move = sitk.ReadImage(data[0],sitk.sitkFloat32)
            except:
                print(seg1,'not such '+mod+' file')
                continue
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
            rs, S = get_matched_segs(C[i], seg1)
            rs.CopyInformation(C[i])
            zz,yy,xx=np.where(S>0)
            center=np.array([zz.min()+zz.max(),yy.min()+yy.max(),xx.min()+xx.max()])/2
            width=np.array([zz.max()-zz.min(),yy.max()-yy.min(),xx.max()-xx.min()])/2
            bg=(center-np.max([width,np.array([50,112,112])],0)).astype(np.int)
            ed=(center+np.max([width,np.array([50,112,112])],0)).astype(np.int)

        C[i] = sitk.GetArrayFromImage(C[i])
        pad_data = (remap_gray(C[i], i, darkmod) * 255).astype(np.uint8)
        rawshape = pad_data.shape
        pad_data = np.reshape(pad_data, (rawshape[0], rawshape[1] * rawshape[2]))
        pad_data = cv2.equalizeHist(pad_data)
        temp = np.reshape(pad_data, (rawshape[0], rawshape[1], rawshape[2]))

        padding=[[0,0],
            [int(max(-bg[1],0)),int(max(ed[1]-rawshape[1],0))],
            [int(max(-bg[2],0)),int(max(ed[2]-rawshape[2],0))]]
        temp=np.pad(temp,padding)
        temp=temp[zz.min()+1:max(zz.max()-1,zz.min()+2),bg[1]+padding[1][0]:ed[1]+padding[1][0],bg[2]+padding[2][0]:ed[2]+padding[2][0]]
        C[i]=temp

    C = np.array(C)
    padding_c=np.zeros((6,1,100,224,224))
    padding_c[:C.shape[0],0,:C.shape[1],:C.shape[2],:C.shape[3]]=C
    mask=np.where(padding_c>0)
    t=np.zeros((100,1))
    t[mask[0]]=1
    #cv2.imwrite('temp.jpg',C[0,5,:,:])
    print(p,'ok!')
    f = h5py.File(os.path.join(output_path,str(int(ismvi))+'_'+p+'.h5'), 'w')
    f.create_dataset(name='raw',data=np.array(padding_c, dtype='float32'))
    f.create_dataset(name='mask',data=np.array(t, dtype='uint8'))
    #f.create_dataset(name='label', data=np.array(S, dtype='uint8'))
    f.close()