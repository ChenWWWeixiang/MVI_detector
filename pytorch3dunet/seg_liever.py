import SimpleITK as sitk
import os,xlrd,glob
import numpy as np
import cv2

from skimage import measure

MOD=['dwi_ADC','t1_PRE','t1_A','t1_V','t1_POST','t2']
th=[[80,180],[190,255],[190,255],[190,250],[190,250],[110,220]]
root='/mnt/data1/mvi2/nrrd_set_new3'
output_jpg='/mnt/data1/mvi2/img_test'
output='/mnt/data1/mvi2/liverseg'
os.makedirs(output,exist_ok=True)
os.makedirs(output_jpg,exist_ok=True)
files=glob.glob(os.path.join(root,'*_segs_*.nrrd'))
patients=list(set([f.split('/')[-1].split('_')[0] for f in files]))
patients.sort()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
error=['B758047','K0356066','Y2308371','Y3889340','Y3694856','Y3680536','Y3618222']
dark=['F837180','G163438','K0242432','K0251050','Y3194079','Y3211079','Y3214716','Y3233897','Y3297829','Y3298933'
      ,'Y3347050','Y3380149','Y3382710','Y3391616','Y3399889','Y3418999','Y3439756','Y3448012','Y3490714','Y3538908',
      'Y3548056','Y3589049','Y3762428']
def remap_gray(I,idx,darkmod=0):
    if darkmod==0:
        if idx==0:
            I[I<200]=200
            I[I >2200] = 2200
        elif idx<2:
            I[I<-0]=-0
            I[I >400]= 400
        elif idx==5:
            I[I<-0]=-0
            I[I >750]= 750
        else:
            I[I<-0]=-0
            I[I >600]= 650
    if darkmod==1:
        if idx==0:
            I[I<200]=200
            I[I >2200] = 2200
        elif idx<2:
            I[I<-0]=-0
            I[I >200]= 200
        elif idx==5:
            I[I<-0]=-0
            I[I >350]= 350
        else:
            I[I<-0]=-0
            I[I >300]= 350
    if darkmod==3:
        if idx==0:
            I[I<0]=0
            I[I >3000] = 3000
        elif idx==5:
            I[I < 50] = 50
            I[I >3000]= 3000
        else:
            I[I<50]=50
            I[I >2000]= 2000
    if darkmod==4:
        if idx==0:
            I[I<1150]=1150
            I[I >3200] = 3200
        elif idx==5:
            I[I < 50] = 50
            I[I >3500]= 3500
        else:
            I[I<50]=50
            I[I >2000]= 2000
    if darkmod==2:
        if idx==0:
            I[I<0]=0
            I[I >3000] = 3000
        elif idx==5:
            I[I < -0] = -0
            I[I >2500]= 2500
        else:
            I[I<20]=20
            I[I >2000]= 2000
    I=I-I.min()
    I=I*1.0/I.max()
    return I
for i,name in enumerate(patients):
    if name in dark:
        darkmod=2
    elif name in error:
        darkmod = 4
    else:
        darkmod=3
    now_seg = glob.glob(os.path.join(root, name + '_segs_*.nrrd'))
    ismvi = int(now_seg[0].split('_')[-1].split('.')[0])
    now_seg = sitk.ReadImage(now_seg)
    now_seg = sitk.GetArrayFromImage(now_seg)[0, :, :, :]
    L=[]
    DD=[]
    for idx, mod in enumerate(MOD):

        data = os.path.join(root, name + '_'+mod+'_'+str(ismvi)+'.nrrd')
        data=sitk.ReadImage(data)
        data=sitk.GetArrayFromImage(data)
        data=(remap_gray(data, idx,darkmod)*255).astype(np.uint8)
        rawshape=data.shape
        data=np.reshape(data,(rawshape[0],rawshape[1]*rawshape[2]))
        data = cv2.equalizeHist(data)
        data = np.reshape(data, (rawshape[0], rawshape[1] , rawshape[2]))
        if idx==2:
            D=data
        DD.append((data).astype(np.uint8))
        liver=(data>th[idx][0])*(data<th[idx][1])*255
        L.append(liver)
    all_L=np.concatenate(L,2)
    all_D = np.concatenate(DD, 2)
    L=np.mean(L,0)
    L = cv2.GaussianBlur(L, (15, 15), 0)
    L=(L>170).astype(np.uint8)
    for ite in range(5):
        L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, kernel,iterations=1)
        L = cv2.morphologyEx(L.transpose((1,2,0)), cv2.MORPH_CLOSE, kernel, iterations=1).transpose((2,0,1))
        L = cv2.morphologyEx(L.transpose((2,0,1)), cv2.MORPH_CLOSE, kernel, iterations=1).transpose((1,2,0))
        L = cv2.morphologyEx(L, cv2.MORPH_OPEN, kernel, iterations=1)
        L = cv2.morphologyEx(L.transpose((1,2,0)), cv2.MORPH_OPEN, kernel, iterations=1).transpose((2,0,1))
        L = cv2.morphologyEx(L.transpose((2,0,1)), cv2.MORPH_OPEN, kernel, iterations=1).transpose((1,2,0))
    L = cv2.dilate(L, kernel, iterations=5)
    labels = measure.label(L, connectivity=2)
    max_num=0
    max_pixel=-1
    for j in range(1, np.max(labels) + 1):
        if np.sum(labels == j) > max_num:
            max_num = np.sum(labels == j)
            max_pixel = j
        #print(np.sum(labels == j), np.sum(labels != 0))
        if np.sum(labels == j) > 0.1 * np.sum(labels != 0):
            labels[labels == j] = max_pixel

    labels[labels != max_pixel] = 0
    labels[labels == max_pixel] = 1
    L=labels
    X=[]
    for j in [40,45,50,55]:
        I=np.concatenate([L[j,:,:].astype(np.uint8)*255,all_L[j,:,:]],1)
        R=np.concatenate([L[j,:,:].astype(np.uint8)*255,all_D[j,:,:]],1)
        I=np.concatenate([I,R],0)
        X.append(I)
    X=np.concatenate(X,0)
    cv2.imwrite(os.path.join(output_jpg,name+'.jpg'),X)
    a=1
    L=sitk.GetImageFromArray(L.astype(np.uint8))
    sitk.WriteImage(L,os.path.join(output,name+'.nrrd'))