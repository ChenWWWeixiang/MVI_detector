import SimpleITK as sitk
import numpy as np
import cv2
def get_resampled_with_box(input,box,resampled_spacing=[1,1,1],l=True):
    resampler = sitk.ResampleImageFilter()
    if l:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(resampled_spacing)
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())
    resampler.SetSize((1000,1000,1000))
    input_ar=sitk.GetArrayFromImage(input)
    input_box_map=np.zeros_like(input_ar)
    input_box_map[box[2]:box[5],box[1]:box[4],box[0]:box[3]]=255
    mask_map=sitk.GetImageFromArray(input_box_map)
    mask_map.CopyInformation(input)

    moving_resampled = resampler.Execute(input)
    moving_resampled_ar = sitk.GetArrayFromImage(moving_resampled)
    try:
        xx, yy, zz = np.where(moving_resampled_ar > 0)
        resampled_data_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    except:
        xx, yy, zz,ee = np.where(moving_resampled_ar > 0)
        resampled_data_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(),0]

    resampled_data = sitk.GetImageFromArray(resampled_data_ar)

    mask_map_re = resampler.Execute(mask_map)
    mask_map_ar = sitk.GetArrayFromImage(mask_map_re)
    if len(mask_map_ar.shape)==4:
        mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(), 0]
    else:
        mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    xx,yy,zz=np.where(mask_map_ar>=128)
    box_new=[xx.min(),yy.min(),zz.min(),xx.max(),yy.max(),zz.max()]#z,y,x
    return  resampled_data,resampled_data_ar,box_new

def get_resampled_with_segs(input,segs,resampled_spacing=[1,1,1],l=True):
    resampler = sitk.ResampleImageFilter()
    if l:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(resampled_spacing)
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())
    resampler.SetSize((1000,1000,1000))

    moving_resampled = resampler.Execute(input)

    moving_resampled_ar = sitk.GetArrayFromImage(moving_resampled)

    try:
        if len(moving_resampled_ar.shape) == 4:
            xx, yy, zz,ee = np.where(moving_resampled_ar > 0)
            resampled_data_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(),0]
        else:
            xx, yy, zz = np.where(moving_resampled_ar > 0)
            resampled_data_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    except:
        a=1

    resampled_data = sitk.GetImageFromArray(resampled_data_ar)
    #resampled_data.CopyInformation(input)
    #resampled_data.SetSpacing(resampled_spacing)

    mask_map_re = resampler.Execute(segs)
    mask_map_ar = sitk.GetArrayFromImage(mask_map_re)

    if len(mask_map_ar.shape)==4:
        mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(), 0]
    else:
        mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    resampled_mask=sitk.GetImageFromArray(mask_map_ar)
    #resampled_mask.CopyInformation(input)
    #resampled_mask.SetSpacing(resampled_spacing)

    return  resampled_data,resampled_data_ar,resampled_mask,mask_map_ar

def get_matched_segs(input,segs,l=True):
    resampler = sitk.ResampleImageFilter()
    if l:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(input.GetSpacing())
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())
    resampler.SetSize(input.GetSize())

    moving_resampled = resampler.Execute(segs)

    moving_resampled_ar = sitk.GetArrayFromImage(moving_resampled)

    resampled_seg = sitk.GetImageFromArray(moving_resampled_ar)
    resampled_seg.CopyInformation(input)
    resampled_seg.SetSpacing(input.GetSpacing())
    if len(moving_resampled_ar.shape)==4:
        moving_resampled_ar=moving_resampled_ar[:,:,:,0]
    return  resampled_seg,moving_resampled_ar
def remap_gray(I,idx):
    if idx==0:
        I[I<-100]=-100
        I[I >2200] = 2200
    elif idx==1:
        I[I<-0]=-0
        I[I >300] = 300
    elif idx==2:
        I[I<-0]=-0
        I[I >300] = 300
    elif idx==3:
        I[I<-0]=-0
        I[I >300] = 300
    elif idx==4:
        I[I<-0]=-0
        I[I >300] = 300
    elif idx==5:
        I[I<-0]=-0
        I[I >400] = 400
    I=I-I.min()
    I=I*1.0/I.max()
    return I

def get_bigger(segs):
    if isinstance(segs,str):
        segs=sitk.ReadImage(segs)
    segs_ar = sitk.GetArrayFromImage(segs)

    kernel = np.ones((3, 3), np.uint8)
    if len(segs_ar.shape)==4:
        segs_ar=segs_ar[:,:,:,0]
    segs_ar_margin=cv2.dilate(segs_ar, kernel, iterations=2)
    segs_margin=sitk.GetImageFromArray(segs_ar_margin)
    segs_margin.CopyInformation(segs)
    return  segs_margin
def get_margin(segs):
    if isinstance(segs,str):
        segs=sitk.ReadImage(segs)
    segs_ar = sitk.GetArrayFromImage(segs)

    kernel = np.ones((3, 3), np.uint8)
    if len(segs_ar.shape)==4:
        segs_ar=segs_ar[:,:,:,0]
    segs_ar_margin=cv2.dilate(segs_ar, kernel, iterations=2) -cv2.erode(segs_ar, kernel, iterations=1)
    segs_margin=sitk.GetImageFromArray(segs_ar_margin)
    segs_margin.CopyInformation(segs)
    return  segs_margin

def get_inside(segs):
    segs=sitk.ReadImage(segs)
    segs_ar = sitk.GetArrayFromImage(segs)
    kernel = np.ones((3, 3), np.uint8)
    segs_ar_margin=cv2.erode(segs_ar, kernel, iterations=2)
    segs_margin=sitk.GetImageFromArray(segs_ar_margin)
    segs_margin.CopyInformation(segs)
    return  segs_margin