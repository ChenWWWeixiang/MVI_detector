import SimpleITK as sitk
import numpy as np
import os,glob,cv2,h5py,random
from pytorch3dunet.datasets.utils_sitk import *
raw_path='/mnt/newdisk3/data_for2d'
seg_path='/mnt/newdisk3/seg_for2d'
atlas='/mnt/newdisk2/reg/atlas/dataatlas.nrrd'
seg='/mnt/newdisk2/reg/atlas/segatlas.nrrd'
lung='/mnt/newdisk2/reg/atlas/lung.nrrd'
output_path='/mnt/newdisk2/reg/for2d'
os.makedirs(output_path,exist_ok=True)
#os.makedirs(output_path2,exist_ok=True)
type1=os.listdir(raw_path)

atlas=sitk.ReadImage(atlas)
#atlas.SetOrigin((0,0,0))
lung=sitk.ReadImage(lung)
#lung.SetOrigin((0,0,0))type1
seg=sitk.ReadImage(seg)
#seg.SetOrigin((0,0,0))type1
seg_ar=sitk.GetArrayFromImage(seg).astype(np.float32)
Maps=[]
for i in range(seg_ar.shape[-1]):
    t=sitk.GetImageFromArray(seg_ar[:,:,:,i])
    t.CopyInformation(seg)
    Maps.append(t)
elastixImageFilter = sitk.ElastixImageFilter()

elastixImageFilter.SetMovingImage(lung)
#segs = sitk.ReadImage(os.path.join(seg_path, item, item2, patient))
#lung = sitk.GetArrayFromImage(lung)
#lung = (lung > 0.5).astype(np.uint8)
#lung = sitk.GetImageFromArray(lung)
#lung.CopyInformation(atlas)

parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
#parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastixImageFilter.SetParameterMap(parameterMapVector)
#elastixImageFilter.SetMovingMask(sitk.Cast(lung, sitk.sitkUInt8))
#elastixImageFilter.LogToConsoleOn()
random.shuffle(type1)
for item in type1:
    type2=os.listdir(os.path.join(raw_path,item))
    random.shuffle(type2)
    for item2 in type2:
        allpatient=os.listdir(os.path.join(raw_path, item,item2))
        random.shuffle(allpatient)
        for patient in allpatient:
            if os.path.exists(os.path.join(output_path,item,item2,patient.replace('.nii','.seg.nii'))):
                continue
            data = sitk.ReadImage(os.path.join(raw_path,item,item2,patient))
            #ro=data.GetOrigin()
            #data.SetOrigin((0,0,0))
            segs = sitk.ReadImage(os.path.join(seg_path, item, item2, patient))
            #segs.SetOrigin((0,0,0))
            #segs=sitk.GetArrayFromImage(segs)
            #segs=(segs>0.5).astype(np.uint8)
           # segs=sitk.GetImageFromArray(segs)
            #segs.CopyInformation(data)
            elastixImageFilter.SetFixedImage(segs)
            #elastixImageFilter.SetFixedMask(sitk.Cast(segs, sitk.sitkUInt8))
            try:
                elastixImageFilter.Execute()
            except:
                continue
            data_ar=sitk.GetArrayFromImage(data)

            moved = elastixImageFilter.GetResultImage()
            results=np.zeros_like(data_ar).astype(np.uint8)
            trans_atlas = sitk.Transformix(atlas, elastixImageFilter.GetTransformParameterMap())
            try:
                for i in range(len(Maps)):
                    resultLabel = sitk.Transformix(Maps[i], elastixImageFilter.GetTransformParameterMap())
                    resultLabel=sitk.GetArrayFromImage(resultLabel)
                    resultLabel=(resultLabel>0.5).astype(np.uint8)

                    results[resultLabel==1]=i+1
            except:
                continue
            #resultLabel=np.stack(results,-1)
            resultLabel=sitk.GetImageFromArray(results)
            resultLabel.CopyInformation(data)
            #resultLabel.SetOrigin(ro)
            #data.SetOrigin(ro)
            #moved.SetOrigin(ro)
            os.makedirs(os.path.join(output_path,item,item2),exist_ok=True)
            sitk.WriteImage(resultLabel,os.path.join(output_path,item,item2,patient.replace('.nii','.seg.nii')))
            #sitk.WriteImage(data, os.path.join(output_path, item, item2, 'data.nii'))
            #sitk.WriteImage(trans_atlas, os.path.join(output_path, item, item2, 'moved.nii'))
            print('OK!',patient)








