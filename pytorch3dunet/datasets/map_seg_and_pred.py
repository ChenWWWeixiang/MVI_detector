import SimpleITK as sitk
import h5py,os,glob
import numpy as np
seg_dir='/mnt/data1/mvi2/h5_set'
pred_dir='/mnt/data1/mvi2/pred'
output='/mnt/data1/mvi2/pred_gt'
os.makedirs(output,exist_ok=True)
#segs=glob.glob(seg_dir+'/*.h5')
preds=glob.glob(pred_dir+'/*.nrrd')
for item in preds:
    data=sitk.ReadImage(item)
    pred=sitk.GetArrayFromImage(data)
    name=item.split('/')[-1].replace('_predictions.nrrd','.h5')
    segs=os.path.join(seg_dir,name)
    f=h5py.File(segs, 'r')
    segs=f['label'][2,:,:,:]
    img=f['raw'][2,:,:,:]
    assert segs.shape==pred.shape
    dice=np.sum(segs*pred)/np.sum((segs+pred)>0)
    print(dice)
    save_seg=sitk.GetImageFromArray(segs)
    save_img=sitk.GetImageFromArray(img)
    sitk.WriteImage(save_seg,os.path.join(output,'{}_label_{:.3f}.nrrd'.format(name.split('.')[0],dice)))
    sitk.WriteImage(data,os.path.join(output, '{}_pred_{:.3f}.nrrd'.format(name.split('.')[0], dice)))
    sitk.WriteImage(save_img, os.path.join(output, '{}_img_{:.3f}.nrrd'.format(name.split('.')[0], dice)))