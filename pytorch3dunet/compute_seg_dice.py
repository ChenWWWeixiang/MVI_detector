import SimpleITK as sitk
import h5py,os,glob
import  numpy as np
from skimage import measure
import cv2
def DiceIndex(a, b):
    #ss=a.shape
    b=b[:a.shape[0],:a.shape[1],:a.shape[2]]
    a = a[:b.shape[0], :b.shape[1], :b.shape[2]]
    return (np.sum(a * b == 1) * 2.0 + 1e-5) / (np.sum(a == 1) + np.sum(b == 1) * 1.0)

ww=open('record-dice.txt','w')

input_pred='/mnt/data1/mvi2/pred_t1post'
output_pred='/mnt/data1/mvi2/pred_X'
#os.makedirs(output_pred,exist_ok=True)
input_gt='/mnt/data1/mvi2/h5_t1post'
All_dice=[]
for item in glob.glob(input_pred+'/*.nrrd'):
    name = item .split('/')[-1].split('_')[1]
    pred=sitk.ReadImage(item)
    pred=sitk.GetArrayFromImage(pred).astype(np.uint8)
    gt_name=os.path.join(input_gt,item .split('/')[-1].split('_')[0]+'_'+
                         name+'.h5')
    gt=np.array(h5py.File(gt_name, 'r')['label'])
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # L = cv2.dilate(pred, kernel, iterations=2)
    # for ite in range(3):
    #     L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, kernel,iterations=1)
    #     L = cv2.morphologyEx(L.transpose((1,2,0)), cv2.MORPH_CLOSE, kernel, iterations=1).transpose((2,0,1))
    #     L = cv2.morphologyEx(L.transpose((2,0,1)), cv2.MORPH_CLOSE, kernel, iterations=1).transpose((1,2,0))
    #     L = cv2.morphologyEx(L, cv2.MORPH_OPEN, kernel, iterations=1)
    #     L = cv2.morphologyEx(L.transpose((1,2,0)), cv2.MORPH_OPEN, kernel, iterations=1).transpose((2,0,1))
    #     L = cv2.morphologyEx(L.transpose((2,0,1)), cv2.MORPH_OPEN, kernel, iterations=1).transpose((1,2,0))
    # L = cv2.erode(L, kernel, iterations=2)
    # labels = measure.label(L, connectivity=2)
    # max_num = 0
    # max_pixel = 0
    # for j in range(1, np.max(labels) + 1):
    #     if np.sum(labels == j) > max_num:
    #         max_num = np.sum(labels == j)
    #         max_pixel = j
    #     print(item, str(j)+':',np.sum(labels == j), np.sum(labels != 0))
    #     if np.sum(labels == j) > 0.25 * np.sum(labels != 0):
    #         labels[labels == j] = max_pixel
    #
    # labels[labels != max_pixel] = 0
    # labels[labels == max_pixel] = 1

    dice=DiceIndex(gt, pred)
    #sitk.WriteImage(sitk.GetImageFromArray(pred.astype(np.uint8)),os.path.join(output_pred,item.split('/')[-1]))
    #sitk.WriteImage(sitk.GetImageFromArray(gt.astype(np.uint8)), os.path.join(output_pred, 'gt_'+item.split('/')[-1]))
    All_dice.append(dice)
    ww.writelines(name+'\t'+str(dice)+'\n')
print(np.mean(All_dice))
#test part : 0.5584895383645027
#train part : 0.30880066197179945