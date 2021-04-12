import os,random
root='/mnt/data1/mvi2/c2d-gbz'
files=os.listdir(root)
random.shuffle(files)
with open('/mnt/data9/deep_R/pytorch-3dunet/pytorch3dunet/datasets/lists/train_cls2d.list2','w') as f:
    for item in files[:len(files)//2]:
        f.writelines(os.path.join(root,item)+'\n')
with open('/mnt/data9/deep_R/pytorch-3dunet/pytorch3dunet/datasets/lists/test_cls2d.list2','w') as f:
    for item in files[len(files)//2:]:
        f.writelines(os.path.join(root,item)+'\n')
