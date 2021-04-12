from math import pi
import numpy as np
import torch,h5py,os,random
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms
import torchvision.transforms.functional as functional
import random,cv2
import torchvision.transforms as transforms
import PIL,pickle
class StatefulRandomCrop(object):
    def __init__(self, insize, outsize):
        self.size = outsize
        self.cropParams = self.get_params(insize, self.size)

    @staticmethod
    def get_params(insize, outsize):
        """Get parameters for ``crop`` for a random crop.
        Args:
            insize (PIL Image): Image to be cropped.
            outsize (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = insize
        th, tw = outsize
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """

        i, j, h, w = self.cropParams

        return functional.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
class RadomWindow(object):
    def __init__(self, bg, ed):
        self.bg=bg
        self.ed=ed
    def __call__(self, img):
        """
        Args:
            img (tensor): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        #I=[]
        I=[torch.clip(img.clone(),self.bg[i],self.ed[i]) for i in range(len(self.bg))]
        I=[(i-i.min())/(i.max()-i.min()+1e-4) for i in I]
        I=torch.stack(I)
        return I

    def __repr__(self):
        return self.__class__.__name__ + '(bg={0}, ed={1})'.format(str(self.edbg), str(self.ed))

class StatefulRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.rand = random.random()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.rand < self.p:
            return functional.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class MviSet(Dataset):
    #/data1/data2019/croped_for_cls/0_Y4537592.h5
    def __init__(self,list,istrain):
        self.list=open(list,'r').readlines()
        self.istrain=istrain

        super().__init__()
    def __getitem__(self, index: int):
        file_path=self.list[index]
        file=h5py.File(file_path[:-1], 'r')
        data = np.array(file['raw'])
        mask =np.array(file['mask'])
        gt=int(file_path.split('/')[-1].split('_')[0])
        if self.istrain:
            data,_=self.do_augmentation(data,)
            #data=data[:,:,:32,:,:].squeeze(1)
        else:
            data=data[:,:,:32,:,:].squeeze(1)
        mask=mask[:32,:]
        data=(data*1.0)/data.max()
        return data,mask,[gt]
    def __len__(self) -> int:
        return len(self.list)
    def do_augmentation(self, array, label=None):
        """Augmentation for the training data.

                :array: A numpy array of size [c, x, y, z]
                :returns: augmented image and the corresponding mask

                """
        # normalize image to range [0, 1], then apply this transform
        patch_size = [32, 224, 224]
        
        
        augmented=array[:,:,:32,:,:].squeeze(1)
        augmented, label = spatial_transforms.augment_resize(augmented, label, [32,244,244])
        augmented = spatial_transforms.augment_mirroring(augmented,axes=(1,2))[0]

        #augmented=noise_transforms.augment_gaussian_noise(augmented)#no!

        #augmented=noise_transforms.augment_blank_square_noise(augmented,square_size=(10,10),n_squares=4)
       # augmented=noise_transforms.augment_gaussian_blur(augmented,sigma_range=(5,10))
        #augmented=noise_transforms.augment_rician_noise(augmented)
        # end = time.time()
        # print('miror',end-start)
        augmented=spatial_transforms.augment_rot90(augmented,None,axes=(1,2))[0]
        augmented=augmented[np.newaxis,:,:,:,:]
        # augmented, label = spatial_transforms.augment_spatial(
        #         augmented, seg=label, patch_size=patch_size, patch_center_dist_from_border=[15,110,110],
        #         do_elastic_deform=True, #alpha=(0., 100.), sigma=(8, 11.),
        #         do_rotation=False, angle_x=(0,pi*30/180), angle_y=(0,pi*30/180), angle_z=(0,0),
        #         do_scale=True, scale=(.8, 1.2),
        #         border_mode_data='constant', border_cval_data=0,
        #         order_data=3,
        #         p_el_per_sample=0.5,
        #         p_scale_per_sample=.5,
        #         p_rot_per_sample=.5,
        #         random_crop=True
        #     )

        return augmented[0, :, :, :,:],None


class MviSet2D(Dataset):
    #/data1/data2019/croped_for_cls/0_Y4537592.h5
    def __init__(self,list,istrain,options):
        super(MviSet2D,self).__init__()
        self.list=open(list,'r').readlines()
        self.istrain=istrain
        random.shuffle(self.list)
        self.options=options
    def get_whole_V(self,index):
        file_path=self.list[index]
        file=h5py.File(file_path[:-1], 'r')
        data = np.array(file['raw'])
        cropped =np.array(file['cropped'])
        gt=int(file_path.split('/')[-1].split('_')[0])
        X1=[]
        X2=[]
        for i in range(data.shape[1]):
            x1,x2=self.bbc(data[:,i,:,:][:,:,:],cropped[:,0,i,:,:],self.istrain)
            X1.append(x1)
            X2.append(x2)
        X1=np.stack(X1,0)
        X2=np.stack(X2,0)
        file.close()
        return X1,X2,[gt]


    def __getitem__(self, index: int):
        # file_path=self.list[index]
        # file=h5py.File(file_path[:-1], 'r')
        # data = np.array(file['raw'])
        # cropped =np.array(file['cropped'])
        # gt=int(file_path.split('/')[-1].split('_')[0])
        data=self.data[index]
        cropped=self.crop[index]
        gt=self.gt[index]
        #valid_len=data.shape[1]
        #sliceid=random.randint(1,valid_len)-1
        #for i in range(6):
        #    cv2.imwrite(f'temp{i}.jpg',cropped[i,0,sliceid,:,:])
    #cv2.imwrite('temp1.jpg',data[5,0,:,:])
        #sliceid=2
        #data=data[:,sliceid,:,:][:,np.newaxis,:,:]
        #cropped=cropped[:,:,sliceid,:,:]
        data,cropped=self.bbc(data,cropped,self.istrain)
        return data,cropped,[gt]
    def load_things(self):
        name=str(self.istrain)
        if os.path.exists(name+'temp_data.pkl'):
            self.data,self.crop,self.gt=pickle.load(open(name+'temp_data.pkl','rb'))
        else:
            self.data=[]
            self.crop=[]
            self.gt=[]
            for file_path in self.list:
                file=h5py.File(file_path[:-1], 'r')
                data = np.array(file['raw'])
                self.data.append(data)
                cropped =np.array(file['cropped'])
                self.crop.append(cropped)
                gt=int(file_path.split('/')[-1].split('_')[0])
                self.gt.append(np.repeat(gt,data.shape[1]))
                file.close()
            self.data=np.concatenate(self.data,1).transpose(1,0,2,3)
            self.crop=np.concatenate(self.crop,2).squeeze(1).transpose(1,0,2,3)
            self.gt=np.concatenate(self.gt,0)
            pickle.dump([self.data, self.crop,self.gt],open(name+'temp_data.pkl','wb'))
        
    def __len__(self) -> int:
        return len(self.data)
    def bbc(self,V, cV, augmentation=True):
        R = torch.zeros((6,6, 256, 256))
        C = torch.zeros((6,6, 224, 224))
        cV=cV.astype(np.uint8)
        if (augmentation):
            crop = StatefulRandomCrop((288, 288), (256, 256))
            flip = StatefulRandomHorizontalFlip(0.5)
            croptransform = transforms.Compose([
                crop,
                flip
            ])
        else:
            croptransform = transforms.CenterCrop((256, 256))

        for cnt,i in enumerate(range(V.shape[0])):
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256),interpolation=PIL.Image.BICUBIC),
                #transforms.CenterCrop((256, 256)),
                croptransform,
                transforms.ToTensor(),
                RadomWindow(self.options['thsets'],self.options['thsets2']),
                transforms.Normalize(0, 1),
            ])(V[i,:,:])
            R[cnt] = result[:,0,:,:]
        if (augmentation):
            crop = StatefulRandomCrop((256, 256), (224, 224))
            flip = StatefulRandomHorizontalFlip(0.5)
            croptransform = transforms.Compose([
                crop,
                flip
            ])
        else:
            croptransform = transforms.CenterCrop((224, 224))
        for cnt,i in enumerate(range(cV.shape[0])):
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256),interpolation=PIL.Image.BICUBIC),
                #transforms.CenterCrop((256, 256)),
                croptransform,
                transforms.ToTensor(),
                RadomWindow(self.options['thsets'],self.options['thsets2']),
                transforms.Normalize(0,1),
            ])(cV[i,:,:])
            C[:cnt] = result[:,0,:,:]
        return R,C

class SetWarpper(MviSet2D):
    def __init__(self,list,istrain,options):
        super(SetWarpper,self).__init__(list,istrain,options)
    def __getitem__(self, index: int): 
        file_path=self.list[index]
        file=h5py.File(file_path[:-1], 'r')
        data = np.array(file['raw'])
        cropped =np.array(file['cropped'])
        gt=int(file_path.split('/')[-1].split('_')[0])
        X1=[]
        X2=[]
        for i in range(data.shape[1]):
            x1,x2=self.bbc(data[:,i,:,:][:,:,:],cropped[:,0,i,:,:],self.istrain)
            X1.append(x1)
            X2.append(x2)
        X1=np.stack(X1,0)
        X2=np.stack(X2,0)
        file.close()
        return X1,X2,[gt]
    def __len__(self) -> int:
        return len(self.list)

##a=MviSet2D('/mnt/data9/deep_R/pytorch-3dunet/pytorch3dunet/datasets/lists/train_cls2d.list',True)
#b=a[1]