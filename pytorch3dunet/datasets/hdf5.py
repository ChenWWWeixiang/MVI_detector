import glob
import os
from itertools import chain
from multiprocessing import Lock
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms
import h5py
import numpy as np

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('HDF5Dataset')
lock = Lock()


class AbstractHDF5Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path,
                 phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 raw_internal_path='raw',
                 label_internal_path='label',
                 weight_internal_path=None):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :para'/home/adrian/workspace/ilastik-datasets/VolkerDeconv/train'm slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        :param weight_internal_path (str or list): H5 internal path to the per pixel weights
        """
        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = phase
        self.file_path = file_path

        # convert raw_internal_path, label_internal_path and weight_internal_path to list for ease of computation
        if isinstance(raw_internal_path, str):
            raw_internal_path = [raw_internal_path]
        if isinstance(label_internal_path, str):
            label_internal_path = [label_internal_path]
        if isinstance(weight_internal_path, str):
            weight_internal_path = [weight_internal_path]

        internal_paths = list(raw_internal_path)
        if label_internal_path is not None:
            internal_paths.extend(label_internal_path)
        if weight_internal_path is not None:
            internal_paths.extend(weight_internal_path)

        input_file = self.create_h5_file(file_path, internal_paths)

        self.raws = self.fetch_and_check(input_file, raw_internal_path)
        if self.raws[0].ndim==4:
            self.raws[0]=np.transpose(self.raws[0],(3,0,1,2))
        min_value, max_value, mean, std = self.ds_stats()

        self.transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                      mean=mean, std=std)
        #self.raw_transform = self.transformer.raw_transform()

        self.cls = int(input_file.filename.split('/')[-1].split('_')[0])
        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()
            self.labels = self.fetch_and_check(input_file, label_internal_path)
            self.labels[0] = (self.labels[0]>0.5)*1.0
            if self.labels[0].ndim == 4:
                self.labels[0] = np.transpose(self.labels[0], (3, 0, 1, 2))

            if weight_internal_path is not None:
                # look for the weight map in the raw file
                self.weight_maps = self.fetch_and_check(input_file, weight_internal_path)
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_maps = None

            self._check_dimensionality(self.raws, self.labels)
            slice_builder = get_slice_builder(self.raws, self.labels, self.weight_maps, slice_builder_config)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.labels = self.fetch_and_check(input_file, label_internal_path)
            self.labels[0] = (self.labels[0] > 0.5) * 1.0
            self.weight_maps = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                padded_volumes = []

                for raw in self.raws:
                    if raw.ndim == 4:
                        channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in raw]
                        padded_volume = np.stack(channels)
                    else:
                        padded_volume = np.pad(raw, pad_width=pad_width, mode='reflect')

                    padded_volumes.append(padded_volume)

                self.raws = padded_volumes
            slice_builder = get_slice_builder(self.raws, None, self.weight_maps, slice_builder_config)
        # build slice indices for raw and label data sets
        #slice_builder = get_slice_builder(self.raws, self.labels, self.weight_maps, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices
        self.cls=[self.cls]*len(self.raw_slices)
        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    def ds_stats(self):
        # calculate global min, max, mean and std for normalization
        min_value, max_value, mean, std = calculate_stats(self.raws)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')
        return min_value, max_value, mean, std

    @staticmethod
    def create_h5_file(file_path, internal_paths):
        raise NotImplementedError

    @staticmethod
    def fetch_datasets(input_file_h5, internal_paths):
        raise NotImplementedError

    def fetch_and_check(self, input_file_h5, internal_paths):
        datasets = self.fetch_datasets(input_file_h5, internal_paths)
        # expand dims if 2d
        fn = lambda ds: np.expand_dims(ds, axis=0) if ds.ndim == 2 else ds
        datasets = list(map(fn, datasets))
        return datasets

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        #raw_patch_transformed = self._transform_patches(self.raws, raw_idx, self.raw_transform)

        if self.phase == 'test':
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            raw_patch_transformed, _ = self.do_augmentation(self.raws[0][raw_idx])
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            raw_patch_transformed, label_patch_transformed=self.do_augmentation(self.raws[0][raw_idx], self.labels[0][label_idx])
            #label_patch_transformed = self._transform_patches(self.labels, label_idx, self.label_transform)

            return raw_patch_transformed,label_patch_transformed,self.cls[idx]

    def do_augmentation(self, array, label=None):
        """Augmentation for the training data.

                :array: A numpy array of size [c, x, y, z]
                :returns: augmented image and the corresponding mask

                """
        # normalize image to range [0, 1], then apply this transform
        patch_size = [32, 128, 128]
        if array.ndim==4:
            mc=1
            #array=np.transpose(array,(3,0,1,2))
        else:
            mc=0
            array = array[np.newaxis, :, :, :]
        if isinstance(label,np.ndarray):
            if label.ndim == 4:
                mcl = 1
                # array=np.transpose(array,(3,0,1,2))
            else:
                mcl = 0
                label = label[np.newaxis, :, :, :]
            # start = time.time()
            augmented, label = spatial_transforms.augment_resize(array, label, [40,144,144])
        # end = time.time()
        # print('resize',end-start)
        #augmented = noise_transforms.augment_gaussian_noise(
        #    array, noise_variance=(0, .05))
        #augmented = noise_transforms.augment_gaussian_blur(
        #    augmented, sigma_range=(1.2, 1.5))

        # start = time.time()
        # print('noise',start-end)
        # augmented = spatial_transforms.augment_mirroring(augmented,)[0]
        # end = time.time()
        # print('miror',end-start)
        # augmented=spatial_transforms.augment_rot90(augmented,None,axes=(1,2))[0]
        # start = time.time()
        # print('rot90',start-end)
        # need to become [bs, c, x, y, z] before augment_spatial
        else:
            augmented=array
        augmented = augmented[None, ...]
        if isinstance(label, np.ndarray):
            label = label[None, ...]
            r_range = (0, (15 / 360.) * 2 * np.pi)
            cval = 0.
            augmented, label = spatial_transforms.augment_spatial(
                augmented, seg=label, patch_size=patch_size, patch_center_dist_from_border=[8, 50, 50],
                do_elastic_deform=True, alpha=(0., 150.), sigma=(7., 11.),
                do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
                do_scale=True, scale=(.8, 1.2),
                border_mode_data='constant', border_cval_data=cval,
                order_data=3,
                p_el_per_sample=0.5,
                p_scale_per_sample=.5,
                p_rot_per_sample=.5,
                random_crop=True
            )
            if mc==1:
                return augmented[0,:, :, :, :], label[0, :,:, :, :]
            else:
        #if isinstance(label, np.ndarray):
                return augmented[0, :, :, :], label[0, :, :, :]
        else:
            return augmented[0, :, :, :],label
    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the label data and apply the label transformer
            transformed_patch = transformer(dataset[label_idx])
            transformed_patches.append(transformed_patch)

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _check_dimensionality(raws, labels):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        for raw, label in zip(raws, labels):
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

            #assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = cls.traverse_h5_paths(file_paths)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              weight_internal_path=dataset_config.get('weight_internal_path', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_h5_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results


class StandardHDF5Dataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from all of the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, mirror_padding=(16, 32, 32),
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path,
                         weight_internal_path=weight_internal_path)

    @staticmethod
    def create_h5_file(file_path, internal_paths):
        return h5py.File(file_path[:-1], 'r')

    @staticmethod
    def fetch_datasets(input_file_h5, internal_paths):
        return [input_file_h5[internal_path][...] for internal_path in internal_paths]


class LazyHDF5Dataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data lazily. It's slower, but has a low memory footprint.

    The problem of loading h5 dataset from multiple loader workers results in an error:

        # WARN: we load everything into memory due to hdf5 bug when reading H5 from multiple subprocesses, i.e.
        # File "h5py/_proxy.pyx", line 84, in h5py._proxy.H5PY_H5Dread
        # OSError: Can't read data (inflate() failed)

    this happens when the H5 dataset is compressed. The workaround is to create the uncompressed datasets
    from a single worker (synchronization is necessary) and use them instead. Assuming the user specified internal
    dataset path as PATH, this will create a corresponding `_uncompressed_PATH` dataset inside the same H5 file.

    Unfortunately even after fixing the above error, reading the H5 from multiple worker threads sometimes
    returns corrupted data and as a result. e.g. cross-entropy loss fails with: RuntimeError: CUDA error: device-side assert triggered.

    This can be workaround by using only a single worker thread, i.e. set `num_workers: 1` in the config.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, mirror_padding=(16, 32, 32),
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path,
                         weight_internal_path=weight_internal_path)

    @staticmethod
    def create_h5_file(file_path, internal_paths):
        # make this part mutually exclusive
        lock.acquire()

        uncompressed_paths = {}
        for internal_path in internal_paths:
            if internal_path is not None:
                assert '_uncompressed' not in internal_path
                uncompressed_paths[internal_path] = f'_uncompressed_{internal_path}'

        with h5py.File(file_path[:-1], 'r+') as f:
            for k, v in uncompressed_paths.items():
                if v not in f:
                    # create uncompressed dataset
                    data = f[k][...]
                    f.create_dataset(v, data=data)

        lock.release()

        # finally return the H5
        return h5py.File(file_path[:-1], 'r')

    @staticmethod
    def fetch_datasets(input_file_h5, internal_paths):
        # convert to uncompressed
        internal_paths = [f'_uncompressed_{internal_path}' for internal_path in internal_paths]
        return [input_file_h5[internal_path] for internal_path in internal_paths]

    def ds_stats(self):
        # Do not calculate stats on the whole stacks when using lazy loader,
        # they min, max, mean, std should be provided in the config
        logger.info(
            'Using LazyHDF5Dataset. Make sure that the min/max/mean/std values are provided in the loaders config')
        return None, None, None, None


class RSModelHDF5Dataset(ConfigDataset):
    """
    Implementation of the HDF5 dataset which loads the data from all of the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, mirror_padding=(8, 16, 16),
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None):
        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            mirror_padding = None
        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"
        self.mirror_padding = mirror_padding
        self.phase = phase
        self.file_path = file_path

        # convert raw_internal_path, label_internal_path and weight_internal_path to list for ease of computation
        if isinstance(raw_internal_path, str):
            raw_internal_path = [raw_internal_path]
        if isinstance(label_internal_path, str):
            label_internal_path = [label_internal_path]

        internal_paths = list(raw_internal_path)
        if label_internal_path is not None:
            internal_paths.extend(label_internal_path)

        input_file = self.create_h5_file(file_path, internal_paths)
        self.raws = self.fetch_and_check(input_file, raw_internal_path)
        self.raws[0]=np.transpose(self.raws[0] ,(3,0,1,2))

        min_value, max_value, mean, std = self.ds_stats()

        self.transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                      mean=mean, std=std)
        # self.raw_transform = self.transformer.raw_transform()

        self.cls = [int(input_file.filename.split('/')[-1].split('_')[0])]

            # create label/weight transform only in train/val phase
        self.label_transform = self.transformer.label_transform()
        self.labels = self.fetch_and_check(input_file, label_internal_path)
        #self.labels[0] = (self.labels[0] > 0.5) * 1.0
        self.labels[0] = np.transpose(self.labels[0], (3, 0, 1, 2))
        if phase == 'test':
            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                padded_volumes = []
                for raw in self.raws:
                    if raw.ndim == 4:
                        channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in raw]
                        padded_volume = np.stack(channels)
                    else:
                        padded_volume = np.pad(raw, pad_width=pad_width, mode='reflect')

                    padded_volumes.append(padded_volume)
                self.raws = padded_volumes
        #self.cls = [self.cls] * len(self.raws)
        self.count = len(self.raws)
        #logger.info(f'Number of patches: {self.count}')
    def ds_stats(self):
        # calculate global min, max, mean and std for normalization
        min_value, max_value, mean, std = calculate_stats(self.raws)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')
        return min_value, max_value, mean, std
    @staticmethod
    def create_h5_file(file_path, internal_paths):
        return h5py.File(file_path[:-1], 'r')

    @staticmethod
    def fetch_datasets(input_file_h5, internal_paths):
        return [input_file_h5[internal_path][...] for internal_path in internal_paths]

    def fetch_and_check(self, input_file_h5, internal_paths):
        datasets = self.fetch_datasets(input_file_h5, internal_paths)
        # expand dims if 2d
        fn = lambda ds: np.expand_dims(ds, axis=0) if ds.ndim == 2 else ds
        datasets = list(map(fn, datasets))
        return datasets

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        # get the slice for a given index 'idx'
        raws = self.raws[idx]
        # get the raw data patch for a given slice
        #raw_patch_transformed = self._transform_patches(self.raws, raw_idx, self.raw_transform)

        if self.phase == 'test':
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            #raw_patch_transformed, _ = self.do_augmentation(raws)
            return raws
        else:
            # get the slice for a given index 'idx'
            gt = self.labels[idx]
            #raw_patch_transformed, label_patch_transformed=self.do_augmentation(raws,gt)
            #label_patch_transformed = self._transform_patches(self.labels, label_idx, self.label_transform)
            return raws,gt,self.cls[idx]

    def do_augmentation(self, array, label=None):
        """Augmentation for the training data.

                :array: A numpy array of size [c, x, y, z]
                :returns: augmented image and the corresponding mask

                """
        # normalize image to range [0, 1], then apply this transform
        patch_size = [64, 128, 128]
        if array.ndim==4:
            mc=1
            #array=np.transpose(array,(3,0,1,2))
        else:
            mc=0
            array = array[np.newaxis, :, :, :]
        if isinstance(label,np.ndarray):
            if label.ndim == 4:
                mcl = 1
                # array=np.transpose(array,(3,0,1,2))
            else:
                mcl = 0
                label = label[np.newaxis, :, :, :]
            # start = time.time()
            augmented, label = spatial_transforms.augment_resize(array, label, [70,144,144])

        else:
            augmented=array
        augmented = augmented[None, ...]
        if isinstance(label, np.ndarray):
            label = label[None, ...]
            r_range = (0, (15 / 360.) * 2 * np.pi)
            cval = 0.
            augmented, label = spatial_transforms.augment_spatial(
                augmented, seg=label, patch_size=patch_size, patch_center_dist_from_border=[20, 30, 30],
                do_elastic_deform=True, alpha=(0., 150.), sigma=(7., 12.),
                do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
                do_scale=True, scale=(.8, 1.2),
                border_mode_data='constant', border_cval_data=cval,
                order_data=3,
                p_el_per_sample=0.5,
                p_scale_per_sample=.5,
                p_rot_per_sample=.5,
                random_crop=True
            )
            if mc==1:
                return augmented[0,:, :, :, :], label[0, :,:, :, :]
            else:
        #if isinstance(label, np.ndarray):
                return augmented[0, :, :, :], label[0, :, :, :]
        else:
            return augmented[0, :, :, :],label

    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the label data and apply the label transformer
            transformed_patch = transformer(dataset[label_idx])
            transformed_patches.append(transformed_patch)

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches

    def __len__(self):
        return self.count

    @staticmethod
    def _check_dimensionality(raws, labels):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        for raw, label in zip(raws, labels):
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

            #assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = cls.traverse_h5_paths(file_paths)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              weight_internal_path=dataset_config.get('weight_internal_path', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_h5_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results