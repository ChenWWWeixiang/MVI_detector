import h5py
import numpy as np
f = h5py.File('info.h5', 'w')
values1 = np.arange(12).reshape(4, 3)
values2 = np.arange(12).reshape(4, 3)
f.create_dataset(name='raw', data=[1,3,4,values2])
#f.create_dataset(name='label', data=np.array(values2, dtype='int64'))
f.close()