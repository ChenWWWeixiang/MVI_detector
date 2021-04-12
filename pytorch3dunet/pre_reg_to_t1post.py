import SimpleITK as sitk
import numpy as np
import os,glob,cv2,h5py
from datasets.utils_sitk import *
raw_path='/mnt/data1/mvi2/cyst_seg'
mvi_data_path='/mnt/data1/mvi2/data2019/MVI'
notmvi_data_path='/mnt/data1/mvi2/data2019/not MVI'

output_path='/mnt/data1/mvi2/h5_t1post_3mod_reg_New'
output_path2='/mnt/data1/mvi2/nrrd_t1post_3mod_reg_New'
os.makedirs(output_path,exist_ok=True)
os.makedirs(output_path2,exist_ok=True)
files=os.listdir(raw_path)
patients=list(set([f.split('_')[0] for f in files]))

ELSEMOD=['t1_PRE','t2']
MOD='t1_post'
crop=False
Elastix=False
if Elastix:
    elastixImageFilter = sitk.ElastixImageFilter()
else:
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsJointHistogramMutualInformation()
    R.SetMetricSamplingPercentage(0.01)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([4, 2, 1])
    R.SetOptimizerAsGradientDescent(
        learningRate=0.1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInterpolator(sitk.sitkBSpline)


for p in patients:
    #get images
    seg1 = glob.glob(os.path.join(raw_path, p + '_' + MOD + '*')) + glob.glob(
        os.path.join(raw_path, p + '_' + MOD.upper() + '*'))
    if len(seg1) == 0:
        print(p, 'no seg!')
        continue
    seg1 = sitk.ReadImage(seg1[0])
    data_r = glob.glob(os.path.join(mvi_data_path, p)) + glob.glob(os.path.join(notmvi_data_path, p))
    if len(data_r) == 0:
        print(p, 'no seg!')
        continue
    ismvi = len(glob.glob(os.path.join(mvi_data_path, p))) > 0
    # elastixImageFilter = sitk.ElastixImageFilter()
    data = glob.glob(os.path.join(data_r[0], p + '_' + MOD + '*.nrrd')) + glob.glob(
        os.path.join(data_r[0], p + '_' + MOD.upper() + '*.nrrd'))
    if len(data) == 0:
        print(p, 'no data')
        continue
    try:
        fix = sitk.ReadImage(data[0],sitk.sitkFloat32)
        #fix=sitk.Cast(fix)
    except:
        print(seg1, 'not such T1_post file')
        continue
    othermods = []

    #set fixed
    if Elastix:
        elastixImageFilter.SetFixedImage(fix)
    for mod in ELSEMOD:
        data_this = glob.glob(os.path.join(data_r[0], p + '_' + mod + '*.nrrd')) + \
                    glob.glob(os.path.join(data_r[0], p + '_' + mod.upper() + '*.nrrd')) + \
                    glob.glob(os.path.join(data_r[0], p + '_' + mod.lower() + '*.nrrd'))
        if len(data_this) == 0:
            print(p, 'not such ' + mod + ' file')
            break
        try:
            data_this = sitk.ReadImage(data_this[0])
            data_this=sitk.Cast(data_this,sitk.sitkFloat32)
        except:
            print(p, 'not such ' + mod + ' file')
            break
        #set moving 
        if Elastix:
            elastixImageFilter.SetMovingImage(data_this)
            elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
            try:
                elastixImageFilter.Execute()
                moved = elastixImageFilter.GetResultImage()
                # moved=sitk.GetArrayFromImage(moved)
                moved.CopyInformation(fix)
            except:
                break
        else:
            R.SetInitialTransform(
                        sitk.CenteredTransformInitializer(
                            data_this,
                            fix,
                            sitk.AffineTransform(3),
                            sitk.CenteredTransformInitializerFilter.MOMENTS,
                        )
                    )
            outTx = R.Execute(data_this, fix)

            

            interpolator = sitk.sitkBSpline
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolator)
            resampler.SetDefaultPixelValue(0)
            resampler.SetReferenceImage(fix)
            resampler.SetTransform(outTx)
            moved=resampler.Execute(data_this)
            #move_raw=resampler.Execute(data)

        othermods.append(moved)
    if len(othermods) < 2:
        continue
    # fix, _, seg1, _=get_resampled_with_segs(fix, seg1)
    # elastixImageFilter.SetFixedImage(fix)
    resampled_data, moving_resampled_ar, resampled_mask, mask_map_ar = get_resampled_with_segs_and_raws(fix, seg1,
                                                                                                        othermods)
    MM, mm = moving_resampled_ar.max(), moving_resampled_ar.min()

    moving_resampled_ar = (moving_resampled_ar - mm) / (MM - mm)
    mask_map_ar = mask_map_ar > 0

    f = h5py.File(os.path.join(output_path, str(int(ismvi)) + '_' + p + '.h5'), 'w')
    f.create_dataset(name='raw', data=np.array(moving_resampled_ar, dtype='float32'))
    f.create_dataset(name='label', data=np.array(mask_map_ar, dtype='uint8'))
    f.close()
    #sitk.WriteImage(resampled_data, os.path.join(output_path2, p + '.data.nrrd'))
    #sitk.WriteImage(resampled_mask, os.path.join(output_path2, p + '.seg.nrrd'))






