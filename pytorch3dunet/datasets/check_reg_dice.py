import SimpleITK as sitk
import numpy as np
import os,glob,cv2,h5py
from utils_sitk import *
raw_path='/mnt/data1/mvi2/cyst_seg'
reg_path='/mnt/data1/mvi2/regseg_new'
os.makedirs(reg_path,exist_ok=True)
mvi_data_path='/mnt/data1/mvi2/data2019/MVI'
notmvi_data_path='/mnt/data1/mvi2/data2019/not MVI'


files=os.listdir(raw_path)
patients=list(set([f.split('_')[0] for f in files]))

MOD_else=['t1_pre','t2']
crop=False
dd=[]
old=[]

Elastix=True
if Elastix:
    elastixImageFilter = sitk.ElastixImageFilter()
else:
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    #R.SetMetricAsJointHistogramMutualInformation()
    R.SetMetricSamplingPercentage(0.01)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([4, 2, 1])
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                           minStep=1e-4,
                                           numberOfIterations=500,
                                           gradientMagnitudeTolerance=1e-8)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInterpolator(sitk.sitkBSpline)
MOD='t1_post'
for p in patients:
    C=[]
    CB=[]
    seg1_n=glob.glob(os.path.join(raw_path,p+'_t1_post*'))+glob.glob(os.path.join(raw_path,p+'_T1_POST*'))
    if len(seg1_n)==0:
        print(p, 'not such seg file')
        continue
        #continue
    seg1 = sitk.ReadImage(seg1_n[0])
    data_r = glob.glob(os.path.join(mvi_data_path, p)) + glob.glob(os.path.join(notmvi_data_path, p))
    ismvi=len(glob.glob(os.path.join(mvi_data_path, p)))>0

    #data=glob.glob(os.path.join(data_r[0], p+'_*.nrrd'))
    if len(data_r)==0:
        print(seg1, 'not such T1_post file !')
        continue
    
    data = glob.glob(os.path.join(data_r[0], p + '_' + MOD + '*.nrrd')) + glob.glob(
        os.path.join(data_r[0], p + '_' + MOD.upper() + '*.nrrd'))
    if len(data)==0:
        print(0, 'not such T1_post file')
        continue
    try:
        fix=sitk.ReadImage(data[0],sitk.sitkFloat32)
        #fix=sitk.Cast(fix,)
    except:
        print(p, 'load error')
        continue
       # continue
    #fix, _, seg1, _=get_resampled_with_segs(fix, seg1)
    if Elastix:
        elastixImageFilter.SetFixedImage(fix)
    #sitk.WriteImage(fix, seg1_n[0].replace('seg.1', 'regseg.1').replace('seg.nrrd','nrrd'))
    #sitk.WriteImage(seg1, seg1_n[0].replace('seg.1', 'regseg.1'))
    seg1, s = get_matched_segs(fix, seg1)
    C.append(sitk.GetArrayFromImage(seg1))
    for idx,mod in enumerate(MOD_else):
        #data = glob.glob(os.path.join(data_r[0], p + '_*.nrrd'))
        data = glob.glob(os.path.join(data_r[0], p + '_' + mod + '*.nrrd')) + \
                    glob.glob(os.path.join(data_r[0], p + '_' + mod.upper() + '*.nrrd')) + \
                    glob.glob(os.path.join(data_r[0], p + '_' + mod.lower() + '*.nrrd'))
        seg_n_p=glob.glob(os.path.join(raw_path,p+'_'+mod+'*.nrrd'))+glob.glob(os.path.join(raw_path,p+'_'+mod.upper()+'*.nrrd'))+\
            glob.glob(os.path.join(raw_path,p+'_'+mod.lower()+'*.nrrd'))
        if len(seg_n_p)==0:
            print(p, 'not such ' + mod + ' file')
            break
        seg_n = sitk.ReadImage(seg_n_p[0])
        #seg_n=sitk.Cast(seg_n,sitk.sitkFloat32)
        if len(data)>0:
            move = sitk.ReadImage(data[0],sitk.sitkFloat32)
           # move=sitk.Cast(move,sitk.sitkFloat32)
        else:
            print(p, 'load error')
            break
        
        if len(seg_n.GetSize())==4:
            seg_n_a=sitk.GetArrayFromImage(seg_n)[:,:,:,0]
            seg_n_a=sitk.GetImageFromArray(seg_n_a)
            seg_n_a.CopyInformation(seg_n)
            seg_n_a=sitk.Cast(seg_n_a,sitk.sitkFloat32)
        else:
            seg_n_a=seg_n
        if Elastix:
            elastixImageFilter.SetMovingImage(move)
            #parameterMapVector = sitk.VectorOfParameterMap()
            #parameterMapVector.append(sitk.GetDefaultParameterMap("rigid"))
            #parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
            #elastixImageFilter.SetParameterMap(parameterMapVector)
            pm=sitk.GetDefaultParameterMap("rigid")
            pm['Transform']=['EulerTransform']
            pm['HowToCombineTransforms']=['Compose']
            pm['Scales']=['17000','17000', '140000','1','1', '1']
            pm['AutomaticTransformInitialization']=['true']

            pm['UseDirectionCosines']=['true']
            pm['NumberOfResolutions']=['2']
            pm['Metric']=['NormalizedMutualInformation']
            pm['ShowExactMetricValues']=["false","false","false"]
            pm['RequiredRatioOfValidSamples']=['0.25']
            pm['NumberOfHistogramBins']=['32']
            pm['NumberOfFixedHistogramBins']=['32']
            pm['NumberOfMovingHistogramBins']=['32']

            pm['MovingKernelBSplineOrder']=['3','3','3']
            pm['NewSamplesEveryIteration']=['true']
            pm['MaximumNumberOfIterations']=['2000']
            pm['MaximumNumberOfSamplingAttempts']=['0','0','0']
            pm['AutomaticParameterEstimation']=['true']
            pm['SigmoidInitialTime']=['0','0','0']
            pm['UseAdaptiveStepSizes']=['true']
            pm['NumberOfSamplesForExactGradient']=['100000']
            pm['Interpolator']=['BSplineInterpolator']
            pm['BSplineInterpolationOrder']=['1','1','1']
            pm['ResampleInterpolator']=['FinalBSplineInterpolator']

            pm['FinalBSplineInterpolationOrder']=['3']
            pm['ImageSampler']=['Grid']
            pm['UseRandomSampleRegion']=['false']
            pm['FixedImageBSplineInterpolationOrder']=['1','1','1']
            pm['FixedImagePyramid']=['FixedShrinkingImagePyramid']
            pm['FixedImagePyramidSchedule']=['2','2','2','1','1','1']
            pm['WritePyramidImagesAfterEachResolution']=['false']

            pm['Resampler']=['DefaultResampler']

            elastixImageFilter.SetParameterMap(pm)
            elastixImageFilter.Execute()
            #     seg_n_a=sitk.Cast(seg_n, sitk.sitkInt8)
            # sitk.WriteImage(seg_n_a,seg_n_p[0])
                #seg_n_a=sitk.ReadImage(seg_n_p[0])
            try:
                moved = sitk.Transformix(seg_n_a,elastixImageFilter.GetTransformParameterMap())
            except Exception:
                a=1
                continue
            #t=sitk.Cast(moved, sitk.sitkInt8)
        #sitk.WriteImage(moved,seg_n_p[0].replace('seg.1','regseg.1'))
        #sitk.WriteImage(elastixImageFilter.GetResultImage(), seg_n_p[0].replace('seg.1', 'regseg.1').replace('seg.nrrd','nrrd'))
        #sitk.WriteImage(move,
        #                seg_n_p[0].replace('seg.1', 'regseg.1').replace('seg.nrrd', 'raw.nrrd'))
        else:
            R.SetInitialTransform(
                        sitk.CenteredTransformInitializer(
                            move,
                            fix,
                            sitk.AffineTransform(3),
                            sitk.CenteredTransformInitializerFilter.MOMENTS,
                        )
                    )
            outTx = R.Execute(move, fix)
            interpolator = sitk.sitkBSpline
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolator)
            resampler.SetDefaultPixelValue(0)
            resampler.SetReferenceImage(fix)
            resampler.SetTransform(outTx)
            moved=resampler.Execute(seg_n_a)

        C.append((sitk.GetArrayFromImage(moved)>0.5).astype(np.uint8))
        rs,s=get_matched_segs(fix,seg_n_a)
        CB.append(s)
        
    def dice(a,b):
        return (np.sum(a*b==1)*2.0+1e-5)/(np.sum(a==1)+np.sum(b==1)*1.0)
    try:
        #print(dice(C[0],C[1]),dice(C[0],C[2]))
        dd.append(np.mean([dice(C[0],C[1]),dice(C[0],C[2])]))
        old.append(np.mean([dice(C[0], CB[0]), dice(C[0], CB[1])]))
        print('new',([dice(C[0],C[1]),dice(C[0],C[2])]),'old',([dice(C[0], CB[0]), dice(C[0], CB[1])]))
        xx=1
    except:
        continue
print(np.mean(dd))
print(np.mean(old))

        