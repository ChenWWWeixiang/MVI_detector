import SimpleITK as sitk
import glob,os
root='/mnt/data1/mvi2/nrrd_set'
output='/mnt/data1/mvi2/nrrd_set2'
os.makedirs(output,exist_ok=True)
check_root='/mnt/data1/mvi2/data2019'
truliste=os.listdir(check_root+'/MVI')
all_fix=glob.glob(os.path.join(root,'t1_PRE_*.nrrd'))
replaclist=['dwi_ADC','t2','t1_A','t1_V','t1_POST']
elastixImageFilter = sitk.ElastixImageFilter()
for item in all_fix:
    name =item.split('/')[-1].split('_')[-1].split('.')[0]
    cls=name in truliste
    try:
        elastixImageFilter.SetFixedImage(sitk.ReadImage(item))
        for re_mod in replaclist:
                mv1=item.replace('t1_PRE',re_mod)
                elastixImageFilter.SetMovingImage(sitk.ReadImage(mv1))
                elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
                elastixImageFilter.Execute()
                sitk.WriteImage(elastixImageFilter.GetResultImage(),os.path.join(output,str(int(cls)))+'_'+name+'_'+re_mod+'.nrrd')
        sitk.WriteImage(sitk.ReadImage(item),
                        os.path.join(output, str(int(cls))) + '_' + name + '_t1_PRE.nrrd')

    except:
        continue