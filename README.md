
# MVI_detector
A detector for MVI based on lesion segmentation network, radiomics extractor and a classifier (better description comes later).

## Segmentor: 3dunet
PyTorch implementation 3D U-Net and its variants:
- Standard 3D U-Net based on [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) 
Özgün Çiçek et al.
- Residual 3D U-Net based on [Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120.pdf) Kisuk Lee et al.
The code allows for training the U-Net for both: **semantic segmentation** (binary and multi-class) and **regression** problems (e.g. de-noising, learning deconvolutions).


## Feature extractor: radiomics
This is implemented on pyradiomics

## Classifier: MLP


