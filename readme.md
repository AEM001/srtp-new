## 新IMU数据快速开始
查看 `QUICKSTART.md` 了解新数据处理流程

## 原DIP-IMU数据集
安装依赖: https://github.com/nghorbani/human_body_prior

运行评估: `python evaluate.py`

数据预处理: `python preprocess.py`

## data download 
before running the code, you may need to download ckpt.zip and data.zip from https://drive.google.com/drive/folders/1rxYTv5j8G-10Sxy1gwmZrP-NUHJPVqyo?usp=drive_link
or from Figshare "xiao, xuan; wang, Jiangjian; Feng, Pingfa; Gong, Ao; Zhang, Xiangyu; Zhang, Jianfu (2024). Test dataset and checkpoint of the paper: Fast Human Motion Reconstruction from Sparse IMUs considering the human shape. figshare. Dataset. https://doi.org/10.6084/m9.figshare.25282732"
and unzip them to the folder ./ckpt and ./data respectively。

## data explain:

>_dip_betas.pt_: 
>>the regressed body shape parameters of DIP-IMU objects

>_imu_test.pt_: 
>>the data of preprocessed DIP_IMU raw data, 
including: 'acc' for data of accelerometer, 'ori' for data of gyroscope, 'pose' for gt values.

>SMPL_male.pkl:  the official SMPL model, which can be downloaded on: "SMPL for Python Users" on https://smpl.is.tue.mpg.de/

>_support_data_: the official SMPL model, which can be downloaded on: "DMPLS for AMASS(dmpls)" on https://smpl.is.tue.mpg.de/
> and Extended SMPL+H model(smplh) on https://mano.is.tue.mpg.de
>, release them and place them like below support_data tree.
>
>the folder tree:
>
> support_data
>>body_models
>>>dmpls
>>>
>>>smplh
>>>

**notice** 
The term "gender" used in the dataset and code refers to the same variable as the term "sex" mentioned in the manuscript.

## demo video
for demo video, please refer to https://youtu.be/9jyCHrsTVt8

## Citation
If you find the project helpful, please consider citing us:

Xiao, X., Wang, J., Feng, P. et al. Fast Human Motion reconstruction from sparse inertial measurement units considering the human shape. Nat Commun 15, 2423 (2024). https://doi.org/10.1038/s41467-024-46662-5

