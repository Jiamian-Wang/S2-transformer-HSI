# S2-transformer-HSI



——————————————————————————————————————————————

We provide the testing data, mask, and pre-trained model

1. [Ten testing hyperspectral images](https://ufile.io/eafy9n2a) are employed for the metric comparison. 

2. One 2D [real mask](https://ufile.io/2zbztqxm) is employed. 

3. Pre-trained model could be found at:
    [pretrained_model: model_epoch_255.pth]: https://ufile.io/bmxw9qdn
4) The reconstruction results could be found at:
    [recon_result: recon_255.mat]: https://ufile.io/eznqmlx9

——————————————————————————————————————————————
To run the code:
1. Use test.py
2. Specify:
	--device: GPU ids
	--test_data_path: directory of the downloaded test data
	--model_dir: S2_transformer by default. Accordingly, please mkdir <S2_transformer> at current directory and put the pre-trained 	   model into the S2_transformer
3. manually save the <pred> to the local directory 
4. Use the Cal_quality_assessment.m to compute the PSNR and SSIM
	Please load the ground_truth [data] and the reconstruction results accordingly. 

