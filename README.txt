***********************************************************************************************************************
This is the README file for the NIPS 2022 submission:
S^2-Transformer for Mask-Aware Hyperspectral Image Reconstruction (ID:51)
***********************************************************************************************************************
Code
 |——network.py [network structure]
 |——test.py [test script]
 |——utils.py [utility functions]
 |——Quality_Metrics
		|——Cal_quality_assessment.m [matlab script to quantitatively measure the results]
		|——others [utility functions]

——————————————————————————————————————————————

We provide the testing data, mask, and pre-trained model

1) Ten testing hyperspectral images could be found at:
    [data: mat.zip]: https://ufile.io/eafy9n2a
2) mask could be found at:
    [Mask.mat]: https://ufile.io/2zbztqxm
3) Pre-trained model could be found at:
    [pretrained_model: model_epoch_255.pth]: https://ufile.io/bmxw9qdn
4) The simulation reconstruction results could be found at:
    [recon_result: recon_255.mat]: https://ufile.io/eznqmlx9
5) The real reconstruction results could be found at:
	
	our method [ours_real_79.mat]: https://ufile.io/i2vpftyc
	our method w/o mask-aware learning [ours_real_81.mat]: https://ufile.io/z2ivtbh8
	MST [MST.mat]: https://ufile.io/ybwhb4ig
	HDNet [HDNet.mat]: https://ufile.io/6fyxhlsq

——————————————————————————————————————————————
To run the code:

1. Use test.py

2. Specify:
	--device: GPU ids
	--test_data_path: directory of the downloaded test data
	--model_dir: S2_transformer by default. Accordingly, please mkdir <S2_transformer> at current directory and put the pre-trained model into the S2_transformer

3. manually save the <pred> to the local directory 

4. Use the Cal_quality_assessment.m to compute the PSNR and SSIM

	Please load the ground_truth [data] and the reconstruction results accordingly. 

5. Use the realeval_noref_NIQE.m to compute the NIQE score on real reconstructions:
	
	Please load the reconstruction results accordingly. 


——————————————————————————————————————————————

Run MST-L on our own platform

1. For the pre-trained model, please refer to  https://ufile.io/v5lqsira

2. For the reconstruction result, please refer to  https://ufile.io/d7fsugas



