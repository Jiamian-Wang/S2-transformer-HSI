# S2-transformer-HSI

This repository contains the network implementation, testing, and evaluation code of the novel S^2-transformer network for hyperspectral image reconstruction. 

	- For simulation data, we use full reference image quality assessments (full-ref IQA): PSNR and SSIM;
	- For real data, we use no reference image quality assessments (no-ref IQA): Naturalness Image Quality Evaluator (NIQE);

## Data and pre-trained models

1. The quantitative comparison is conducted upon simulation data. In this project, we emplot the benchmark testing dataset, which contains [ten ground truth testing hyperspectral images](https://ufile.io/eafy9n2a).  

2. One 2D [real mask](https://ufile.io/2zbztqxm) is employed for the simulation data reconstruction. 

3. Pre-trained model ([model_epoch_255.pth](https://ufile.io/bmxw9qdn)) is provided for reproducing the simulation reconstruction results. 

4. We also provide the [simulation reconstruction results](https://ufile.io/eznqmlx9) by the above pre-trained model. The data are saved in the `.mat` file and could be employed for the metric computation. 

5. On the other hand, we provide the the real hyperspectral reconstruction results ([ours_real_79.mat](https://ufile.io/i2vpftyc)) upon the practical measurements. We further provide the following real reconstruction results for a better comparasion:
	
	- The real reconstruction results ([ours_real_81](https://ufile.io/z2ivtbh8)) by the proposed method **without the mask-aware learning strategy**.
	- The real reconstruction results ([MST.mat](https://ufile.io/ybwhb4ig)) by MST.
	- The real reconstruction results ([HDNet.mat](https://ufile.io/6fyxhlsq)) by HDNet. 


## Quick Start

1. For simulation data: 

	- Specify `device` as GPU id(s), `test_data_path` as directory of the downloaded test data. `model_dir` as the directory of the pre-trained model, by default, is pre-defined as `S2_transformer`. `mask_path` as the directory of the mask. 
	
	- For example, run 
	
		`python test.py --device 0,1 --test_data_path ./your_test_data/ --model_dir S2_transformer --mask_path ./your_mask_dir/ ` 
	
	- Please save the `pred` to the local directory if desired. 

	- Use the **Cal_quality_assessment.m** to compute the PSNR and SSIM. Please load the ground_truth data and the reconstruction results accordingly. 
	
2. For real data: 

	- Use the **realeval_noref_NIQE.m** to compute the NIQE score on real reconstructions. Please load the reconstruction results accordingly. 

