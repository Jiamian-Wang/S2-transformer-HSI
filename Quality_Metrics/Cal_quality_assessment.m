
clear;clc;



gt_path = '' % load the 10 testing data

res_path = './recon_sig.mat'; % load the reconstruction results by the test.py, i.e., save the mat result as recon_sig.mat
load(res_path);

Test_file       =    {'scene01','scene02','scene03','scene04','scene05','scene06','scene07','scene08','scene09','scene10'};

psnr_total=0.0;
ssim_total=0.0;
psnr_list = zeros(10,1);
ssim_list = zeros(10,1);
for i=1:10
    S = load(strcat(gt_path, Test_file{i},'.mat'));
    S = S.img;
    
    Z = squeeze(recon_sig(i,:,:,:));

    Z = double(Z);
    
    Z(Z>1.0) = 1.0;  
    Z(Z<0.0) = 0.0;
    
    [psnr, rmse, ergas, sam, uiqi, ssim] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z)), 0, 1);
    
    pred(1,i) = psnr;
    pred(2,i) = rmse;
    pred(3,i) = ergas;
    pred(4,i) = sam;
    pred(5,i) = uiqi;
    pred(6,i) = ssim;
    psnr_list(i,1) = psnr;
    ssim_list(i,1) = ssim;
    
    psnr_total = psnr_total+psnr;
    ssim_total = ssim_total+ssim;
end
psnr = mean(psnr_list);
ssim = mean(ssim_list);
fprintf('The PNSR=%f\n',psnr);
fprintf('The SSIM=%f\n',ssim);

