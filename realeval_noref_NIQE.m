% No-reference image quality assessment
% use Naturalness Image Quality Evaluator (NIQE)
% helper: Score = NIQE(A), A is 2D image, lower reflect better perceptual quality of image A with respect to the input model.

%% reference
% niqe: https://www.mathworks.com/help/images/ref/niqe.html#d123e221507
% image quality assessment summary: https://www.mathworks.com/help/images/image-quality-metrics.html
%% load real reconstructions results 
mst = load('./MST.mat') % mst
hdnet = load('./HDNet.mat') % hdnet
ours = load('./ours_real_81.mat') % ours

pred_mst = mst.pred
pred_hdnet = hdnet.pred
pred_ours = ours.pred;


%% compute the NIQE score channel wise

niqe_mst = 0;
niqe_hdnet = 0;
niqe_ours = 0;


for img=1:5
    
    pred_ours_img = fliplr(flipud(squeeze(pred_ours(img,:,:,:))));
    pred_hdnet_img = fliplr(flipud(squeeze(pred_hdnet(img,:,:,:))));
    pred_mst_img   = fliplr(flipud(squeeze(pred_mst(img,:,:,:))));

for chl=1:28
    
    niqe_mst  = niqe_mst   + niqe(pred_mst_img(:,:,chl))
    niqe_hdnet= niqe_hdnet + niqe(pred_hdnet_img(:,:,chl))
    niqe_ours = niqe_ours  + niqe(pred_ours_img(:,:,chl))

 
end

end

niqe_mst = niqe_mst/ (5*28); 
niqe_hdnet = niqe_hdnet/ (5*28); 
niqe_ours = niqe_ours/ (5*28); 



