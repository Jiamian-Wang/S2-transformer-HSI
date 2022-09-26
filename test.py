


import os
import time
import torch
import scipy
import datetime
import argparse
from utils_spectral import *
from network_backbone_channelattn import SpaSpe_Attn_Net


parser = argparse.ArgumentParser()
# GPU setting #################################################
parser.add_argument('--device', default='0,1', help='CUDA ID')
# directories #################################################
parser.add_argument('--mask_path',                 default="./Data",  help="mask directory")
parser.add_argument('--test_data_path',            default="./Data/testing/simu/mat/",       help="testing data directory")
parser.add_argument('--model_dir',                 default="S2_transformer",                       help="checkpoint directory")
parser.add_argument('--recon_dir',                 default="S2_transformer",                       help="reconstruction results directory")
parser.add_argument("--batch_size",      type=int, default=1,      help="batch size")

# optimization #################################################
parser.add_argument("--last_train",      type=int,    default=255,    help='specify the checkpoint to be load for breakpoint training/testing')
parser.add_argument("--in_channels",     type=int,    default=28,     help='number of the spectral channels in the hyperspectral data')
parser.add_argument("--img_size",        type=int,    default=256,    help='spatial size (height/width) of the input video frames')
# network #################################################
parser.add_argument('--meas_init',                default="meas*mask",choices=['meas*mask', 'meas'], help="if initialzing the measurement with the mask")
parser.add_argument('--attention_type',           default="Parall_CatSS_Attn", choices=['Spa_Attn', 'Spe_Attn', 'SpaSpe_Attn', 'SpeSpa_Attn', 'SpaSpa_Attn', 'SpeSpe_Attn', 'Parall_SS_Attn', 'Parall_CatSS_Attn'], help="four attention types in a transformer layer")
parser.add_argument("--upscale",      type=int,   default=1,          help='spatial upscale factor for image SR, i.e., 2,3,4,8. For our task, no upscale needed')
parser.add_argument("--window_size",  type=int,   default=8,          help='window size in the swin transformer')
parser.add_argument("--img_range",    type=float, default=1.,         help='input image pixel value range')
parser.add_argument("--embed_dim",    type=int,   default=60,         help='patch embedding dimension')
parser.add_argument("--depths",       type=int,   default=[6,6,6,6],  nargs='+', help='specify the number of RSTBs and the number of STL layers in each RSTB. *RSTB: Residual Swin Transformer Block')
parser.add_argument("--num_heads",    type=int,   default=[6,6,6,6],  nargs='+', help='specify the number of heads in STLs, STLs in the same RSTB use the same #heads. Note number of head should be divided by embed_dim. *STL: Swin transformer layer')
parser.add_argument("--mlp_ratio",    type=int,   default=4,          help='ratio of mlp hidden dim to embedding dim ')
#################################################
args = parser.parse_args()


print(args.device)
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')




mask3d_batch = generate_masks(args.mask_path, args.batch_size)

test_data = LoadTest(args.test_data_path)


S2_transformer = SpaSpe_Attn_Net(upscale=args.upscale,
               in_chans=args.in_channels,
               img_size=args.img_size,
               window_size=args.window_size,
               img_range=args.img_range,
               depths=args.depths,
               embed_dim=args.embed_dim,
               num_heads=args.num_heads,
               mlp_ratio=args.mlp_ratio,
               upsampler='',
               resi_connection='1conv',
               attention_type=args.attention_type).cuda()

S2_transformer = torch.nn.DataParallel(S2_transformer)



if args.last_train != 0:
    checkpoint_path = './'+args.model_dir + '/model_epoch_{}.pth'.format(args.last_train)
    checkpoint = torch.load(checkpoint_path)
    S2_transformer.load_state_dict(checkpoint['model_weights'])
    args.learning_rate = checkpoint['lr_last_record'][0]
    print('>>>>>>load lr:',args.learning_rate, type(args.learning_rate))
    print('------Successfully load the pre-trained model!------')



def gen_meas(data_batch, mask3d_batch, meas_init='meas*mask', is_training=True):
    nC = data_batch.shape[1]
    if is_training is False:
        [batch_size, nC, H, W] = data_batch.shape
        mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()
    temp = shift(mask3d_batch*data_batch, 2)
    meas = torch.sum(temp, 1)/nC*2
    y_temp = shift_back(meas)
    PhiTy = torch.mul(y_temp, mask3d_batch)
    if meas_init == 'meas*mask':
        return PhiTy
    elif meas_init == 'meas':
        return y_temp



def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2

def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, data_range=None):

    im_true, im_test = _as_floats(im_true, im_test)
    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)


def test():
    psnr_list_tsa, psnr_list_gsm, ssim_list = [], [], []
    test_gt = test_data.cuda().float()
    test_PhiTy = gen_meas(test_gt, mask3d_batch, args.meas_init, is_training = False)
    S2_transformer.eval()
    begin = time.time()
    with torch.no_grad():
        model_out = S2_transformer(test_PhiTy)
    end = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val_tsa = torch_psnr(model_out[k,:,:,:], test_gt[k,:,:,:])
        psnr_list_tsa.append(psnr_val_tsa.detach().cpu().numpy())
        psnr_val_gsm = compare_psnr(test_gt[k, :, :, :].cpu().numpy(), model_out[k, :, :, :].cpu().numpy(),data_range=1.0)
        psnr_list_gsm.append(psnr_val_gsm)

        ssim_val = torch_ssim(model_out[k,:,:,:], test_gt[k,:,:,:])
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean_tsa = np.mean(np.asarray(psnr_list_tsa))
    psnr_mean_gsm = np.mean(np.asarray(psnr_list_gsm))

    ssim_mean = np.mean(np.asarray(ssim_list))
    # print('===>testing psnr = {:.5f}(tsa)/{:.5f}(gsm), ssim = {:.5f}, time: {:.5f}mins'.format(psnr_mean_tsa, psnr_mean_gsm, ssim_mean, (end - begin)/60.))

    # please save the <pred> to the local directory for the further metric evaluation. 


    return (pred, truth, psnr_list_tsa, ssim_list, psnr_mean_tsa, ssim_mean)


     
def main():

    (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test()


if __name__ == '__main__':
    main()
    

