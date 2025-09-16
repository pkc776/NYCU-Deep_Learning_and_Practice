import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
from utils import LoadTestData, LoadMaskData
from torch.utils.data import Dataset,DataLoader
from torchvision import utils as vutils
import os
from models import MaskGit as VQGANTransformer
import yaml
import torch.nn.functional as F

class MaskGIT:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.model.load_transformer_checkpoint(args.load_transformer_ckpt_path)
        self.model.eval()
        self.total_iter=args.total_iter
        self.mask_func=args.mask_func
        self.model.gamma = self.model.gamma_func(self.mask_func)
        self.sweet_spot=args.sweet_spot
        self.device=args.device
        self.prepare()

    @staticmethod
    def prepare():
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("mask_scheduling", exist_ok=True)
        os.makedirs("imga", exist_ok=True)

##TODO3 step1-1: total iteration decoding  
#mask_b: iteration decoding initial mask, where mask_b is true means mask
    
    def inpainting(self,image,mask_b,i): #MakGIT inference
        maska = torch.zeros(self.total_iter, 3, 16, 16, device=self.device) #save all iterations of masks in latent domain
        imga = torch.zeros(self.total_iter+1, 3, 64, 64, device=self.device)#save all iterations of decoded images
        mean = torch.tensor([0.4868, 0.4341, 0.3844],device=self.device).view(3, 1, 1)  
        std = torch.tensor([0.2620, 0.2527, 0.2543],device=self.device).view(3, 1, 1)
        ori=(image[0]*std)+mean
        imga[0]=ori #mask the first image be the ground truth of masked image

        self.model.eval()
        with torch.no_grad():
            z_indices = self.model.encode_to_z(image)  # (B=1, 256)
            z_indices_predict = z_indices.clone()
            mask_bc = mask_b.to(device=self.device)    # True = mask
            dec_img_ori = imga[0]

            orig_mask = mask_bc.clone()                     # (1, N) bool tensor
            orig_mask_count = int(orig_mask.sum().item())   # scalar

            # print(f"orig_mask_count: {orig_mask_count}")
            # iterative decoding
            for step in range(self.total_iter):
                # scheduling ratio t/T in [0,1)
                ratio = float(step+1) / float(self.total_iter)
                _mask_bc = mask_bc.clone()
                # single step inpainting
                z_indices_predict, keep_mask = self.model.inpainting(
                    z_indices_predict, _mask_bc, ratio, orig_mask_count
                )
                # update mask
                mask_bc = _mask_bc & (~keep_mask)

                # visualize mask
                mask_i = mask_bc.view(1, 16, 16)
                mask_image = torch.ones(3, 16, 16, device=self.device)
                mask_coords = torch.nonzero(mask_i[0], as_tuple=False)  # shape: (num_mask, 2)
                if mask_coords.numel() > 0:
                    mask_image[:, mask_coords[:, 0], mask_coords[:, 1]] = 0
                maska[step] = mask_image

                # decode image
                shape=(1,16,16,256)
                z_q = self.model.vqgan.codebook.embedding(z_indices_predict).view(shape)
                z_q = z_q.permute(0, 3, 1, 2)
                decoded_img=self.model.vqgan.decode(z_q)
                dec_img_ori=(decoded_img[0]*std)+mean
                imga[step+1]=dec_img_ori

                if step == self.sweet_spot:
                    break

            ##decoded image of the sweet spot / last step
            vutils.save_image(dec_img_ori.clamp(0,1), os.path.join("test_results", f"image_{i:03d}.png"), nrow=1) 

            #demo score 
            vutils.save_image(maska.cpu(), os.path.join("mask_scheduling", f"test_{i}.png"), nrow=10) 
            vutils.save_image(imga.cpu(), os.path.join("imga", f"test_{i}.png"), nrow=7)
class MaskedImage:
    def __init__(self, args):
        mi_ori=LoadTestData(root= args.test_maskedimage_path, partial=args.partial)
        self.mi_ori =  DataLoader(mi_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        mask_ori =LoadMaskData(root= args.test_mask_path, partial=args.partial)
        self.mask_ori =  DataLoader(mask_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        self.device=args.device

    def get_mask_latent(self,mask):    
        downsampled1 = torch.nn.functional.avg_pool2d(mask, kernel_size=2, stride=2)
        resized_mask = torch.nn.functional.avg_pool2d(downsampled1, kernel_size=2, stride=2)
        resized_mask[resized_mask != 1] = 0       #1,3,16*16   check use  
        mask_tokens=(resized_mask[0][0]//1).flatten()   ##[256] =16*16 token
        mask_tokens=mask_tokens.unsqueeze(0)
        mask_b = torch.zeros(mask_tokens.shape, dtype=torch.bool, device=self.device)
        mask_b |= (mask_tokens == 0) #true means mask
        return mask_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT for Inpainting")
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')#cuda
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for MaskGIT')
    
    
#TODO3 step1-2: modify the path, MVTM parameters
    parser.add_argument('--load-transformer-ckpt-path', type=str, default='ckpt/transformer.pth', help='load ckpt')

    #dataset path
    parser.add_argument('--test-maskedimage-path', type=str, default='./dataset/masked_image', help='Path to testing image dataset.')
    parser.add_argument('--test-mask-path', type=str, default='./dataset/mask64', help='Path to testing mask dataset.')
    #MVTM parameter
    parser.add_argument('--sweet-spot', type=int, default=7, help='sweet spot: the best step in total iteration')
    parser.add_argument('--total-iter', type=int, default=7, help='total step for mask scheduling')
    parser.add_argument('--mask-func', type=str, default='linear', help='mask scheduling function')

    args = parser.parse_args()

    t=MaskedImage(args)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    maskgit = MaskGIT(args, MaskGit_CONFIGS)

    i=0
    for image, mask in zip(t.mi_ori, t.mask_ori):
        image=image.to(device=args.device)
        mask=mask.to(device=args.device)
        mask_b=t.get_mask_latent(mask)       
        maskgit.inpainting(image,mask_b,i)
        i+=1
        