import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log, log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size, latent_dim=12):
    # logvar = torch.clamp(logvar, min=-10.0, max=10.0)   
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size * latent_dim
    return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.args = args
        self.current_epoch = current_epoch
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        
        

    def update(self):
        self.current_epoch += 1
        return self.current_epoch

    def get_beta(self):
        if(self.kl_anneal_type == 'Cyclical'):
            return self.frange_cycle_linear(self.current_epoch, start=0.0, stop=1.0, n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio)
        elif(self.kl_anneal_type == 'Monotonic'):
            # print(f"Monotonic KL Annealing: current_epoch={self.current_epoch}, kl_anneal_type={self.kl_anneal_type}")
            if(self.current_epoch < self.kl_anneal_cycle):
                return self.frange_cycle_linear(self.current_epoch, start=0.0, stop=1.0, n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio)  
            else:
                return 1.0
        elif(self.kl_anneal_type == 'None'):
            return 1
        else:
            raise NotImplementedError(f"Unknown kl_anneal_type: {self.kl_anneal_type}")

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        """Linear annealing with cyclical behavior"""
        x = (n_iter % n_cycle) / n_cycle
        if x <= ratio:
            return start + (stop - start) * (x / ratio)
        else:
            return stop


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)

        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.args.num_epoch, eta_min=1e-6)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss(reduction='sum')
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        # Weights & Biases logger
        import wandb
        wandb.init(project="Lab4_Dance_VAE", config=vars(args))
        self.wandb = wandb

    def forward(self, curr_frame, prev_frame, curr_label):
        # Encode previous frame and label
        prev_feat = self.frame_transformation(prev_frame).detach()      # [batch, F_dim]
        curr_feat = self.frame_transformation(curr_frame)                # [batch, F_dim]
        label_feat = self.label_transformation(curr_label)      # [batch, L_dim]

        # Concatenate features
        # enc_feat = torch.cat([curr_feat, label_feat], dim=1)   # [batch, F_dim + L_dim]

        # Posterior prediction
        z, mu, logvar = self.Gaussian_Predictor.forward(curr_feat, label_feat)         # [batch, N_dim] each

        # Decoder fusion
        # dec_feat = torch.cat([prev_feat, label_feat, z], dim=1) # [batch, F_dim + L_dim + N_dim]
        fusion = self.Decoder_Fusion.forward(prev_feat, label_feat, z)                  # [batch, D_out_dim]

        # Generate frame
        gen_frame = self.Generator.forward(fusion)                   # [batch, C, H, W]

        return gen_frame, mu, logvar

    def training_stage(self):
        for i in range(1, self.args.num_epoch + 1):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            epoch_loss = 0

            for batch_idx, (img, label) in enumerate(pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, beta = self.training_one_step(img, label, adapt_TeacherForcing)
                epoch_loss += loss.item()

                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            # wandb log per epoch
            self.wandb.log({
                'Loss/train_epoch': epoch_loss / len(train_loader),
                'TeacherForcingRatio/epoch': self.tfr,
                'Beta/epoch': self.kl_annealing.get_beta(),
            }, step=self.current_epoch)

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        val_losses = []
        val_psnrs = []
        for (img, label) in (pbar := tqdm(val_loader, ncols=120, desc=f"Validation Epoch {self.current_epoch}")):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, avg_psnr = self.val_one_step(img, label)
            val_losses.append(loss.item())
            val_psnrs.append(avg_psnr)
            pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{self.scheduler.get_last_lr()[0]:.6f}", psnr=f"{avg_psnr:.4f}")
        self.wandb.log({
            'Loss/val_epoch': np.mean(val_losses),
            'PSNR/val_epoch': np.mean(val_psnrs),
            'lr': self.scheduler.get_last_lr()[0],
        }, step=self.current_epoch)
        print(f"(val) Epoch {self.current_epoch} completed")

    def training_one_step(self, img, label, adapt_TeacherForcing):
        self.train()
        batch_size, seq_len, C, H, W = img.size()
        loss_total = 0

        prev_frame = img[:, 0]  # initial previous frame
        for t in range(1, seq_len):
            curr_frame = img[:, t]
            curr_label = label[:, t]

            # Teacher Forcing: use ground truth previous frame, else use generated
            if adapt_TeacherForcing or t == 1:
                input_prev = prev_frame
            else:
                input_prev = gen_frame.detach()  # use last generated frame

            # Forward pass: expects current frame and previous frame
            out = self.forward(curr_frame, input_prev, curr_label)
            gen_frame, mu, logvar = out  # adjust if your forward returns differently
            
            # NaN prevention for outputs
            gen_frame = torch.clamp(gen_frame, 0, 1)
            # ...existing code...
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)

            # Reconstruction loss
            recon_loss = self.mse_criterion(gen_frame, curr_frame) / batch_size
            # KL loss
            kl_loss = kl_criterion(mu, logvar, batch_size)
            # Annealing
            beta = self.kl_annealing.get_beta()
            loss = recon_loss + beta * kl_loss

            loss_total += loss

            prev_frame = curr_frame if adapt_TeacherForcing else gen_frame

        # loss_total /= (batch_size)  # average loss over batch and sequence length
        self.optim.zero_grad()
        loss_total.backward()
        self.optimizer_step()
        return loss_total, beta

    
    @torch.no_grad()
    def val_one_step(self, img, label):
        super().eval()
        batch_size, seq_len, C, H, W = img.size()
        loss_total = 0
        psnr_list = []

        prev_frame = img[:, 0]
        for t in range(1, seq_len):
            curr_frame = img[:, t]
            curr_label = label[:, t]

            input_prev = prev_frame if t == 1 else gen_frame.detach()
            gen_frame, mu, logvar = self.forward(curr_frame, input_prev, curr_label)
            gen_frame = torch.clamp(gen_frame, 0, 1)

            recon_loss = self.mse_criterion(gen_frame, curr_frame) / batch_size
            kl_loss = kl_criterion(mu, logvar, batch_size)
            beta = self.kl_annealing.get_beta()
            loss = recon_loss + beta * kl_loss

            loss_total += loss

            psnr = Generate_PSNR(gen_frame, curr_frame)
            psnr_list.append(psnr.item())
            prev_frame = gen_frame
        # loss_total /= (batch_size)
        avg_psnr = np.mean(psnr_list)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(psnr_list, label='PSNR per-frame')
        plt.xlabel('Frame')
        plt.ylabel('PSNR')
        plt.title(f'Validation PSNR per-frame (Epoch {self.current_epoch})')
        plt.legend()
        plt.tight_layout()
        # save the plot to a temporary file
        fig_path = f'val_psnr_epoch{self.current_epoch}.png'
        plt.savefig(fig_path)
        plt.close()
        self.wandb.log({'PSNR/val_seq': avg_psnr, 'PSNR/val_frame_plot': self.wandb.Image(fig_path)}, step=self.current_epoch)
        # Optional: delete temporary file

        import os
        if os.path.exists(fig_path):
            os.remove(fig_path)

        return loss_total, avg_psnr
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0.0, self.tfr*self.tfr_d_step)

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.6f}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.args.num_epoch, eta_min=1e-6)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 5.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.95,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.01,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    # tensorboard 
    parser.add_argument('--log_root', type=str, default='runs', help="The path to save tensorboard logs")

    

    args = parser.parse_args()
    
    main(args)

