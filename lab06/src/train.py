"""
Minimal training script for DDPM on the iCLEVR dataset (multi-hot conditioning).

Usage examples:
  python train.py --data_dir ./iclevr --out_dir ./checkpoints --batch_size 8 --epochs 10 --image_size 128

Notes:
 - Expects the dataset layout: <data_dir>/images, <data_dir>/train.json, <data_dir>/objects.json
 - Saves checkpoints in out_dir and periodic sample images for inspection.
"""

import os
import argparse
from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import numpy as np
try:
    import wandb
except Exception:
    wandb = None

from iclevr.dataloader import make_dataloader, IclevrDataset
from model import UNet, DiffusionDDPM
from diffusers import DDPMScheduler
from inference import make_multihot_batch, load_json_list, load_objects, sample_conditional_batch
from evaluator import evaluation_model


def sample_conditional(model: nn.Module, scheduler, cond: torch.Tensor, shape: tuple, device: torch.device, progress: bool = False):
    """
    Run reverse diffusion conditioned on `cond` (B, cond_dim).
    - model: UNet expecting (x, timesteps, cond)
    - scheduler: diffusers scheduler with .step API
    - cond: (B, cond_dim)
    - shape: (B, C, H, W)

    Returns tensor (B, C, H, W) in device.
    """
    model = model.to(device)
    model.eval()
    num_timesteps = scheduler.config.num_train_timesteps if hasattr(scheduler, 'config') else scheduler.num_train_timesteps

    with torch.no_grad():
        sample = torch.randn(shape, device=device)
        B = shape[0]
        for t in range(num_timesteps - 1, -1, -1):
            ts = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = model(sample, ts, cond)
            step_out = scheduler.step(pred_noise, t, sample)
            if hasattr(step_out, 'prev_sample'):
                prev = step_out.prev_sample
            elif isinstance(step_out, dict) and 'prev_sample' in step_out:
                prev = step_out['prev_sample']
            else:
                prev = step_out
            sample = prev
    model.train()
    return sample


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(str(tmp), str(path))


def capture_denoise_sequence(unet, scheduler, cond: torch.Tensor, shape: tuple, device: torch.device, num_steps: int = 100):
    """
    Run the reverse denoising loop but record `num_steps` evenly spaced intermediate samples.
    Returns tensor of shape (num_steps, C, H, W) with values in [-1,1] on CPU.
    """
    unet = unet.to(device)
    unet.eval()
    total_T = scheduler.config.num_train_timesteps if hasattr(scheduler, 'config') else scheduler.num_train_timesteps
    # choose timesteps to log (integers) evenly across T: from T-1 down to 0
    ts_to_log = set([int(x) for x in np.linspace(total_T - 1, 0, num_steps)])
    with torch.no_grad():
        sample = torch.randn(shape, device=device)
        B = shape[0]
        recorded = []
        for t in range(total_T - 1, -1, -1):
            ts = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = unet(sample, ts, cond)
            step_out = scheduler.step(pred_noise, t, sample)
            if hasattr(step_out, 'prev_sample'):
                prev = step_out.prev_sample
            elif isinstance(step_out, dict) and 'prev_sample' in step_out:
                prev = step_out['prev_sample']
            else:
                prev = step_out
            sample = prev
            if t in ts_to_log:
                # record the first item in batch (or whole batch if you want)
                recorded.append(sample.detach().cpu().clone())
    unet.train()
    # recorded is list of tensors (B,C,H,W). Stack and return only the first sample across time
    recorded = torch.stack(recorded, dim=0)  # (S, B, C, H, W)
    # return first element in batch across time
    return recorded[:, 0]


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # prepare dataset
    ds = IclevrDataset(args.data_dir, split_file=args.split_file, image_size=args.image_size)
    dataloader = make_dataloader(args.data_dir, split_file=args.split_file, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers, image_size=args.image_size)

    num_objects = ds.num_objects
    tqdm.write(f"Dataset size: {len(ds)}, num_objects: {num_objects}")

    # build model
    unet = UNet(image_channels=3, base_channels=args.base_channels,
                channel_mults=args.channel_mults, num_res_blocks=args.num_res_blocks,
                time_emb_factor=args.time_emb_factor, num_heads=args.num_heads,
                attn_at=args.attn_at, dropout=args.dropout, cond_dim=num_objects)
    unet = unet.to(device)

    # build scheduler from CLI choice and pass into DiffusionDDPM so experiments can vary beta schedule
    scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps, beta_schedule=args.beta_schedule)
    ddpm = DiffusionDDPM(unet, num_train_timesteps=args.num_timesteps, device=str(device), scheduler=scheduler)
    # ddpm.scheduler will point to the scheduler we created

    # load object map for building multi-hot conditions for sampling evaluation
    try:
        obj_map = load_objects(os.path.join(args.data_dir, 'objects.json'))
    except Exception:
        obj_map = None

    # initialize wandb if requested
    if args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        # log beta schedule once as histogram for visualization
        try:
            betas = ddpm.scheduler.betas.cpu().numpy()
            wandb.log({"beta_schedule_hist": wandb.Histogram(betas)}, commit=False)
            # also log a small sampled sequence of betas for reference
            wandb.log({"beta_schedule_values": betas.tolist()}, commit=False)
        except Exception:
            pass
        # initialize evaluator on same device for computing accuracy during evaluation
        try:
            evaluator = evaluation_model(device=str(device))
        except Exception:
            evaluator = None
    else:
        evaluator = None

    optimizer = Adam(unet.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start = time.time()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            imgs = batch['image'].to(device)
            cond = batch['cond'].to(device)
            B = imgs.shape[0]

            timesteps = torch.randint(0, ddpm.num_train_timesteps, (B,), device=device).long()
            noise = torch.randn_like(imgs)
            noisy = ddpm.scheduler.add_noise(imgs, noise, timesteps)

            pred = unet(noisy, timesteps, cond)
            loss = loss_fn(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            global_step += 1

            # no per-step logging to avoid breaking tqdm
            pass

        elapsed = time.time() - start
        # no epoch loss print to keep training output clean
        tqdm.write(f"Epoch {epoch} done. time: {elapsed:.1f}s")

        # Sampling every sample_epoch_interval epochs
        if (epoch + 1) % args.sample_epoch_interval == 0:
            # use first batch from dataset for conditioning
            try:
                sample_batch = next(iter(dataloader))
                cond_vis = sample_batch['cond'].to(device)
                n_sample = min(4, cond_vis.shape[0])
                cond_vis = cond_vis[:n_sample]
                samples = sample_conditional(unet, scheduler, cond_vis, (n_sample, 3, args.sample_size, args.sample_size), device)
                samples = (samples.clamp(-1, 1) + 1.0) / 2.0
                save_image(samples, out_dir / f"sample_epoch{epoch+1}.png", nrow=n_sample)
            except Exception:
                # do not break training if sampling fails
                pass

        # periodic step checkpoint (atomic)
        if args.save_every > 0 and global_step % args.save_every == 0:
            ckpt = {
                'model_state_dict': unet.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'step': global_step,
                'epoch': epoch,
            }
            # include scheduler state if available
            try:
                if hasattr(scheduler, 'state_dict'):
                    ckpt['scheduler_state'] = scheduler.state_dict()
            except Exception:
                pass
            _atomic_save(ckpt, out_dir / f"ckpt_{global_step}.pt")

        # Sampling every sample_epoch_interval epochs
        if (epoch + 1) % args.sample_epoch_interval == 0:
            # use the inference utilities to sample and log to wandb
            try:
                for target_json in ['test.json', 'new_test.json']:
                    json_path = os.path.join(args.data_dir, target_json)
                    lists = load_json_list(json_path)
                    # pick first N conditions (or pad if fewer)
                    Bsample = min(len(lists), args.sample_batch_size)
                    if Bsample == 0:
                        continue
                    batch_lists = lists[:Bsample]
                    cond_batch = make_multihot_batch(batch_lists, obj_map, device)

                    # sample final images using the inference sampling function
                    samples = sample_conditional_batch(unet, ddpm.scheduler, cond_batch, (Bsample, 3, args.sample_size, args.sample_size), device)
                    samples_vis = (samples.clamp(-1, 1) + 1.0) / 2.0

                    # save grid locally
                    grid = make_grid(samples_vis, nrow=min(8, Bsample))
                    save_image(grid, out_dir / f"sample_epoch{epoch+1}_{target_json}.png")

                    # log to wandb
                    if args.use_wandb and wandb is not None:
                        # convert grid to numpy HWC [0,255]
                        grid_np = (grid.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
                        wandb.log({f"samples_{target_json}": wandb.Image(grid_np, caption=f"epoch_{epoch+1}")}, step=global_step)

                    # compute and log accuracy using evaluator if available
                    if evaluator is not None:
                        try:
                            # evaluator expects images in [-1,1] on evaluator device
                            eval_device = next(evaluator.resnet18.parameters()).device
                            imgs_for_eval = samples.clamp(-1, 1).to(eval_device)
                            # build labels on CPU
                            labels_cpu = make_multihot_batch(batch_lists, obj_map, torch.device('cpu')).cpu()
                            acc = evaluator.eval(imgs_for_eval, labels_cpu)
                            tqdm.write(f"Eval {target_json} accuracy: {acc:.4f}")
                            if args.use_wandb and wandb is not None:
                                wandb.log({f"acc_{target_json}": float(acc)}, step=global_step)
                        except Exception as e:
                            tqdm.write(f"Evaluator failed for {target_json}: {e}")

                # --- capture denoising process for a fixed label set and save as single grid ---
                try:
                    # explicit label set requested by user
                    label_set = ["red sphere", "cyan cylinder", "cyan cube"]
                    if obj_map is None:
                        obj_map = load_objects(os.path.join(args.data_dir, 'objects.json'))
                    cond_label = make_multihot_batch([label_set], obj_map, device)
                    # capture 10 frames evenly across the entire denoising T
                    denoise_steps = 10
                    denoise_seq = capture_denoise_sequence(unet, ddpm.scheduler, cond_label, (1, 3, args.sample_size, args.sample_size), device, num_steps=denoise_steps)
                    # denoise_seq: (S, C, H, W) in [-1,1]
                    denoise_vis = (denoise_seq.clamp(-1, 1) + 1.0) / 2.0
                    # make a single-row grid with S columns
                    denoise_grid = make_grid(denoise_vis, nrow=denoise_steps)
                    out_name = out_dir / f"denoise_{label_set[0].replace(' ','_')}_{label_set[1].replace(' ','_')}_{label_set[2].replace(' ','_')}_epoch{epoch+1}.png"
                    save_image(denoise_grid, out_name)
                    if args.use_wandb and wandb is not None:
                        grid_np = (denoise_grid.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
                        wandb.log({"denoise_process": wandb.Image(grid_np, caption=f"labels={label_set}, epoch={epoch+1}")}, step=global_step)
                except Exception as e:
                    tqdm.write(f"Denoise capture failed: {e}")
            except Exception as e:
                tqdm.write(f"Sampling/logging failed at epoch {epoch+1}: {e}")

        # save per-epoch checkpoint if requested
        if args.save_epoch_interval > 0 and (epoch + 1) % args.save_epoch_interval == 0:
            epoch_ckpt = {
                'model_state_dict': unet.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'step': global_step,
                'epoch': epoch + 1,
            }
            try:
                if hasattr(scheduler, 'state_dict'):
                    epoch_ckpt['scheduler_state'] = scheduler.state_dict()
            except Exception:
                pass
            target_path = out_dir / f"ckpt_epoch{epoch+1}.pt"
            try:
                _atomic_save(epoch_ckpt, target_path)
                tqdm.write(f"Saved epoch checkpoint: {target_path}")
            except Exception as e:
                tqdm.write(f"Failed to save epoch checkpoint {target_path}: {e}")

        # wandb logging
        if args.use_wandb and wandb is not None:
            wandb.log({"epoch": epoch, "loss": epoch_loss / len(dataloader)}, step=global_step)

    # final save (atomic)
    final_ckpt = {'model_state_dict': unet.state_dict(), 'step': global_step, 'epoch': args.epochs}
    try:
        if hasattr(scheduler, 'state_dict'):
            final_ckpt['scheduler_state'] = scheduler.state_dict()
    except Exception:
        pass
    _atomic_save(final_ckpt, out_dir / 'final_model.pt')
    tqdm.write(f"Training finished. final step: {global_step}. Model saved to {out_dir / 'final_model.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='iclevr')
    parser.add_argument('--split_file', type=str, default='train.json')
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--sample_size', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channel_mults', type=lambda s: [int(x) for x in s.split(',')], default=[1,2,4,8])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--time_emb_factor', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--attn_at', type=lambda s: [int(x) for x in s.split(',')], default=[2])
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--save_epoch_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--sample_epoch_interval', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', help='Log metrics and images to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='dlp_lab6')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--log_denoise_steps', type=int, default=100, help='Number of intermediate denoising steps to log')
    parser.add_argument('--sample_batch_size', type=int, default=32, help='Number of samples to generate for logging')
    # Supported beta schedules (passed to diffusers.DDPMScheduler):
    #  - 'linear'             : linear beta schedule (default)
    #  - 'scaled_linear'      : scaled linear schedule
    #  - 'squaredcos_cap_v2' : cosine-based schedule (often better quality)
    SUPPORTED_BETA_SCHEDULES = ['linear', 'scaled_linear', 'squaredcos_cap_v2']
    parser.add_argument('--beta_schedule', type=str, choices=SUPPORTED_BETA_SCHEDULES, default='linear',
                        help='Beta schedule for DDPMScheduler. Supported: ' + ', '.join(SUPPORTED_BETA_SCHEDULES))

    args = parser.parse_args()

    # normalize args lists
    if isinstance(args.channel_mults, str):
        args.channel_mults = [int(x) for x in args.channel_mults.split(',')]
    if isinstance(args.attn_at, str):
        args.attn_at = [int(x) for x in args.attn_at.split(',')]

    train(args)
