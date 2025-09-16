"""
Inference script for conditional DDPM on iCLEVR.
Reads a JSON file of object lists (like `test.json` or `new_test.json`) and `objects.json` mapping,
constructs multi-hot condition vectors, and generates images conditioned on those labels.

Usage examples:
  python inference.py --data_dir ./iclevr --json_file iclevr/test.json --checkpoint ./checkpoints/final_model.pt --out_dir ./samples --batch_size 8 --image_size 64 --num_timesteps 1000 --seed 42

If you want deterministic outputs provide --seed. You can also provide a .npy noise file with shape (N,C,H,W) via --noise_file.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from evaluator import evaluation_model
from tqdm import tqdm

from model import UNet, DiffusionDDPM


def load_objects(objects_path: str) -> dict:
    with open(objects_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_json_list(json_path: str) -> List[List[str]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Accept either list-of-lists or dict mapping filename->list
    if isinstance(data, dict):
        return list(data.values())
    return data


def make_multihot_batch(list_of_lists: List[List[str]], obj_map: dict, device: torch.device) -> torch.Tensor:
    num_objects = len(obj_map)
    B = len(list_of_lists)
    mh = torch.zeros((B, num_objects), dtype=torch.float32, device=device)
    for i, lst in enumerate(list_of_lists):
        for obj in lst:
            if obj in obj_map:
                mh[i, obj_map[obj]] = 1.0
    return mh


def sample_conditional_batch(unet: UNet, scheduler, cond: torch.Tensor, sample_shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Silent sampling of a batch conditioned on `cond` using scheduler.step API.
    cond: (B, cond_dim)
    sample_shape: (B, C, H, W)
    Returns tensor on device in range [-1,1]
    """
    unet = unet.to(device)
    unet.eval()
    num_timesteps = scheduler.config.num_train_timesteps if hasattr(scheduler, 'config') else scheduler.num_train_timesteps
    B = sample_shape[0]

    with torch.no_grad():
        if sample_shape[0] != cond.shape[0]:
            raise ValueError('Batch size mismatch between cond and sample_shape')
        sample = torch.randn(sample_shape, device=device)
        for t in range(num_timesteps - 1, -1, -1):
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
    unet.train()
    return sample


def main(args):
    device = torch.device('cuda' if (torch.cuda.is_available() and args.device == 'cuda') else 'cpu')
    data_dir = Path(args.data_dir)
    objects_path = data_dir / 'objects.json'
    if not objects_path.exists():
        raise FileNotFoundError(f'objects.json not found at {objects_path}')

    obj_map = load_objects(str(objects_path))
    inv_obj_map = {v: k for k, v in obj_map.items()}
    json_lists = load_json_list(args.json_file)

    # build model and load checkpoint
    num_objects = len(obj_map)
    unet = UNet(image_channels=3, base_channels=args.base_channels,
                channel_mults=args.channel_mults, num_res_blocks=args.num_res_blocks,
                time_emb_factor=args.time_emb_factor, num_heads=args.num_heads,
                attn_at=args.attn_at, dropout=args.dropout, cond_dim=num_objects)
    unet = unet.to(device)

    if args.checkpoint is None:
        raise ValueError('Please provide --checkpoint path to model state dict')
    ckpt = torch.load(args.checkpoint, map_location=device)
    # support both dict with 'model_state_dict' or direct state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        unet.load_state_dict(ckpt['model_state_dict'])
    else:
        unet.load_state_dict(ckpt)

    ddpm = DiffusionDDPM(unet, num_train_timesteps=args.num_timesteps, device=str(device))
    scheduler = ddpm.scheduler

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = None
    if args.evaluate:
        # initialize evaluator on same device
        evaluator = evaluation_model(device=str(device))
        total_acc = 0.0
        total_count = 0

    # optionally load provided noise file
    noise_array = None
    if args.noise_file:
        noise_array = np.load(args.noise_file)

    # make batches
    total = len(json_lists)
    batch_size = args.batch_size

    idx = 0
    while idx < total:
        batch_lists = json_lists[idx: idx + batch_size]
        B = len(batch_lists)
        cond = make_multihot_batch(batch_lists, obj_map, device)

        # determine sample shape
        sample_shape = (B, 3, args.image_size, args.image_size)

        # if a noise file provided, use its slice
        if noise_array is not None:
            # expect noise_array shape (N,C,H,W)
            if noise_array.shape[0] < B:
                raise ValueError('Provided noise file has fewer samples than batch size')
            sample_noise = torch.from_numpy(noise_array[idx: idx + B]).to(device).float()
            # override initial noise in sampling by modifying the sampling function locally
            # simple approach: set torch.manual_seed and rely on scheduler loop to start from this sample
            initial_sample = sample_noise
            # run sampling loop manually using same logic but starting from initial_sample
            unet.eval()
            with torch.no_grad():
                sample = initial_sample
                num_timesteps = scheduler.config.num_train_timesteps if hasattr(scheduler, 'config') else scheduler.num_train_timesteps
                for t in range(num_timesteps - 1, -1, -1):
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
            samples = sample
            unet.train()
        else:
            # deterministic by seed if provided
            if args.seed is not None:
                torch.manual_seed(args.seed + idx)
            samples = sample_conditional_batch(unet, scheduler, cond, sample_shape, device)

        # keep copy in [-1,1] for evaluation, denormalize only for saving
        samples_clipped = samples.clamp(-1, 1)

        # If model generates 32x32 images but we want to save/evaluate at 64x64,
        # upsample here using bicubic interpolation which usually preserves circular shapes
        # better than bilinear or nearest (reduces the chance of spheres becoming blocky/cuboid).
        if args.image_size == 32:
            target_size = (64, 64)
            samples_clipped = F.interpolate(samples_clipped, size=target_size, mode='bicubic', align_corners=False)
            # keep samples variable consistent for downstream bookkeeping
            samples = samples_clipped

        samples_save = (samples_clipped + 1.0) / 2.0
        # save images in grids of up to 32 images per file
        group_size = 32
        Bsaved = samples_save.shape[0]
        for g in range(0, Bsaved, group_size):
            group = samples_save[g:g+group_size]
            start_idx = idx + g
            out_path = out_dir / f"grid_{start_idx:05d}.png"
            nrow = min(8, group.shape[0])  # 8 columns -> up to 4 rows for 32 images
            save_image(group, out_path, nrow=nrow)

        # evaluate if requested
        if evaluator is not None:
            # evaluator expects images normalized to [-1,1] and on evaluator device
            # move images to evaluator device
            eval_device = next(evaluator.resnet18.parameters()).device
            imgs_for_eval = samples_clipped.to(eval_device)
            # labels: build one-hot tensor on CPU (evaluator will .cpu() internally)
            labels = torch.from_numpy(np.stack([np.pad(np.zeros(0), (0,0)) for _ in range(0)]) ) if False else None
            # we have cond already as a torch tensor on device; convert to CPU as evaluator.eval does .cpu()
            try:
                batch_lists = json_lists[idx: idx + batch_size]
                # reconstruct cond in CPU for labels
                labels = make_multihot_batch(batch_lists, obj_map, device).cpu()
            except Exception:
                labels = None
            if labels is not None:
                # accuracy
                acc = evaluator.eval(imgs_for_eval, labels)
                print(f"Batch accuracy: {acc:.4f}")
                total_acc += acc * samples.shape[0]
                total_count += samples.shape[0]

                # also print predicted labels per sample (top-k and threshold >=0.5)
                with torch.no_grad():
                    preds = evaluator.resnet18(imgs_for_eval).cpu()
                # Only print samples where the prediction is incorrect according to the evaluator's top-k metric.
                incorrect_count = 0
                for i in range(preds.shape[0]):
                    gt_indices = (labels[i] > 0).nonzero(as_tuple=False).squeeze(1).tolist()
                    if isinstance(gt_indices, int):
                        gt_indices = [gt_indices]
                    k = len(gt_indices)
                    # determine correctness using top-k when k>0
                    if k > 0:
                        topk_idx = preds[i].topk(k).indices.tolist()
                        correct = all([g in topk_idx for g in gt_indices])
                    else:
                        # if no ground-truth objects, consider correct only if model predicts none above threshold
                        thresh_idx = (preds[i] >= 0.5).nonzero(as_tuple=False).squeeze(1).tolist()
                        if isinstance(thresh_idx, int):
                            thresh_idx = [thresh_idx]
                        correct = (len(thresh_idx) == 0)

                    if not correct:
                        incorrect_count += 1
                        # prepare printable lists
                        if k > 0:
                            topk_idx = preds[i].topk(k).indices.tolist()
                        else:
                            topk_idx = []
                        thresh_idx = (preds[i] >= 0.5).nonzero(as_tuple=False).squeeze(1).tolist()
                        if isinstance(thresh_idx, int):
                            thresh_idx = [thresh_idx]
                        gt_names = [inv_obj_map[idx] for idx in gt_indices]
                        topk_names = [inv_obj_map[idx] for idx in topk_idx]
                        thresh_names = [inv_obj_map[idx] for idx in thresh_idx]
                        # false negatives (GT not in top-k)
                        false_neg = [inv_obj_map[g] for g in gt_indices if (g not in topk_idx)] if k > 0 else []
                        # false positives from thresholded preds
                        false_pos = [inv_obj_map[p] for p in thresh_idx if p not in gt_indices]
                        tqdm.write(f"ERROR Sample {idx + i:05d}: GT={gt_names} | Missed={false_neg} | FP_thresh={false_pos} | Pred_top{k}={topk_names} | Pred_thresh={thresh_names}")
                if incorrect_count > 0:
                    tqdm.write(f"Batch starting {idx:05d}: {incorrect_count}/{preds.shape[0]} incorrect")

        idx += batch_size

    if evaluator is not None and total_count > 0:
        avg_acc = total_acc / total_count
        print(f"Generated {total} samples to {out_dir}. Avg accuracy: {avg_acc:.4f}")
    else:
        print(f"Generated {total} samples to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./iclevr')
    parser.add_argument('--json_file', type=str, required=True, help='Path to test.json or new_test.json')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./samples')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--noise_file', type=str, default=None, help='Optional numpy .npy file with initial noise samples')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channel_mults', type=lambda s: [int(x) for x in s.split(',')], default=[1,2,4,8])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--time_emb_factor', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--attn_at', type=lambda s: [int(x) for x in s.split(',')], default=[2])
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--evaluate', action='store_true', help='Run evaluator on generated images')

    args = parser.parse_args()
    # normalize args lists
    if isinstance(args.channel_mults, str):
        args.channel_mults = [int(x) for x in args.channel_mults.split(',')]
    if isinstance(args.attn_at, str):
        args.attn_at = [int(x) for x in args.attn_at.split(',')]

    main(args)
