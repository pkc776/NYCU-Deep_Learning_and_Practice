import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer  


class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # --- VQGAN---
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
        for p in self.vqgan.parameters():
            p.requires_grad = False
        self.vqgan.eval()

        self.num_image_tokens     = configs['num_image_tokens']          # 256 (=16*16)
        self.num_codebook_vectors = configs['num_codebook_vectors']      # 1024
        self.mask_token_id        = self.num_codebook_vectors            # 1024 
        self.choice_temperature   = configs.get('choice_temperature', 4.5)
        self.gamma                = self.gamma_func(configs.get('gamma_type', 'cosine'))

        # --- Transformer ---
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

        # train-time buffers
        self._last_mask  = None
        self._last_ratio = None

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        _, codebook_indices, _ = self.vqgan.encode(x)
        z_indices = codebook_indices.view(-1, self.num_image_tokens).long()
        return z_indices

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1.0 - r
        elif mode == "cosine":
            return lambda r: np.cos(np.pi * r / 2.0)
        elif mode == "square":
            return lambda r: 1.0 - (r ** 2)
        else:
            raise NotImplementedError
    def forward(self, x):
        device = x.device
        z_indices = self.encode_to_z(x)  # (B, 256)
        B, N = z_indices.shape

        # get mask ratio for each sample
        r = torch.rand(B, device=device)
        mask_ratio = torch.from_numpy(self.gamma(r.cpu().numpy())).to(device)
        n_mask = torch.clamp((mask_ratio * N).ceil().long(), min=1, max=N)

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            perm = torch.randperm(N, device=device)[: n_mask[b]]
            mask[b, perm] = True

        tokens_in = z_indices.clone()
        tokens_in[mask] = self.mask_token_id

        logits = self.transformer(tokens_in)  # (B, N, vocab)

        self._last_mask = mask
        self._last_ratio = mask_ratio
        return logits, z_indices, mask

    @staticmethod
    def _gumbel(shape, device):
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + 1e-9) + 1e-9)

    # single step decoding
    @torch.no_grad()
    def inpainting(self, z_indices, mask, ratio, orig_mask_count):
        device = z_indices.device
        B, N = z_indices.shape

        # replace mask=True with mask_token_id
        tokens_in = z_indices.clone()
        tokens_in[mask] = self.mask_token_id
        logits = self.transformer(tokens_in)
        logits[..., self.mask_token_id] = -1e9

        # softmax → probability, prediction
        probs = torch.softmax(logits, dim=-1)
        z_prob, z_pred = probs.max(dim=-1)  # (B,N)

        # Gumbel-log-p confidence
        if not torch.is_tensor(ratio):
            ratio = torch.tensor(float(ratio), device=device)
        tau = self.choice_temperature * (1.0 - ratio).clamp(min=0.0)
        g = self._gumbel(z_prob.shape, device=device)
        confidence = torch.log(z_prob + 1e-9) + tau * g

        # only consider original mask region
        confidence = confidence.masked_fill(~mask, -float('inf'))

        # remain_mask = ceil(gamma(ratio) * orig_mask_count)
        remain_mask = int(math.floor(self.gamma(ratio.item()) * orig_mask_count))
        remain_mask = max(0, min(orig_mask_count, remain_mask))

        # calculate current mask count
        current_mask_count = int(mask.sum().item())

        # number of tokens to be unmasked = current_mask_count - remain_mask
        n_unmask = max(0, current_mask_count - remain_mask)

        # select top-n_unmask tokens in confidence
        keep_mask = torch.zeros_like(mask)  # (B,N)
        for b in range(B):
            if n_unmask <= 0:
                continue
            topk_idx = torch.topk(confidence[b], k=n_unmask, largest=True).indices
            keep_mask[b, topk_idx] = True
        
        # update z_indices: only keep_mask=True tokens are replaced with prediction
        z_next = z_indices.clone()
        z_next[mask] = z_pred[mask]

        return z_next, keep_mask




__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
