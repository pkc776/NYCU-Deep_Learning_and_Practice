import os
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

# TODO2 step1-4
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(args.device)
        self.optimizer, self.scheduler = self.configure_optimizers()
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def configure_optimizers(self):
        lr = self.args.learning_rate if self.args.learning_rate > 0 else 4e-4
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=lr, betas=(0.9, 0.96))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.args.epochs), eta_min=lr * 0.01
        )
        return optimizer, scheduler

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        device = self.args.device
        pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)

        total_loss, total_correct, total_mask = 0.0, 0, 0
        accum = max(1, self.args.accum_grad)
        self.optimizer.zero_grad(set_to_none=True)

        for step, imgs in enumerate(pbar, 1):
            imgs = imgs.to(device, non_blocking=True)
            logits, tgt, mask = self.model(imgs)              # logits: (B,N,vocab)
            B, N, K = logits.shape
            
            # Debug prints
            # print(f"logits shape: {logits.shape}")
            # print(f"tgt shape: {tgt.shape}")
            # print(f"mask shape: {mask.shape}")
            # print(f"logits.view(-1, K) shape: {logits.view(-1, K).shape}")
            # print(f"tgt.view(-1) shape: {tgt.view(-1).shape}")

            loss_all = F.cross_entropy(logits.view(-1, K), tgt.view(-1), reduction="none")
            mask_flat = mask.view(-1).float()
            loss = (loss_all * mask_flat).sum() / torch.clamp(mask_flat.sum(), min=1.0)

            (loss / accum).backward()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct = ((pred == tgt) & mask).sum().item()
                total_correct += correct
                total_mask += int(mask.sum().item())

            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.transformer.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{total_loss/step:.4f}",
                acc=f"{(total_correct/max(1,total_mask))*100:.2f}%"
            )

        return {"loss": total_loss / max(1, step),
                "acc": total_correct / max(1, total_mask)}

    @torch.no_grad()
    def eval_one_epoch(self, loader, epoch):
        self.model.eval()
        device = self.args.device
        pbar = tqdm(loader, desc=f"Val {epoch}", leave=False)
        total_loss, total_correct, total_mask = 0.0, 0, 0

        for step, imgs in enumerate(pbar, 1):
            imgs = imgs.to(device, non_blocking=True)
            logits, tgt, mask = self.model(imgs)
            B, N, K = logits.shape
            loss_all = F.cross_entropy(logits.view(-1, K), tgt.view(-1), reduction="none")
            mask_flat = mask.view(-1).float()
            loss = (loss_all * mask_flat).sum() / torch.clamp(mask_flat.sum(), min=1.0)

            pred = logits.argmax(dim=-1)
            correct = ((pred == tgt) & mask).sum().item()
            total_correct += correct
            total_mask += int(mask.sum().item())
            total_loss += loss.item()

            pbar.set_postfix(
                loss=f"{total_loss/step:.4f}",
                acc=f"{(total_correct/max(1,total_mask))*100:.2f}%"
            )

        return {"loss": total_loss / max(1, step),
                "acc": total_correct / max(1, total_mask)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    # TODO2: check your dataset path is correct
    parser.add_argument('--train_d_path', type=str, default="./dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=0.1, help='Number of epochs to train (default: 50)')
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    # you can modify the hyperparameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    trainer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True,
                              shuffle=True)

    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)

    # TODO2 step1-5
    best = float("inf")
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        tr = trainer.train_one_epoch(train_loader, epoch)
        va = trainer.eval_one_epoch(val_loader, epoch)

        print(f"[Epoch {epoch}] train_loss={tr['loss']:.4f} acc={tr['acc']*100:.2f}% | "
              f"val_loss={va['loss']:.4f} acc={va['acc']*100:.2f}%")

        if epoch % args.save_per_epoch == 0:
            torch.save(trainer.model.transformer.state_dict(),
                       os.path.join("transformer_checkpoints", f"epoch_{epoch:04d}.pt"))
        if va["loss"] < best:
            best = va["loss"]
            torch.save(trainer.model.transformer.state_dict(),
                       os.path.join("transformer_checkpoints", "best.pt"))
