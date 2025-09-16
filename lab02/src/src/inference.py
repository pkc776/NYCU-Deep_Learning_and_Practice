import argparse
import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import SimpleOxfordPetDataset
from utils import dice_score


def load_model(model_path, model_type='unet', device='cpu'):
    """Load trained model"""
    print(f"Loading model from {model_path}")
    
    # Create model
    if model_type == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    elif model_type == 'resnet34_unet':
        model = ResNet34UNet(in_channels=3, num_classes=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch'] + 1}")
    
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    test_dice = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    all_losses = []
    all_dice_scores = []
    
    print("Evaluating model on test set...")
    
    # TensorBoard writer (logdir: runs/inference_DATE_TIME)
    from datetime import datetime
    log_dir = os.path.join("runs", f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log directory: {log_dir}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            test_loss += loss.item()
            dice = dice_score(outputs, masks)
            test_dice += dice
            
            # Store individual scores for logging
            all_losses.append(loss.item())
            all_dice_scores.append(dice)

            # Log sample images to TensorBoard (only for the first batch)
            if batch_idx == 0:
                num_images = min(4, images.size(0))
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
                denormalized_images = (images[:num_images] * std) + mean
                display_images = torch.clamp(denormalized_images, 0, 1)
                writer.add_images('Test/Images', display_images, 0)
                writer.add_images('Test/Ground_Truth', masks[:num_images], 0)
                pred_probs = torch.sigmoid(outputs[:num_images])
                writer.add_images('Test/Predictions', pred_probs, 0)
                pred_binary = (pred_probs > 0.5).float()
                writer.add_images('Test/Binary_Predictions', pred_binary, 0)

            # Log per-batch metrics
            writer.add_scalar('Test/Loss', loss.item(), batch_idx)
            writer.add_scalar('Test/Dice_Score', dice, batch_idx)

    avg_loss = test_loss / len(test_loader)
    avg_dice = test_dice / len(test_loader)

    # Log average metrics to TensorBoard
    writer.add_scalar('Test/Average_Loss', avg_loss)
    writer.add_scalar('Test/Average_Dice_Score', avg_dice)
    writer.add_scalar('Test/Loss_Std', float(np.std(all_losses)))
    writer.add_scalar('Test/Dice_Score_Std', float(np.std(all_dice_scores)))
    writer.close()

    print(f"Test Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Average Dice Score: {avg_dice:.4f}")
    print(f"TensorBoard logs saved to: {log_dir}")
    
    return avg_loss, avg_dice, all_losses, all_dice_scores


    # Plotting removed; use TensorBoard for visualization


def get_args():
    parser = argparse.ArgumentParser(description='Test trained model on test set')
    parser.add_argument('--model', required=True, help='Path to the stored model weight (.pth file)')
    parser.add_argument('--data_path', type=str, default='./dataset/oxford-iiit-pet', 
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', '-b', type=int, default=8, 
                        help='Batch size for testing')
    parser.add_argument('--model_type', type=str, default='unet', 
                        choices=['unet', 'resnet34_unet'], 
                        help='Type of model architecture')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        print("Available models in saved_models/:")
        if os.path.exists("saved_models"):
            for f in os.listdir("saved_models"):
                if f.endswith('.pth'):
                    print(f"  - saved_models/{f}")
        exit(1)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print(f"Loading test dataset from: {args.data_path}")
    test_dataset = SimpleOxfordPetDataset(
        root=args.data_path,
        mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    print(f"Test dataset loaded: {len(test_dataset)} images")
    
    # Load model
    model = load_model(args.model, args.model_type, device)
    
    # Evaluate model
    avg_loss, avg_dice, all_losses, all_dice_scores = evaluate_model(model, test_loader, device)
    
    # Results are logged to TensorBoard; no matplotlib plots
    
    print(f"\nInference completed successfully!")
    print(f"Test set size: {len(test_dataset)} images")
    print(f"Final Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Average Dice Score: {avg_dice:.4f}")
    print(f"  Loss std: {np.std(all_losses):.4f}")
    print(f"  Dice Score std: {np.std(all_dice_scores):.4f}")