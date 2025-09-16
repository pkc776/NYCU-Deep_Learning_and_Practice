import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import dice_score

def evaluate(net, data_loader, device, writer=None, epoch=None, phase="val"):
    """
    Evaluate model performance on the given data loader
    
    Args:
        net: model
        data_loader: data loader
        device: computation device
        writer: TensorBoard SummaryWriter (optional)
        epoch: current epoch (optional)
        phase: phase name, e.g. "val", "test" (default: "val")
    
    Returns:
        tuple: (avg_loss, avg_dice)
    """
    net.eval()
    val_loss = 0.0
    val_dice = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().to(device)
            
            outputs = net(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            dice = dice_score(outputs, masks)
            val_dice += dice
            
            # Log sample images to TensorBoard (only for the first batch)
            if writer is not None and epoch is not None and batch_idx == 0:
                # Take the first 4 images (or fewer if batch is smaller)
                num_images = min(4, images.size(0))
                
                # Reverse ImageNet normalization for display
                # ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
                
                # Denormalize: pixel = (normalized_pixel * std) + mean
                denormalized_images = (images[:num_images] * std) + mean
                # Ensure values are in [0, 1] range
                display_images = torch.clamp(denormalized_images, 0, 1)
                
                # Log original images
                writer.add_images(f'{phase}/Images', display_images, epoch)
                
                # Log ground truth masks
                writer.add_images(f'{phase}/Ground_Truth', masks[:num_images], epoch)
                
                # Log prediction results (as probabilities)
                pred_probs = torch.sigmoid(outputs[:num_images])
                writer.add_images(f'{phase}/Predictions', pred_probs, epoch)
                
                # Log binary prediction results
                pred_binary = (pred_probs > 0.5).float()
                writer.add_images(f'{phase}/Binary_Predictions', pred_binary, epoch)
    
    avg_val_loss = val_loss / len(data_loader)
    avg_val_dice = val_dice / len(data_loader)
    
    # Log average metrics to TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar(f'{phase}/Loss', avg_val_loss, epoch)
        writer.add_scalar(f'{phase}/Dice_Score', avg_val_dice, epoch)
    
    return avg_val_loss, avg_val_dice 