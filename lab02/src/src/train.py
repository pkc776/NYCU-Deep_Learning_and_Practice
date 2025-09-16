import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from oxford_pet import load_dataset
from utils import dice_score
from evaluate import evaluate

def train(args):
    # uses GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device : {device}")
    
    # TensorBoard settings
    log_dir = os.path.join("runs", f"{args.model}_lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log directory: {log_dir}")

    # Load dataset
    print("Loading training data...")
    # Use data augmentation to reduce overfitting
    use_augmentation = getattr(args, 'augment', True)  # Default to True
    if use_augmentation:
        print("Data augmentation enabled")
    else:
        print("Data augmentation disabled")

    train_dataset = load_dataset(args.data_path, mode="train", augment=use_augmentation)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )

    print("Loading validation data...")
    val_dataset = load_dataset(args.data_path, mode="valid", augment=False)  # Validation does not use augmentation
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2
    )

    # Build model
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=1)
        print("Using UNet model")
    elif args.model == 'resnet34_unet':
        model = ResNet34UNet(in_channels=3, num_classes=1)
        print("Using ResNet34+UNet model")

    model = model.to(device)

    # Log model architecture to TensorBoard
    sample_input = torch.randn(1, 3, 256, 256).to(device)
    writer.add_graph(model, sample_input)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Suitable for logits output
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Define learning rate scheduler
    scheduler = None

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"Using CosineAnnealingLR scheduler: T_max={args.epochs}")
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, patience=args.patience)
        print(f"Using ReduceLROnPlateau scheduler: factor={args.gamma}, patience={args.patience}")
    elif args.scheduler == 'none':
        print("Not using learning rate scheduler")

    print(f"Starting training for {args.epochs} epochs...")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].float().to(device)
                masks = batch['mask'].float().to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Calculate metrics
                train_loss += loss.item()
                dice = dice_score(outputs.detach(), masks.detach())
                train_dice += dice

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice:.4f}'
                })

        # Calculate average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        # Log training metrics to TensorBoard
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Dice_Score', avg_train_dice, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation phase - call evaluate function (pass in writer and epoch)
        avg_val_loss, avg_val_dice = evaluate(model, val_loader, device, writer, epoch, "val")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(avg_val_loss)  # ReduceLROnPlateau needs monitored metric
            else:
                scheduler.step()  # CosineAnnealingLR do not need metric

        # Print results
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        if scheduler is not None:
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 50)

        # Save model checkpoint
        if epoch == args.epochs - 1:
            # Ensure saved_models directory exists
            os.makedirs("saved_models", exist_ok=True)

            # Generate checkpoint name with parameter information
            scheduler_name = args.scheduler if args.scheduler != 'none' else 'no_scheduler'
            checkpoint_name = f"checkpoint_{args.model}_ep{epoch+1}_lr{args.learning_rate}_bs{args.batch_size}_{scheduler_name}.pth"
            checkpoint_path = os.path.join("saved_models", checkpoint_name)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice,
                'args': vars(args) 
            }, checkpoint_path)
            print(f"Model checkpoint saved: {checkpoint_path}")

    # Close TensorBoard writer
    writer.close()
    print("Training complete!")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='./dataset/oxford-iiit-pet', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model', type=str, default='resnet34_unet', choices=['unet', 'resnet34_unet'], 
                        help='choose model architecture')
    
    # Data augmentation parameters
    parser.add_argument('--augment', action='store_true', default=True,
                        help='enable data augmentation to reduce overfitting')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='disable data augmentation')

    # Learning rate scheduler related parameters
    parser.add_argument('--scheduler', type=str, default='none',
                        choices=['none', 'step', 'cosine', 'plateau'],
                        help='learning rate scheduler')
    parser.add_argument('--patience', type=int, default=5, 
                        help='patience for ReduceLROnPlateau scheduler')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)