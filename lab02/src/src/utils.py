import torch

def dice_score(pred_mask, gt_mask):

    # output of unet is logits, apply sigmoid
    pred_mask = torch.sigmoid(pred_mask)

    # get binary mask
    pred_mask = (pred_mask > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()
    
    intersection = (pred_mask * gt_mask).sum()

    pred_size = pred_mask.sum()
    gt_size = gt_mask.sum()

    # Avoid division by zero
    if pred_size + gt_size == 0:
        return 1.0 
    
    dice = (2.0 * intersection) / (pred_size + gt_size)
    
    # from Tensor to Python float
    return dice.item()
