import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader 
import albumentation as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import SegDataset
import config


def get_loader(img_dir, mask_dir, val_dir, val_maskdir, batch_size):
    
    train_tr = A.Compose([
            A.Rotate(limit = 25, p = 0.25),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p = 0.25),
            A.Normalize(mean = [0.0, 0.0, 0.0],
                        std = [1.0, 1.0, 1.0],
                        max_pixel_value = 255.0),
            ToTensorV2()])
    
    val_tr = A.Compose([
            A.Normalize(mean = [0.0, 0.0, 0.0],
                        std = [1.0, 1.0, 1.0],
                        max_pixel_value = 255.0),
            ToTensorV2()])
    
    train_data = SegDataset(image_dir = img_dir, mask_dir = mask_dir, transform = train_tr)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers = 2, shuffle = True)
    val_data = SegDataset(image_dir = val_dir, mask_dir = val_maskdir, transform = val_tr)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers = 2, shuffle = False)
    
    return train_loader, val_loader


def check_acc(loader, model, device = torch.device('cuda')):
    
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
        
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()
            num_correct += (pred == y).sum()
            num_pixels += torch.numel(pred)
            dice_score += (2 * (pred * y).sum()) / ((pred + y).sum() + 1e-8)
            
    print(f'Got {num_correct}/{num_pixels} with acc {(num_correct/num_pixels) * 100:.3f}.')
    print(f'Dice Score: {dice_score/len(loader)}')
    model.train()
    
    
def save_preds(loader, model, save_dir, device = torch.device('cuda')):
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        vutils.save_image(y.unsqueeze(1), f'{save_dir}/real{idx}.png')
        vutils.save_image(preds, f'{save_dir}/segmented{idx}.png')
        grid = vutils.make_grid([y.unsqueeze(1), preds], padding = 2, nrows = 2, normalize = True)
        grid = grid.permute(1,2,0)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.imshow(grid)
        grid = np.asarray(grid)
        plt.imsave(open(config.SAVE_DIR) + '/Result.png', grid,  format = 'png')
        