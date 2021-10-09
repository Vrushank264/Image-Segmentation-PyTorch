import torch
import torch.nn as nn
from tqdm import tqdm
from Unet import UNet
from utils import *
import config 


def train(loader, model, opt, criterion, scaler):
    
    loop = tqdm(loader, position = 0, leave = True)
    for idx, (image, mask) in enumerate(loop):
        
        image = image.to(config.DEVICE)
        mask = mask.to(config.DEVICE)
        
        with torch.cuda.amp.autocast:
            
            preds = model(image)
            loss = criterion(preds, mask)
        
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        loop.set_postfix(loss = loss.item())
        

def main():
    
    train_loader, val_loader = get_loader(config.IMAGE_DIR, config.MASK_DIR, 
                                          config.VALID_IMAGE_DIR, config.VALID_MASK_DIR,
                                          config.BATCH_SIZE)
    model = UNet(num_classes = 1).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.oprim.Adam(model.parameters(), lr = config.LR)
    scaler = torch.cuda.amp.GradScaler()
    
    
    for epoch in range(1, config.NUM_EPOCHS):
        
        print(f"\nEpoch: {epoch}\n",)
        train(train_loader, model, opt, criterion, scaler)
        print("\nChecking Training Accuracy...\n")
        check_acc(train_loader, model)
        print("\nChecking Validation Accuracy...\n")
        check_acc(val_loader, model)
        torch.save(model.state_dict(), open(config.MODEL_SAVE_DIR + 'unet_model.pth', 'wb'))
        

if __name__ == '__main__':
    
    main()
