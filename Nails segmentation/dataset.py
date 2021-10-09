import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2


class SegDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, transform = None):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
                                                                                                                                                
        return len(self.images)
    
    def __getitem__(self, index):
        
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype = np.float32)
        image = cv2.resize(image, (388, 388), interpolation = cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (388,388), interpolation = cv2.INTER_CUBIC)
        image = np.pad(image, ((84, 100),(100,84),(0,0)), mode = 'reflect')
        mask[mask == 255.0] = 1.0
        
        if self.transform is not None:
            
            aug = self.transform(image = image, mask = mask)
            image = aug["image"]
            mask = aug["mask"]
            
        return image, mask
            
            