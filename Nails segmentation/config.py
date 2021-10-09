import torch

IMAGE_DIR = 'E:/Computer Vision/Segmentation/Data/Train/Images'
MASK_DIR = 'E:/Computer Vision/Segmentation/Data/Train/labels'
VALID_IMAGE_DIR = 'E:/Computer Vision/Segmentation/Data/Validation/Images'
VALID_MASK_DIR = 'E:/Computer Vision/Segmentation/Data/Validation/labels'
MODEL_SAVE_DIR = 'E:/Computer Vision/Segmentation'
LR = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 10
DEVICE = torch.device('cuda')