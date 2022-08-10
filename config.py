from albumentations import augmentations
from albumentations.augmentations import geometric
from albumentations.augmentations.geometric.functional import longest_max_size
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "old_new_dataset"
VAL_DIR = "old_new_dataset"
EVAL_DIR = "evaluations"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 0
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [
        A.augmentations.geometric.resize.LongestMaxSize(IMAGE_SIZE),
        A.augmentations.transforms.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0), 
    ],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        # A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
