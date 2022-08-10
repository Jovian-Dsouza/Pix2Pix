import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        h, w, _ = image.shape
        input_image = image[:, :w//2, :]
        target_image = image[:, w//2:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    import os
    import shutil

    dataset = MapDataset("old_new_dataset")
    x, y = dataset.__getitem__(4)
    save_image(x*0.5+0.5, "x.png")
    save_image(y*0.5+0.5, "y.png")
    print(x.shape, y.shape)
    # loader = DataLoader(dataset, batch_size=4)
    # for x, y in loader:
    #     print(x.shape)
    #     save_image(x, "x.png")
    #     save_image(y, "y.png")
    #     import sys

    #     sys.exit()
