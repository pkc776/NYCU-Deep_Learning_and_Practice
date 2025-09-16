import os
import torch
import shutil
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # ImageNet normalization preprocessing
        # Step 1: Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Step 2: Apply ImageNet mean and std normalization
        # ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Normalize each channel: (pixel - m   ean) / std
        image = (image - imagenet_mean) / imagenet_std

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class AugmentedOxfordPetDataset(SimpleOxfordPetDataset):
    """
    Oxford Pet dataset with data augmentation.
    Uses torchvision transforms for simple but effective augmentation.
    """
    def __init__(self, root, mode="train", augment=True):
        super().__init__(root, mode)
        self.augment = augment and mode == "train"  # Only apply augmentation during training
        
        # Define data augmentation transforms
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.1
            )
    
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        
        if self.augment:
            # Convert tensor back to PIL Image for augmentation
            # First, denormalize
            image_tensor = torch.from_numpy(sample["image"])
            mask_tensor = torch.from_numpy(sample["mask"])
            
            # Denormalize to [0, 1] range
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_denorm = image_tensor * imagenet_std + imagenet_mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            
            # Apply random transforms
            # 1. Random horizontal flip (50% chance)
            if random.random() > 0.5:
                image_denorm = TF.hflip(image_denorm)
                mask_tensor = TF.hflip(mask_tensor)
            
            # 2. Random vertical flip (20% chance)
            if random.random() > 0.8:
                image_denorm = TF.vflip(image_denorm)
                mask_tensor = TF.vflip(mask_tensor)
            
            # 3. Random rotation (-15 to 15 degrees, 30% chance)
            if random.random() > 0.7:
                angle = random.uniform(-15, 15)
                image_denorm = TF.rotate(image_denorm, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_tensor = TF.rotate(mask_tensor, angle, interpolation=TF.InterpolationMode.NEAREST)
            
            # 4. Color jitter (40% chance)
            if random.random() > 0.6:
                image_denorm = self.color_jitter(image_denorm)
            
            # 5. Random Gaussian blur (20% chance)
            if random.random() > 0.8:
                kernel_size = random.choice([3, 5])
                sigma = random.uniform(0.1, 1.0)
                image_denorm = TF.gaussian_blur(image_denorm, kernel_size, sigma)
            
            # 6. Random affine transform (25% chance)
            if random.random() > 0.75:
                # Slight translation and scaling
                translate = (random.uniform(-0.1, 0.1) * 256, random.uniform(-0.1, 0.1) * 256)
                scale = random.uniform(0.9, 1.1)
                shear = random.uniform(-5, 5)
                
                image_denorm = TF.affine(
                    image_denorm, 
                    angle=0, 
                    translate=translate, 
                    scale=scale, 
                    shear=shear,
                    interpolation=TF.InterpolationMode.BILINEAR
                )
                mask_tensor = TF.affine(
                    mask_tensor, 
                    angle=0, 
                    translate=translate, 
                    scale=scale, 
                    shear=shear,
                    interpolation=TF.InterpolationMode.NEAREST
                )
            
            # Re-normalize
            image_norm = (image_denorm - imagenet_mean) / imagenet_std
            
            # Update sample
            sample["image"] = image_norm.numpy()
            sample["mask"] = mask_tensor.numpy()
        
        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode, augment=False):
    # Implement the load dataset function here
    if not os.path.exists(os.path.join(data_path, "images")):
        OxfordPetDataset.download(data_path)

    # groud truth annotations
    if not os.path.exists(os.path.join(data_path, "annotations")):
        OxfordPetDataset.download(data_path)

    if augment and mode == "train":
        dataset = AugmentedOxfordPetDataset(
            root=data_path,
            mode=mode,
            augment=True
        )
    else:
        dataset = SimpleOxfordPetDataset(
            root=data_path,
            mode=mode
        )

    return dataset